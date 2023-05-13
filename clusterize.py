import warnings

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# Filter out the warning message for ShardedTensor
warnings.filterwarnings("ignore", category=UserWarning, message="Please use DTensor instead and we are deprecating "
                                                                "ShardedTensor.")

import os

import click
import torch
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader.collate import collate_fn
from dataset.fonts_dataset import FontsDataset
from dataset.iam_dataset import IAMDataset
from dataset.iam_resized_dataset import IAMResizedDataset
from dataset.cvl_resized_dataset import CVLResizedDataset
from model.supervised_encoder import SupervisedEncoder

import sklearn.cluster
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import umap

import kmeans_gpu


@click.command()
@click.option("--cluster-config-path", type=click.Path(exists=True), required=True, help="Path to clusterization configuration")
@click.option("--visualize", is_flag=True, help="Visualize")
def main(cluster_config_path: click.Path, visualize: bool):
    with open(str(cluster_config_path), "r") as config_file:
        test_config = yaml.safe_load(config_file)

    model_config_path = test_config["model_config_path"]
    with open(str(model_config_path), "r") as config_file:
        model_config = yaml.safe_load(config_file)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    print("Initializing dataset")
    debug = test_config["debug"]
    dataset_root = test_config["dataset"]["root_path"]
    dataset_type = test_config["dataset"]["type"]
    if dataset_type == "synthetic":
        test_fonts_file = "fonts_debug.json" if debug else test_config["dataset"]["fonts_test"]
        test_words = os.path.join(dataset_root, test_config["dataset"]["words_test"])
        test_fonts = os.path.join(dataset_root, test_fonts_file)
        test_dataset = FontsDataset(dataset_root, test_words, test_fonts, transform)
    elif dataset_type == "iam":
        test_dataset = IAMDataset(dataset_root, transform)
    elif dataset_type == "iam_resized":
        test_dataset = IAMResizedDataset(dataset_root, transform)
    elif dataset_type == "cvl_resized":
        test_dataset = CVLResizedDataset(dataset_root, transform)
    else:
        raise Exception(f"Dataset unknown type: {dataset_type}")

    print("Initializing dataloader")
    batch_size = test_config["dataloader"]["batch_size"]
    num_workers = test_config["dataloader"]["num_workers"]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)

    print("Initializing model")
    if test_config["model"] == "encoder":
        encoder = SupervisedEncoder.load_from_checkpoint(test_config["checkpoint_path"], config=model_config, map_location=torch.device('mps'))
    else:
        raise Exception(f"Invalid model: {test_config['model']}")
    encoder.eval()
    encoder.to("mps")

    print("Creating embeddings")
    embeddings_path = test_config["embedding_path"]
    if os.path.exists(embeddings_path):
        embeddings = torch.load(embeddings_path)
        print("Loaded embeddings from cache")
    else:
        print("No cached embeddings found")
        embeddings = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Getting embeddings"):
                image_batch = batch["image_tensor"]
                image_batch = image_batch.to("mps")
                embeddings.append(encoder(image_batch))

        embeddings = torch.cat(embeddings)
        torch.save(embeddings, embeddings_path)
    embeddings = embeddings.detach().cpu().numpy()
    
    print("Embeddings shape:", embeddings.shape)

    print("Applying UMAP...")
    umap_config = test_config["umap"]
    if visualize:
        umap_config["n_components"] = 2

    reducer = umap.UMAP(**umap_config)
    reducer.fit(embeddings)
    embeddings_umap = reducer.transform(embeddings)
    print("UMAP result shape:", embeddings_umap.shape)

    if visualize:
        data_viz = {
            "x": embeddings_umap[:, 0],
            "y": embeddings_umap[:, 1],
        }
        df = pd.DataFrame(data=data_viz)
        sns.scatterplot(data=df, x="x", y="y")
        plt.show()
        return

    s_scores = []
    min_samples = test_config["cluster"]["min_samples"] 
    max_samples = test_config["cluster"]["max_samples"]
    step = test_config["cluster"]["step"]
    n_samples_range = list(range(min_samples, max_samples, step))
    p_bar = tqdm(n_samples_range, desc="Testing silhouette")
    for n_samples in p_bar:
        if test_config["clustering_method"] == "kmeans_gpu":
            kmeans = kmeans_gpu.KMeans(n_clusters=n_samples, **test_config["clustering_params"])
            embeddings_torch = torch.from_numpy(embeddings_umap)
            labels, _ = kmeans.fit_predict(embeddings_torch)
            labels = labels.detach().cpu().numpy()
        else:
            clustering = getattr(sklearn.cluster, test_config["clustering_method"])(
                n_clusters=n_samples, **test_config["clustering_params"]
            ).fit(embeddings_umap)
            labels = clustering.labels_

        s_scores.append(silhouette_score(embeddings_umap, labels, **test_config["silhouette_params"]))
        p_bar.set_description(f"Testing silhouette: {n_samples} samples -> {s_scores[-1]}")

    data_s = {
        "s_x": n_samples_range,
        "s_y": s_scores,
    }
    df_s = pd.DataFrame(data=data_s)

    sns.lineplot(data=df_s, x="s_x", y="s_y")
    plt.savefig(test_config["plot_output_path"])


if __name__ == "__main__":
    main()
