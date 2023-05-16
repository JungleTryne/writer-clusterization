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
from torch.nn.functional import normalize
from torchvision import transforms

from dataloader.collate import collate_fn
from dataset.fonts_dataset import FontsDataset
from dataset.iam_dataset import IAMDataset
from dataset.iam_resized_dataset import IAMResizedDataset
from dataset.cvl_resized_dataset import CVLResizedDataset
from model.supervised_encoder import SupervisedEncoder

import sklearn.cluster
from sklearn.metrics import silhouette_score, calinski_harabasz_score, rand_score, adjusted_rand_score
from sklearn import preprocessing

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import umap

import kmeans_gpu


DEVICE = "cuda"


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
    cut = test_config["dataset"]["cut"] if "cut" in test_config["dataset"] else None
    if dataset_type == "synthetic":
        test_fonts_file = "fonts_debug.json" if debug else test_config["dataset"]["fonts_test"]
        test_words = os.path.join(dataset_root, test_config["dataset"]["words_test"])
        test_fonts = os.path.join(dataset_root, test_fonts_file)
        test_dataset = FontsDataset(dataset_root, test_words, test_fonts, transform)
    elif dataset_type == "iam":
        test_dataset = IAMDataset(dataset_root, transform)
    elif dataset_type == "iam_resized":
        test_dataset = IAMResizedDataset(dataset_root, transform, cut=cut)
    elif dataset_type == "cvl_resized":
        test_dataset = CVLResizedDataset(dataset_root, transform, cut=cut)
    else:
        raise Exception(f"Dataset unknown type: {dataset_type}")

    print("Initializing dataloader")
    batch_size = test_config["dataloader"]["batch_size"]
    num_workers = test_config["dataloader"]["num_workers"]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)

    print("Initializing model")
    if test_config["model"] == "encoder":
        encoder = SupervisedEncoder.load_from_checkpoint(test_config["checkpoint_path"], config=model_config, map_location=torch.device(DEVICE))
    else:
        raise Exception(f"Invalid model: {test_config['model']}")
    encoder.eval()
    encoder.to(DEVICE)

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
                image_batch = image_batch.to(DEVICE)
                embeddings.append(encoder(image_batch))

        embeddings = torch.cat(embeddings)
        torch.save(embeddings, embeddings_path)
        
    embeddings = embeddings.detach().cpu().numpy()
    
    print("Embeddings shape:", embeddings.shape)

    if "umap" not in test_config:
        embeddings_umap = embeddings
    else:
        print("Applying UMAP...")
        umap_embeddings_path = embeddings_path + ".umap"
        umap_config = test_config["umap"]

        if visualize:
            raise Exception("It is broken with cache")
            umap_config["n_components"] = 2

        if os.path.exists(umap_embeddings_path):
            embeddings_umap = torch.load(umap_embeddings_path)
            print("Loaded UMAP embeddings from cache")
        else:
            reducer = umap.UMAP(**umap_config)
            reducer.fit(embeddings)
            embeddings_umap = reducer.transform(embeddings)
            embeddings_umap = preprocessing.StandardScaler().fit_transform(embeddings_umap)
            torch.save(embeddings_umap, umap_embeddings_path)
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

    if test_config["clustering_method"] == "MeanShift":
        print("{} clustring method determines the number of clusters itself".format(test_config["clustering_method"]))
        clustering = getattr(sklearn.cluster, test_config["clustering_method"])(**test_config["clustering_params"]).fit(embeddings_umap)
        print(f"Number of clusters: {clustering.labels_.max()}")
        score = silhouette_score(embeddings_umap, clustering.labels_, **test_config["silhouette_params"])
        print(f"Silhouette score: {score}")
        return

    s_scores = []
    c_scores = []
    r_scores = []
    ra_scores = []

    labels_true = test_dataset.get_labels()

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
        c_scores.append(calinski_harabasz_score(embeddings_umap, labels))
        r_scores.append(rand_score(labels_true, labels))
        ra_scores.append(adjusted_rand_score(labels_true, labels))

        p_bar.set_description(f"Testing metrics: {n_samples} samples -> {s_scores[-1]} {c_scores[-1]} {r_scores[-1]}")

    data = {
        "clusters_num": n_samples_range,
        "silhouette": s_scores,
        "calinski_harabasz": c_scores,
        "rand": r_scores,
        "rand_adjusted": ra_scores,
    }
    df = pd.DataFrame(data=data)

    sns.lineplot(data=df_s, x="clusters_num", y="silhouette").set(title='Silhouette score')
    plt.axvline(x=test_config["cluster"]["correct"], color="r", alpha=0.5, linestyle="--")
    plt.savefig("silhouette_" + test_config["plot_output_path"])

    sns.lineplot(data=df_s, x="clusters_num", y="calinski_harabasz").set(title='Calinski-Harabasz score')
    plt.axvline(x=test_config["cluster"]["correct"], color="r", alpha=0.5, linestyle="--")
    plt.savefig("calinski_harabasz_" + test_config["plot_output_path"])

    sns.lineplot(data=df_s, x="clusters_num", y="rand").set(title='Rand score')
    plt.axvline(x=test_config["cluster"]["correct"], color="r", alpha=0.5, linestyle="--")
    plt.savefig("rand_" + test_config["plot_output_path"])

    sns.lineplot(data=df_s, x="clusters_num", y="rand_adjusted").set(title='Adjusted Rand score')
    plt.axvline(x=test_config["cluster"]["correct"], color="r", alpha=0.5, linestyle="--")
    plt.savefig("rand_" + test_config["plot_output_path"])


if __name__ == "__main__":
    main()
