import warnings

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# Filter out the warning message for ShardedTensor
warnings.filterwarnings("ignore", category=UserWarning, message="Please use DTensor instead and we are deprecating "
                                                                "ShardedTensor.")

import os

import numpy as np
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
from model.snn import SiameseNN

import sklearn.cluster
from sklearn.metrics import silhouette_score, calinski_harabasz_score, rand_score, adjusted_rand_score
from sklearn import preprocessing
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import umap

import kmeans_gpu


DEVICE = "cuda"


def get_dataset(test_config):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

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
    return test_dataset


def get_model(test_config):
    model_config_path = test_config["model_config_path"]
    with open(str(model_config_path), "r") as config_file:
        model_config = yaml.safe_load(config_file)

    if model_config["model"] == "encoder":
        encoder = SupervisedEncoder.load_from_checkpoint(
            model_config["checkpoint_path"], 
            config=model_config, 
            map_location=torch.device(DEVICE)
        )
    elif model_config["model"] == "snn":
        encoder = SiameseNN.load_from_checkpoint(
            model_config["checkpoint_path"], 
            config=model_config, 
            map_location=torch.device(DEVICE), 
            train_dataset=None, 
            val_dataset=None
        )
    else:
        raise Exception(f"Invalid model: {model_config['model']}")
    encoder.eval()
    encoder.to(DEVICE)
    return encoder


def get_full_embeddings(test_config, test_loader, encoder, run_name):
    embeddings_path = os.path.join(test_config["embedding_path"], f"{run_name}.pt")
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
    return embeddings


def apply_umap(test_config, embeddings, test_dataset, run_name):
    umap_embeddings_path = os.path.join(test_config["embedding_path"], f"{run_name}.umap")
    umap_config = test_config["umap"]

    if os.path.exists(umap_embeddings_path) and False:
        embeddings = torch.load(umap_embeddings_path)
        print("Loaded UMAP embeddings from cache")
    else:
        print("Training UMAP")
        labels = test_dataset.get_labels()
        reducer = umap.UMAP(**umap_config)

        if "umap_training" in test_config:
            umap_alpha = test_config["umap_training"]["alpha"]
            index = np.random.choice(labels.shape[0], int(labels.shape[0] * umap_alpha), replace=False)  
            umap_train_embedd = embeddings[index]
            umap_train_labels = labels[index]

            print("UMAP Training...")        
            reducer.fit(umap_train_embedd, umap_train_labels)
        else:
            print("UMAP fitting without training...")
            reducer.fit(embeddings)

        print("Transforming data")
        embeddings = reducer.transform(embeddings)
        torch.save(embeddings, umap_embeddings_path)
        print("UMAP result shape:", embeddings.shape)

    return embeddings


def evaluate(test_config, embeddings, test_dataset):
    s_scores = []
    c_scores = []
    r_scores = []
    ra_scores = []

    labels_true = test_dataset.get_labels()

    min_samples = test_config["cluster"]["min_samples"] 
    max_samples = test_config["cluster"]["max_samples"]
    step = test_config["cluster"]["step"]
    n_samples_range = list(range(min_samples, max_samples + 1, step))
    p_bar = tqdm(n_samples_range, desc="Testing silhouette")
    for n_samples in p_bar:
        if test_config["clustering_method"] == "kmeans_gpu":
            kmeans = kmeans_gpu.KMeans(n_clusters=n_samples, **test_config["clustering_params"])
            embeddings_torch = torch.from_numpy(embeddings)
            labels, _ = kmeans.fit_predict(embeddings_torch)
            labels = labels.detach().cpu().numpy()
        else:
            clustering = getattr(sklearn.cluster, test_config["clustering_method"])(
                n_clusters=n_samples, **test_config["clustering_params"]
            ).fit(embeddings)
            labels = clustering.labels_

        s_scores.append(silhouette_score(embeddings, labels, **test_config["silhouette_params"]))
        c_scores.append(calinski_harabasz_score(embeddings, labels))
        r_scores.append(rand_score(labels_true, labels))
        ra_scores.append(adjusted_rand_score(labels_true, labels))

        p_bar.set_description(f"Testing metrics: {n_samples} samples -> {s_scores[-1]} {c_scores[-1]} {ra_scores[-1]}")

    return s_scores, c_scores, r_scores, ra_scores, n_samples_range


def draw_results(test_config, df, plot_name):
    sns.set()

    _, axes = plt.subplots(2, 2, figsize=(16,9))

    sns.lineplot(ax=axes[0, 0], data=df, x="clusters_num", y="silhouette").set(title='Silhouette score')
    axes[0, 0].axvline(x=test_config["cluster"]["correct"], color="r", alpha=0.5, linestyle="--")

    sns.lineplot(ax=axes[0, 1], data=df, x="clusters_num", y="calinski_harabasz").set(title='Calinski-Harabasz score')
    axes[0, 1].axvline(x=test_config["cluster"]["correct"], color="r", alpha=0.5, linestyle="--")

    sns.lineplot(ax=axes[1, 0], data=df, x="clusters_num", y="rand").set(title='Rand score')
    axes[1, 0].axvline(x=test_config["cluster"]["correct"], color="r", alpha=0.5, linestyle="--")

    sns.lineplot(ax=axes[1, 1], data=df, x="clusters_num", y="rand_adjusted").set(title='Adjusted Rand score')
    axes[1, 1].axvline(x=test_config["cluster"]["correct"], color="r", alpha=0.5, linestyle="--")

    plt.subplots_adjust(hspace = 0.4)
    plt.savefig(os.path.join(test_config["plot_output_path"] + f"{plot_name}.jpg"))


@click.command()
@click.option("--cluster-config-path", type=click.Path(exists=True), required=True, help="Path to clusterization configuration")
@click.option("--model-config-path", type=click.Path(exists=True), required=True)
def main(cluster_config_path: str, model_config_path: str):
    run_name = f"{os.path.basename(cluster_config_path)[0]}-{os.path.basename(model_config_path)[0]}"
    print("Run name:", run_name)

    with open(cluster_config_path, "r") as config_file:
        test_config = yaml.safe_load(config_file)

    test_config["model_config_path"] = model_config_path

    print("Initializing dataset")
    test_dataset = get_dataset(test_config)

    print("Initializing dataloader")
    batch_size = test_config["dataloader"]["batch_size"]
    num_workers = test_config["dataloader"]["num_workers"]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)

    print("Initializing model")
    encoder = get_model(test_config)

    print("Creating embeddings")
    embeddings = get_full_embeddings(test_config, test_loader, encoder, run_name)
    print("Embeddings shape:", embeddings.shape)

    print("Applying PCA")
    embeddings = PCA(n_components=100, svd_solver="randomized").fit_transform(embeddings)

    if "umap" in test_config:
        print("Applying UMAP")
        embeddings = apply_umap(test_config, embeddings, test_dataset, run_name)

    s_scores, c_scores, r_scores, ra_scores, n_samples_range = evaluate(test_config, embeddings, test_dataset)
    data = {
        "clusters_num": n_samples_range,
        "silhouette": s_scores,
        "calinski_harabasz": c_scores,
        "rand": r_scores,
        "rand_adjusted": ra_scores,
    }
    df = pd.DataFrame(data=data)

    draw_results(test_config, df, run_name)


if __name__ == "__main__":
    main()
