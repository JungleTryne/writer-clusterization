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
from model.supervised_encoder import SupervisedEncoder

import sklearn.cluster
from sklearn.metrics import silhouette_score
# from sklearn.cluster import AgglomerativeClustering, KMeans

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import umap
import numpy as np


@click.command()
@click.option("--cluster-config-path", type=click.Path(exists=True), required=True, help="Path to clusterization configuration")
def main(cluster_config_path: click.Path):
    with open(str(cluster_config_path), "r") as config_file:
        test_config = yaml.safe_load(config_file)

    model_config_path = test_config["model_config_path"]
    with open(str(model_config_path), "r") as config_file:
        model_config = yaml.safe_load(config_file)

    debug = test_config["debug"]

    dataset_root = test_config["dataset"]["root_path"]
    test_fonts_file = "fonts_debug.json" if debug else test_config["dataset"]["fonts_test"]
    test_words = os.path.join(dataset_root, test_config["dataset"]["words_test"])
    test_fonts = os.path.join(dataset_root, test_fonts_file)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_dataset = FontsDataset(dataset_root, test_words, test_fonts, transform)
    batch_size = test_config["dataloader"]["batch_size"]
    num_workers = test_config["dataloader"]["num_workers"]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)

    print(f"Number of fonts: {test_dataset.number_of_fonts()}")

    if test_config["model"] == "encoder":
        encoder = SupervisedEncoder.load_from_checkpoint(test_config["checkpoint_path"], config=model_config)
    else:
        raise Exception(f"Invalid model: {test_config['model']}")
    encoder.eval()

    embeddings = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Getting embeddings"):
            embeddings.append(encoder(batch["image_tensor"]))
    embeddings = torch.cat(embeddings).detach().cpu().numpy()
    print("Embeddings shape:", embeddings.shape)

    print("Applying UMAP...")
    reducer = umap.UMAP(**test_config["umap"])
    reducer.fit(embeddings)
    embeddings_umap = reducer.transform(embeddings)
    print("UMAP result shape:", embeddings_umap.shape)

    s_scores = []
    min_samples = test_config["cluster"]["min_samples"] 
    max_samples = test_config["cluster"]["max_samples"]
    step = test_config["cluster"]["step"]
    n_samples_range = list(range(min_samples, max_samples, step))
    p_bar = tqdm(n_samples_range, desc="Testing silhouette")
    for n_samples in p_bar:
        clustering = getattr(sklearn.cluster, test_config["clustering_method"])(n_clusters=n_samples).fit(embeddings_umap)
        s_scores.append(silhouette_score(embeddings_umap, clustering.labels_, **test_config["silhouette_params"]))
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
