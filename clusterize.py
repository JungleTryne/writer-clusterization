import warnings

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

from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import umap


@click.command()
@click.option("--test-config-path", type=click.Path(exists=True), required=True, help="Path to clusterization configuration")
@click.option("--model-config-path", type=click.Path(exists=True), required=True, help="Path to clusterization configuration")
def main(test_config_path: click.Path, model_config_path: click.Path):
    with open(str(test_config_path), "r") as config_file:
        test_config = yaml.safe_load(config_file)

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

    print("Applying UMAP...")
    reducer = umap.UMAP(**test_config["umap"])
    reducer.fit(embeddings)
    embeddings_umap = reducer.transform(embeddings)
    print("UMAP result shape:", embeddings_umap.shape)

    s_scores = []
    n_samples_range = list(range(test_config["cluster"]["min_samples"], test_config["cluster"]["max_samples"], test_config["cluster"]["step"]))
    for n_samples in tqdm(n_samples_range, desc="Testing silhouette"):
        clustering = AgglomerativeClustering(n_clusters=n_samples).fit(embeddings_umap)
        s_scores.append(silhouette_score(embeddings_umap, clustering.labels_))
        print(n_samples, s_scores[-1])

    data_s = {
        "s_x": n_samples_range,
        "s_y": s_scores,
    }
    df_s = pd.DataFrame(data=data_s)

    sns.lineplot(data=df_s, x="s_x", y="s_y")
    plt.savefig("silhouette.png")


if __name__ == "__main__":
    main()
