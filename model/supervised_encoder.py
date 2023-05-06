import os.path
from typing import Dict, Any

import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision import transforms

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataloader.collate import collate_fn
from dataset.fonts_dataset import FontsDataset

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import seaborn as sns
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_metric_learning import losses


# TorchVision ResNet-18 output before classification layer
EMBEDDING_SIZE = 512


class SupervisedEncoder(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super(SupervisedEncoder, self).__init__()

        backbone = config["model"]["backbone"]
        num_classes = config["model"]["number_of_classes"]

        self.encoder = getattr(models, backbone)(weights=None, num_classes=num_classes)

        criterion_name = config["model"]["criterion"]
        if criterion_name == "cross_entropy":
            self.classifier = self.encoder.fc
            self.criterion = nn.CrossEntropyLoss()
        elif criterion_name == "arcface":
            self.classifier = None
            self.criterion = losses.ArcFaceLoss(num_classes, EMBEDDING_SIZE)
        else:
            raise Exception(f"Invalid criterion: {criterion_name}")

        self.encoder.fc = nn.Identity()

        self.val_embeddings = []
        self.val_classes = []

        self.config = config
        self.logs_path = config["logs"]

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def configure_optimizers(self):
        opt_name = self.config["optimizer"]["name"]
        opt_params = self.config["optimizer"]["params"]
        optimizer = getattr(optim, opt_name)(self.parameters(), **opt_params)

        sch_name = self.config["scheduler"]["name"]
        sch_params = self.config["scheduler"]["params"]
        scheduler = getattr(optim.lr_scheduler, sch_name)(self.parameters(), **sch_params)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            }
        }

    def step(self, batch, _batch_idx):
        images = batch["image_tensor"]
        targets = batch["font_id"]
        embeddings: torch.Tensor = self.forward(images)

        if self.config["model"]["criterion"] == "cross_entropy":
            embeddings = self.classifier(embeddings)

        loss = self.criterion(embeddings, targets)

        return loss

    def training_step(self, train_batch, batch_idx):
        loss = self.step(train_batch, batch_idx)
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        embeddings = torch.cat(self.val_embeddings).detach().cpu().numpy()
        classes = torch.cat(self.val_classes).detach().cpu().numpy()
        assert embeddings.shape[0] == classes.shape[0]

        tsne = TSNE(n_components=2, learning_rate="auto", init="pca", random_state=42).fit_transform(embeddings)

        dt = pd.DataFrame(data={
            "x": tsne[:, 0],
            "y": tsne[:, 1],
            "class": classes
        })

        plt.clf()
        sns.scatterplot(data=dt, x="x", y="y", hue="class", palette="hls")
        plt.savefig(os.path.join(self.logs_path, f"{self.current_epoch}-tsne.jpg"))

        self.val_classes = []
        self.val_embeddings = []

    def validation_step(self, val_batch, batch_idx):
        images = val_batch["image_tensor"]
        targets = val_batch["font_id"]
        embeddings = self.forward(images)

        if self.config["model"]["criterion"] == "cross_entropy":
            embeddings = self.classifier(embeddings)

        loss = self.criterion(embeddings, targets)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.val_embeddings.append(embeddings)
        self.val_classes.append(targets)

        return loss

    def test_step(self, test_batch, batch_idx):
        loss = self.step(test_batch, batch_idx)
        self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss


def train_encoder(config: Dict[str, Any]):
    debug = config["debug"]

    dataset_root = config["dataset"]["root_path"]

    train_fonts_file = "fonts_debug.json" if debug else config["dataset"]["fonts_train"]
    val_fonts_file = "fonts_debug.json" if debug else config["dataset"]["fonts_val"]

    train_words = os.path.join(dataset_root, config["dataset"]["words_train"])
    val_words = os.path.join(dataset_root, config["dataset"]["words_val"])
    train_fonts = os.path.join(dataset_root, train_fonts_file)
    val_fonts = os.path.join(dataset_root, val_fonts_file)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = FontsDataset(dataset_root, train_words, train_fonts, transform)
    val_dataset = FontsDataset(dataset_root, val_words, val_fonts, transform)

    model = SupervisedEncoder(config)

    batch_size = config["dataloader"]["batch_size"]
    num_workers = config["dataloader"]["num_workers"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
    logger = TensorBoardLogger("tb_logs", name="resnet-18-metric-learning")

    trainer = pl.Trainer(
        logger=logger,
        accelerator=config["training"]["device"],
        max_epochs=config["training"]["epochs"],
        devices=[0],
    )

    trainer.fit(model, train_loader, val_loader)