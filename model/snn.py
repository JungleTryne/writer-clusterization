from typing import Dict, Any

import numpy as np
import os

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision import transforms

from dataloader.collate import collate_fn
from dataset.fonts_dataset import FontsDataset

from torch.utils.data import DataLoader

from dataset.segments import SegmentsDataset
from utils.preprocessing import ssn_preprocessing_pipeline
import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger


class SiameseNN(pl.LightningModule):
    def __init__(
            self,
            config: Dict[str, Any],
            train_dataset: SegmentsDataset,
            val_dataset: SegmentsDataset
    ):
        super(SiameseNN, self).__init__()

        backbone = config["model"]["backbone"]
        alpha = config["model"]["alpha"]

        self.alpha = alpha
        self.encoder = getattr(models, backbone)(weights=False)
        self.criterion = nn.MSELoss()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.config = config

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def configure_optimizers(self):
        opt_name = self.config["optimizer"]["name"]
        params = self.config["optimizer"]["params"]
        optimizer = getattr(optim, opt_name)(self.parameters(), **params)
        return optimizer

    def step(self, batch, batch_idx, dataset):
        images_left = batch["image_tensor"]

        batch_size = len(batch["font_id"])
        can_link = batch_idx % 2 == 0
        target = np.ones(batch_size) * can_link
        target = torch.Tensor(target).to(self.device)

        images_right = []
        for i in range(batch_size):
            author_id = int(batch["font_id"][i])
            images_right.append(dataset.get_random_pair(author_id, can_link))

        # Make them one size
        images_right = collate_fn(images_right)["image_tensor"].to(self.device)

        embeddings_left: torch.Tensor = self.encoder(images_left)
        embeddings_right: torch.Tensor = self.encoder(images_right)

        distances = torch.sum((embeddings_left - embeddings_right) ** 2, axis=1)
        distances /= embeddings_left.shape[1]
        distances[distances > self.alpha] = self.alpha

        loss = self.criterion(distances, target)

        return loss

    def training_step(self, train_batch, batch_idx):
        loss = self.step(train_batch, batch_idx, self.train_dataset)
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self.step(val_batch, batch_idx, self.val_dataset)
        self.log("val_loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss


def train_snn(config: Dict[str, Any]):
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

    model = SiameseNN(config, train_dataset, val_dataset)

    batch_size = config["dataloader"]["batch_size"]
    num_workers = config["dataloader"]["num_workers"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)

    logger = TensorBoardLogger("tb_logs", name="resnet-18-snn")

    trainer = pl.Trainer(
        logger=logger,
        accelerator=config["training"]["device"],
        max_epochs=config["training"]["epochs"],
        devices=[0],
    )

    trainer.fit(model, train_loader, val_loader)