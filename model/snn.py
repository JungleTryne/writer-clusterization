from typing import Dict, Any

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

from torch.utils.data import DataLoader

from dataset.segments import SegmentsDataset
from utils.preprocessing import ssn_preprocessing_pipeline
import pytorch_lightning as pl


class SiameseNN(pl.LightningModule):
    def __init__(
            self,
            config: Dict[str, Any],
            train_dataset: SegmentsDataset,
            val_dataset: SegmentsDataset,
            test_dataset: SegmentsDataset
    ):
        super(SiameseNN, self).__init__()

        backbone = config["model"]["backbone"]
        alpha = config["model"]["alpha"]

        self.alpha = alpha
        self.encoder = getattr(models, backbone)(weights=False)
        self.criterion = nn.MSELoss()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

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
        images_right = images_left.clone()

        batch_size = len(batch["author_id"])
        can_link = batch_idx % 2 == 0
        target = np.ones(batch_size) * can_link
        target = torch.Tensor(target).to(self.device)

        for i in range(batch_size):
            author_id = int(batch["author_id"][i])
            images_right[i] = dataset.get_random_pair(author_id, can_link)["image_tensor"]

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
        self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        loss = self.step(test_batch, batch_idx, self.test_dataset)
        self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss


def train_snn(config: Dict[str, Any]):
    debug = config["debug"]
    train_dataset_path = config["dataset"]["train_path"]
    val_dataset_path = config["dataset"]["val_path"]
    test_dataset_path = config["dataset"]["test_path"]

    train_dataset = SegmentsDataset(train_dataset_path, debug, ssn_preprocessing_pipeline)
    val_dataset = SegmentsDataset(val_dataset_path, debug)
    test_dataset = SegmentsDataset(test_dataset_path, debug)

    model = SiameseNN(config, train_dataset, val_dataset, test_dataset)

    batch_size = config["dataloader"]["batch_size"]
    num_workers = config["dataloader"]["num_workers"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    trainer = pl.Trainer(
        accelerator=config["training"]["device"],
        max_epochs=config["training"]["epochs"],
        devices=[0],
    )

    trainer.fit(model, train_loader, val_loader)