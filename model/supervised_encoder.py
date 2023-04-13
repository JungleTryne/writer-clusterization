from typing import Dict, Any

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

import pytorch_lightning as pl


class SupervisedEncoder(pl.LightningModule):
    def __init__(
            self,
            config: Dict[str, Any],
    ):
        super(SupervisedEncoder, self).__init__()

        backbone = config["model"]["backbone"]
        num_classes = config["model"]["number_of_classes"]

        self.encoder = getattr(models, backbone)(weights=False, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()

        self.config = config

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def configure_optimizers(self):
        opt_name = self.config["optimizer"]["name"]
        params = self.config["optimizer"]["params"]
        optimizer = getattr(optim, opt_name)(self.parameters(), **params)

        decay_factor = self.config["scheduler"]["decay_factor"]
        scheduler = ExponentialLR(optimizer, gamma=decay_factor)

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
        classes: torch.Tensor = self.encoder(images)
        loss = self.criterion(classes, targets)

        return loss

    def training_step(self, train_batch, batch_idx):
        loss = self.step(train_batch, batch_idx)
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self.step(val_batch, batch_idx)
        self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        loss = self.step(test_batch, batch_idx)
        self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss
