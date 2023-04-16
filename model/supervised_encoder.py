from typing import Dict, Any

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import ExponentialLR

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset.synthetic_segments import SyntheticSegments


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
        self.log("val_loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        loss = self.step(test_batch, batch_idx)
        self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss


def train_encoder(config: Dict[str, Any]):
    debug = config["debug"]
    train_dataset_path = config["dataset"]["train_path"]
    val_dataset_path = config["dataset"]["val_path"]

    transform = transforms.Compose([
        transforms.ToTensor()
        # TODO: do i need to resize?
    ])

    train_dataset = SyntheticSegments(train_dataset_path, debug, transform)
    val_dataset = SyntheticSegments(val_dataset_path, debug, transform)

    model = SupervisedEncoder(config)

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