import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

import pytorch_lightning as pl


class AutoEncoder(pl.LightningModule):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),
        )

        self.decoder = nn.Sequential(
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(1),
            ),
        )

        self.criterion = nn.MSELoss()

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def put_through(self, x):
        embedding = self.encoder(x)
        decoded = self.decoder(embedding)
        return decoded

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=1e-3, momentum=0.9)
        scheduler = MultiStepLR(optimizer, milestones=[10, 30], gamma=0.1)

        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        images = train_batch["image"]
        images = images[:, None, :, :]

        output = self.decoder(self.encoder(images))
        loss = self.criterion(output, images)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        images = val_batch["image"]
        images = images[:, None, :, :]

        output = self.decoder(self.encoder(images))
        loss = self.criterion(output, images)
        self.log("val_loss", loss)
        return {
            "loss": loss,
            "images": (images, output)
        }