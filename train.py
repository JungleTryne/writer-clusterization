import warnings

# Filter out the warning message for ShardedTensor
warnings.filterwarnings("ignore", category=UserWarning, message="Please use DTensor instead and we are deprecating "
                                                                "ShardedTensor.")

import click
import yaml
from torch.utils.data import DataLoader

from dataset.segments import SegmentsDataset
from model.snn import SiameseNN
from utils.preprocessing import preprocessing_pipeline
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


@click.command()
@click.option("--config-path", type=click.Path(exists=True), help="Path to training configuration")
def main(config_path: click.Path):
    """
    Training script.
    """
    with open(str(config_path), "r") as config_file:
        config = yaml.safe_load(config_file)

    debug = config["debug"]
    train_dataset_path = config["dataset"]["train_path"]
    val_dataset_path = config["dataset"]["val_path"]
    test_dataset_path = config["dataset"]["test_path"]

    train_dataset = SegmentsDataset(train_dataset_path, debug, preprocessing_pipeline)
    val_dataset = SegmentsDataset(val_dataset_path, debug)
    test_dataset = SegmentsDataset(test_dataset_path, debug)

    model = SiameseNN(config, train_dataset, val_dataset, test_dataset)

    batch_size = config["dataloader"]["batch_size"]
    num_workers = config["dataloader"]["num_workers"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    wandb_logger = WandbLogger()
    wandb_logger.experiment.config.update(config)

    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator=config["training"]["device"],
        max_epochs=config["training"]["epochs"],
        devices=[0],
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
