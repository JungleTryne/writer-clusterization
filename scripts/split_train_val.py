import click
import os
import shutil

import numpy as np


@click.command()
@click.option("--train-path", type=click.Path(exists=True), help="Path to train dataset")
@click.option("--val-path", type=click.Path(exists=True), help="Path to val dataset")
@click.option("--ratio", type=float, default=0.1, help="Split ratio")
def main(train_path: click.Path, val_path: click.Path, ratio: float):
    """
    Script that splits train into train a val dataset
    """
    segments = np.array(os.listdir(str(train_path)))
    classes = np.random.choice([0, 1], size=len(segments), p=[1 - ratio, ratio])
    val_idx = np.where(classes == 1)
    val_files = segments[val_idx]

    for file in val_files:
        from_path = os.path.join(str(train_path), file)
        to_path = os.path.join(str(val_path), file)
        shutil.move(from_path, to_path)


if __name__ == "__main__":
    main()