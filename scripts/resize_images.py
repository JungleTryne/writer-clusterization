import cv2
import click
import imutils
import os

from glob import glob

import numpy as np
from tqdm import tqdm


@click.command()
@click.option("--root_pattern", type=str, help="Path to folder with images")
@click.option("--output_folder", type=click.Path(), help="Path to output folder")
@click.option("--height", type=int, help="Height of the resulting image")
def main(root_pattern: str, output_folder: str, height: int):
    images = glob(root_pattern)
    for image_path in tqdm(images):
        image_name = image_path.split("/")[-1]
        image: np.ndarray = cv2.imread(image_path)
        if image is None:
            continue
        if image.shape[0] > height:
            image = imutils.resize(image, height=height)
        cv2.imwrite(os.path.join(output_folder, image_name), image)


if __name__ == "__main__":
    main()