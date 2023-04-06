import click
import cv2
import os

from glob import glob
from tqdm import tqdm
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from dataset.lines import CVLHandwritingLinesDataset

import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

WINDOW_SIZE = 224


def clean_output_path(output_path: click.Path):
    logging.info("Cleaning output path")
    files = glob(os.path.join(str(output_path), "*"))
    for file in tqdm(files):
        os.remove(file)


def get_variance(image: np.ndarray) -> float:
    variance = cv2.Laplacian(image, cv2.CV_64F).var()
    return variance


def generate_sliding_windows(
        lines_dataset: CVLHandwritingLinesDataset,
        output_path: click.Path,
        window_size: int,
        variance_threshold: Optional[float],
        debug: bool
):
    clean_output_path(output_path)
    variances = []

    global_idx = 0
    logging.info("Generating segments")
    # noinspection PyTypeChecker
    for line in tqdm(lines_dataset):
        segments = []

        if debug and global_idx > 1000:
            break

        image_tensor = line["image_tensor"]
        author_id = line["author_id"]
        text_id = line["text_id"]
        line_id = line["line_id"]

        # TODO: For now we skip last window of the line
        for segment_start in range(0, image_tensor.shape[1], window_size):
            if segment_start + window_size > image_tensor.shape[1]:
                break

            segment_tensor = image_tensor[:, segment_start:segment_start + window_size]
            if variance_threshold is not None:
                variance = get_variance(segment_tensor)
                if variance < variance_threshold:
                    continue
                variances.append(variance)

            segments.append(segment_tensor)

        for idx, segment in enumerate(segments):
            global_idx += 1
            file_name = f"{author_id}-{text_id}-{line_id}-{global_idx}.png"
            file_path = os.path.join(str(output_path), str(file_name))
            cv2.imwrite(file_path, segment)

    plt.hist(variances)
    plt.show()


@click.command()
@click.option("--lines-path", type=click.Path(exists=True), help="Path to lines dataset")
@click.option("--segments-path", type=click.Path(exists=True), help="Path to folder for resulting segments")
@click.option("--window-size", type=int, default=WINDOW_SIZE, help="Sliding window size")
@click.option("--variance-threshold", type=float, required=False, help="Threshold for variance to filter blank segments")
@click.option("--debug", is_flag=True, help="Toggle debug mode")
def main(lines_path: click.Path, segments_path: click.Path, window_size: int, variance_threshold: Optional[float], debug: bool):
    """Script that generates segments from lines using a sliding window"""
    lines_dataset = CVLHandwritingLinesDataset(lines_path)
    generate_sliding_windows(lines_dataset, segments_path, window_size, variance_threshold, debug)


if __name__ == "__main__":
    main()
