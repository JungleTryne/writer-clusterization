import pathlib
from typing import Dict, Any
import cv2

from torch.utils.data import Dataset

import os

from utils.image import image_resize, convert_image

IMAGE_HEIGHT: int = 112


def _get_info_from_filename(image_path: str) -> Dict[str, Any]:
    sample_info = image_path.split('/')[-1].split(".")[0].split('-')
    author_id = sample_info[0]
    text_id = sample_info[1]
    line_id = sample_info[2]

    return {
        "author_id": int(author_id),
        "text_id": int(text_id),
        "line_id": int(line_id),
    }


class CVLHandwritingLinesDataset(Dataset):
    """
    Dataset of handwritten lines, taken from CVL Database
    """

    def __init__(self, root_dir: pathlib.Path, image_height: int = IMAGE_HEIGHT, debug: bool = False):
        self.root_dir = root_dir
        self.image_height = image_height
        self.files = os.listdir(self.root_dir)
        if debug:
            self.files = self.files[:100]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path = os.path.join(self.root_dir, self.files[idx])
        image_tensor = cv2.imread(image_path, -1)

        image_tensor = image_resize(image_tensor, height=self.image_height)
        image_tensor = convert_image(image_tensor)

        item = {
            "image_tensor": image_tensor
        }
        item.update(_get_info_from_filename(image_path))

        return item
