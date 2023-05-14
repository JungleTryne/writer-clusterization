import os
import pathlib

import cv2
from torch.utils.data import Dataset

from utils.constants import DEBUG_FILES_THRESHOLD


class SyntheticSegments(Dataset):
    """
    Dataset of artificial synthetic generated handwritten segments
    """

    def __init__(self, root_dir: pathlib.Path, debug: bool = False, transform = None):
        self.root_dir = root_dir
        self.files = os.listdir(self.root_dir)
        self.transform = transform

        if debug:
            self.files = self.files[:DEBUG_FILES_THRESHOLD]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        image_path = os.path.join(self.root_dir, self.files[idx])
        image_tensor = cv2.imread(image_path, -1)
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        font_id = int(self.files[idx].split("_")[0])
        return {
            "image_tensor": image_tensor,
            "font_id": font_id
        }
