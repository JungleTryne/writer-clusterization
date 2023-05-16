from typing import Optional

import cv2
from glob import glob
from torch.utils.data import Dataset
import numpy as np


class IAMResizedDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, cut: Optional[int] = None):
        self.fonts_dir = root_dir
        self.images = sorted(glob(f"{self.fonts_dir}/*.png"))
        if cut is not None:
            self.images = self.images[:cut]

        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image_path = self.images[idx]
        image_tensor = cv2.imread(image_path, -1)
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        font_id = self._get_font_id(image_path)
        return {
            "image_tensor": image_tensor,
            "font_id": font_id
        }

    def get_labels(self):
        result = []
        for image in self.images:
            result.append(self._get_font_id(image))
        return np.array(result)

    def number_of_authors(self) -> int:
        fonts = set()
        for image_path in self.images:
            font_id = self._get_font_id(image_path)
            fonts.add(font_id)
        return len(fonts)

    @staticmethod
    def _get_font_id(image_path):
        return int(image_path.split("/")[-1].split("-")[0])