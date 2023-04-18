import os

import cv2
import json
from torch.utils.data import Dataset


class FontsDataset(Dataset):
    """
    Dataset of artificial synthetic generated handwritten segments
    """

    def __init__(self, root_dir: str, words_list_path: str, fonts_list_path: str, transform=None):
        self.fonts_dir = root_dir
        with open(words_list_path, "r") as config:
            self.words_list = json.load(config)

        with open(fonts_list_path, "r") as config:
            self.fonts_list = json.load(config)

        self.transform = transform

    def __len__(self) -> int:
        return len(self.words_list) * len(self.fonts_list)

    def __getitem__(self, idx: int):
        world_idx = idx % len(self.words_list)
        font_idx = idx // len(self.words_list)
        image_path = os.path.join(
            self.fonts_dir,
            self.fonts_list[font_idx],
            f"{self.words_list[world_idx]}.jpg"
        )

        image_tensor = cv2.imread(image_path, -1)
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        font_id = int(self.fonts_list[font_idx].split("/")[1].split("_")[0])

        return {
            "image_tensor": image_tensor,
            "font_id": font_id
        }

