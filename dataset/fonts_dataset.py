import os

import cv2
import json
from torch.utils.data import Dataset
import numpy as np


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

        self.fonts_to_files = {}
        for i, font in enumerate(self.fonts_list):
            self.fonts_to_files[self._get_font_id(font)] = i

        self.transform = transform

    def number_of_fonts(self) -> int:
        return len(self.fonts_list)

    def __len__(self) -> int:
        return len(self.words_list) * len(self.fonts_list)

    def __getitem__(self, idx: int):
        world_idx = idx % len(self.words_list)
        font_idx = idx // len(self.words_list)

        assert font_idx < len(self.fonts_list), "{} {} {} {} {}".format(font_idx, world_idx, idx, len(self.fonts_list), self.__len__())

        image_path = os.path.join(
            self.fonts_dir,
            self.fonts_list[font_idx],
            f"{self.words_list[world_idx]}.jpg"
        )

        image_tensor = cv2.imread(image_path, -1)
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        font_id = self._get_font_id(self.fonts_list[font_idx])

        return {
            "image_tensor": image_tensor,
            "font_id": font_id
        }

    def get_labels(self):
        result = []
        for idx in self.__len__():
            font_idx = idx // len(self.words_list)
            font_id = self._get_font_id(self.fonts_list[font_idx])
            result.append(font_id)
        return np.array(result)

    def _get_font_id(self, font_path):
        return int(font_path.split("/")[1].split("_")[0])
        
    def get_random_pair(self, author_id: int, can_link: bool):
        if can_link:
            file_idx = self.fonts_to_files[author_id]
        else:
            file_idx = np.random.randint(0, len(self.fonts_list))
            random_author = self.fonts_list[file_idx]
            font_id = self._get_font_id(random_author)
            while font_id == author_id:
                file_idx = np.random.randint(0, len(self.fonts_list))
                random_author = self.fonts_list[file_idx]
                font_id = self._get_font_id(random_author)

        image_id = np.random.randint(0, len(self.words_list))
        return self.__getitem__(file_idx * len(self.words_list) + image_id)
            

