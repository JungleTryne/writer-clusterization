from torch.utils.data import Dataset
import pathlib
from typing import Optional, Any, Dict
import os
import cv2 as cv
import torch
import numpy as np


class SlicesDataset(Dataset):
    def __init__(self, root_dir: pathlib.Path, transforms: Optional[Any]=None, debug: bool=True):
        self.root_dir = root_dir
        self.files = sorted(os.listdir(self.root_dir))
        self.transforms = transforms
        if debug:
            self.files = self.files[:100]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        filename = self.files[idx]
        path_to_image = os.path.join(self.root_dir, filename)
        image = cv.imread(path_to_image, 0)

        if self.transforms is not None:
            image = self.transforms(image) 

        return {
            "image": torch.from_numpy(image).to(torch.float32) / 255,
            "author": int(filename.split("-")[0]),
            "text_id": int(filename.split("-")[1]),
        }
    
    def get_author_text(self, author_id: int, text_id: int) -> Dict[str, Any]:
        result = []
        for file in self.files:
            author = int(file.split("-")[0])
            text = int(file.split("-")[1])
            if author == author_id and text == text_id:
                path_to_image = os.path.join(self.root_dir, file)
                result.append(cv.imread(path_to_image, 0))
        
        result = np.array(result)
        return {
            "images": torch.from_numpy(result).to(torch.float32) / 255,
            "author": author_id,
            "text_id": text_id,
        }