import cv2
from glob import glob
from torch.utils.data import Dataset


class CVLDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.fonts_dir = root_dir
        self.images = glob(f"{self.fonts_dir}/*/*.tif")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image_path = self.images[idx]
        image_tensor = cv2.imread(image_path, -1)
        image_tensor = cv2.cvtColor(image_tensor, cv2.COLOR_GRAY2RGB)
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        font_id = image_path.split("/")[-2]
        return {
            "image_tensor": image_tensor,
            "font_id": font_id
        }

