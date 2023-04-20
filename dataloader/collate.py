import torch
import torch.nn.functional as F


def collate_fn(samples):
    font_ids = [sample["font_id"] for sample in samples]
    images = [sample["image_tensor"] for sample in samples]
    out_width = max([img.shape[-1] for img in images])

    images = [F.pad(img, pad=(0, out_width - img.shape[-1], 0, 0)) for img in images]

    return {
        "image_tensor": torch.stack(images).float(),
        "font_id": torch.LongTensor(font_ids)
    }
