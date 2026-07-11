"""Torch Dataset for camera-only behavioral cloning (Fase 2).

Wraps the flat index from ``ai.dataset_index`` and returns
``(image_tensor, steer_target)`` using the SAME preprocessing the car will run.
"""
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ai.shared.image_pipeline import preprocess


class SteeringDataset(Dataset):
    """Yields ``(img (3,66,200) float32, steer (1,) float32)`` per frame."""

    def __init__(self, index):
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        rec = self.index[i]
        img_bgr = cv2.imread(rec["image"], cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(rec["image"])
        x = preprocess(img_bgr)  # (3, 66, 200) float32, BGR, [-1, 1]
        y = np.array([rec["steer"]], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)
