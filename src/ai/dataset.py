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


from ai.shared.lidar_pipeline import apply_fov_mask, normalize_sectors_m


class DrivingDataset(Dataset):
    """Dual-input: yields ``(img (3,66,200), lidar (72,) in [0,1], target (3,))``.

    Target = [steer, throttle, brake]. LiDAR is loaded per-episode via mmap and
    normalized with the SAME function the car will use at inference.

    ``fov_deg`` (Fase 6a) blinds the sectors the real car cannot see -- its own
    chassis occludes the rear of the LiDAR. The mask is applied on the way out,
    so ``lidar.npy`` keeps the full 360 deg scan and the FOV can be changed
    without recollecting. ``None`` = no mask (the Fase 4 behaviour).
    """

    def __init__(self, index, max_range=12.0, fov_deg=None):
        self.index = index
        self.max_range = max_range
        self.fov_deg = fov_deg
        self._lidar_cache = {}

    def __len__(self):
        return len(self.index)

    def _lidar_array(self, path):
        arr = self._lidar_cache.get(path)
        if arr is None:
            arr = np.load(path, mmap_mode="r")
            self._lidar_cache[path] = arr
        return arr

    def __getitem__(self, i):
        rec = self.index[i]
        img_bgr = cv2.imread(rec["image"], cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(rec["image"])
        x = preprocess(img_bgr)
        sectors_m = np.asarray(self._lidar_array(rec["lidar"])[rec["row"]], dtype=np.float32)
        if self.fov_deg is not None:
            sectors_m = apply_fov_mask(sectors_m, self.fov_deg, self.max_range)
        lidar = normalize_sectors_m(sectors_m, self.max_range)
        target = np.array([rec["steer"], rec["throttle"], rec["brake"]], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(lidar), torch.from_numpy(target)
