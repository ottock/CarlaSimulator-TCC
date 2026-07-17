"""Tests for the torch steering Dataset (Fase 2).

Loads a frame via the shared preprocessing and pairs it with the steer target.
"""
import numpy as np
import pytest
import torch

from ai.dataset_writer import EpisodeWriter
from ai.dataset_index import build_index
from ai.dataset import SteeringDataset


def _episode(path, steers):
    w = EpisodeWriter(str(path))
    for s in steers:
        w.add(np.zeros((360, 640, 3), dtype=np.uint8), np.zeros(72, dtype=np.float32),
              {"steer": s, "throttle": 0.5, "brake": 0.0, "v": 1.0,
               "x": 0, "y": 0, "yaw": 0, "noise_active": False})
    w.close()


def test_len_matches_index(tmp_path):
    _episode(tmp_path / "ep_0001", [0.1, -0.2, 0.3])
    ds = SteeringDataset(build_index([str(tmp_path / "ep_0001")]))
    assert len(ds) == 3


def test_item_shapes_and_target(tmp_path):
    _episode(tmp_path / "ep_0001", [0.1, -0.2])
    ds = SteeringDataset(build_index([str(tmp_path / "ep_0001")]))
    x, y = ds[0]
    assert tuple(x.shape) == (3, 66, 200)
    assert x.dtype == torch.float32
    assert tuple(y.shape) == (1,)
    assert float(y[0]) == pytest.approx(0.1)


from ai.dataset import DrivingDataset


def _episode_dual(path, rows):
    """rows: lista de (steer, throttle, brake). LiDAR do frame i = np.full(72, i)."""
    w = EpisodeWriter(str(path))
    for i, (s, th, br) in enumerate(rows):
        w.add(np.zeros((360, 640, 3), dtype=np.uint8),
              np.full(72, float(i), dtype=np.float32),
              {"steer": s, "throttle": th, "brake": br, "v": 1.0,
               "x": 0, "y": 0, "yaw": 0, "noise_active": False})
    w.close()


def test_driving_item_shapes_and_target(tmp_path):
    _episode_dual(tmp_path / "ep_0001", [(0.1, 0.5, 0.0), (-0.2, 0.0, 0.9)])
    ds = DrivingDataset(build_index([str(tmp_path / "ep_0001")]))
    img, lidar, target = ds[1]
    assert tuple(img.shape) == (3, 66, 200) and img.dtype == torch.float32
    assert tuple(lidar.shape) == (72,) and lidar.dtype == torch.float32
    assert tuple(target.shape) == (3,)
    assert float(target[0]) == pytest.approx(-0.2)
    assert float(target[1]) == pytest.approx(0.0)
    assert float(target[2]) == pytest.approx(0.9)


def test_driving_lidar_row_alignment_and_norm(tmp_path):
    # frame 1 -> lidar.npy linha 1 = full(1.0) -> normalizado 1/12.
    _episode_dual(tmp_path / "ep_0001", [(0.0, 0.5, 0.0), (0.0, 0.5, 0.0)])
    ds = DrivingDataset(build_index([str(tmp_path / "ep_0001")]), max_range=12.0)
    _, lidar, _ = ds[1]
    assert float(lidar[0]) == pytest.approx(1.0 / 12.0)
    assert float(lidar.min()) >= 0.0 and float(lidar.max()) <= 1.0
