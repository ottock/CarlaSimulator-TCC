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
