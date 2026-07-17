"""Tests for on-disk episode writing.

One episode = ``frames/NNNNNN.jpg`` + ``lidar.npy`` (N x 72, metres) + a
``labels.csv`` row per frame, all aligned by index so image, LiDAR and control
come from the same synchronous tick.
"""
import csv
import json

import numpy as np
import pytest

from ai.dataset_writer import EpisodeWriter, write_meta, LABEL_COLUMNS


def _label(steer=0.0):
    return {"steer": steer, "throttle": 0.5, "brake": 0.0, "v": 2.0,
            "x": 1.0, "y": 2.0, "yaw": 3.0, "noise_active": False}


def test_writes_frames_lidar_and_labels(tmp_path):
    ep = tmp_path / "ep_0001"
    w = EpisodeWriter(str(ep))
    for i in range(3):
        img = np.full((360, 640, 3), i, dtype=np.uint8)
        lidar = np.full(72, float(i), dtype=np.float32)
        w.add(img, lidar, _label(steer=0.1 * i))
    w.close()

    frames = sorted((ep / "frames").glob("*.jpg"))
    assert [f.name for f in frames] == ["000000.jpg", "000001.jpg", "000002.jpg"]

    arr = np.load(ep / "lidar.npy")
    assert arr.shape == (3, 72)
    assert arr.dtype == np.float32
    assert np.allclose(arr[2], 2.0)

    with open(ep / "labels.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert list(rows[0].keys()) == LABEL_COLUMNS
    assert rows[0]["frame"] == "000000"
    assert float(rows[1]["steer"]) == pytest.approx(0.1)


def test_add_returns_incrementing_index(tmp_path):
    w = EpisodeWriter(str(tmp_path / "ep"))
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    lidar = np.zeros(72, dtype=np.float32)
    assert w.add(img, lidar, _label()) == 0
    assert w.add(img, lidar, _label()) == 1


def test_write_meta(tmp_path):
    write_meta(str(tmp_path), {"pipeline_version": 1, "fov": 62.2})
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["fov"] == 62.2
    assert meta["pipeline_version"] == 1
