"""Tests for dataset indexing (pure, no torch).

Builds the frame index the training Dataset consumes, splits by EPISODE (never
by frame - consecutive frames are near-duplicates, so a frame split leaks the
validation set), and computes per-sample weights to fight the ~88%-straight
imbalance.
"""
import numpy as np
import pytest

from ai.dataset_writer import EpisodeWriter
from ai.dataset_index import list_episodes, split_episodes, build_index, sample_weights, sample_weights_dual


def _make_episode(path, steers):
    w = EpisodeWriter(str(path))
    for s in steers:
        w.add(np.zeros((8, 8, 3), dtype=np.uint8), np.zeros(72, dtype=np.float32),
              {"steer": s, "throttle": 0.5, "brake": 0.0, "v": 1.0,
               "x": 0, "y": 0, "yaw": 0, "noise_active": False})
    w.close()


def test_list_episodes_sorted(tmp_path):
    _make_episode(tmp_path / "ep_0002", [0.0])
    _make_episode(tmp_path / "ep_0001", [0.0])
    eps = list_episodes(str(tmp_path))
    assert [e.replace("\\", "/").split("/")[-1] for e in eps] == ["ep_0001", "ep_0002"]


def test_split_by_episode_is_disjoint_and_deterministic():
    eps = ["ep_%02d" % i for i in range(10)]
    train, val = split_episodes(eps, val_frac=0.2, seed=0)
    assert len(val) == 2 and len(train) == 8
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(eps)
    assert split_episodes(eps, val_frac=0.2, seed=0) == (train, val)  # deterministic


def test_build_index_maps_frames_to_labels(tmp_path):
    _make_episode(tmp_path / "ep_0001", [0.1, -0.2, 0.3])
    index = build_index([str(tmp_path / "ep_0001")])
    assert len(index) == 3
    assert index[0]["image"].endswith("000000.jpg")
    assert index[1]["steer"] == pytest.approx(-0.2)
    assert index[2]["row"] == 2  # row index into lidar.npy


def test_sample_weights_upweight_curves():
    steers = [0.0, 0.01, 0.5, -0.4]  # 2 straight, 2 curve
    w = sample_weights(steers, straight_thr=0.05, straight_w=1.0, curve_w=3.0)
    assert list(w) == [1.0, 1.0, 3.0, 3.0]


def test_sample_weights_dual_upweights_curve_and_brake():
    steers = [0.0, 0.5, 0.0, 0.5]
    brakes = [0.0, 0.0, 0.8, 0.8]
    w = sample_weights_dual(steers, brakes, curve_w=3.0, brake_w=4.0)
    assert w[0] == 1.0          # reto, sem freio
    assert w[1] == 3.0          # curva
    assert w[2] == 4.0          # freio
    assert w[3] == 4.0          # curva + freio -> máximo dos dois
