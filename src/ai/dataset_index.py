"""Dataset indexing for training (pure, no torch).

Turns a collected dataset into a flat list of per-frame records, splits by
episode (never by frame), and computes per-sample weights to counter the
straight-line imbalance. The torch ``Dataset`` in ``ai.dataset`` consumes this.
"""
import csv
import os
import random
from pathlib import Path

import numpy as np


def list_episodes(dataset_dir):
    """Return sorted paths of ``ep_*`` episode directories (separator-robust)."""
    return sorted(str(p) for p in Path(dataset_dir).glob("ep_*") if p.is_dir())


def split_episodes(episodes, val_frac=0.2, seed=0):
    """Split episodes into (train, val) by whole episode. Deterministic per seed."""
    eps = list(episodes)
    random.Random(seed).shuffle(eps)
    n_val = max(1, int(round(len(eps) * val_frac)))
    val = sorted(eps[:n_val])
    train = sorted(eps[n_val:])
    return train, val


def _read_labels(episode_dir):
    path = os.path.join(episode_dir, "labels.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_index(episodes):
    """Flatten episodes into per-frame records.

    Each record has: ``image`` (jpg path), ``lidar`` (episode's lidar.npy path),
    ``row`` (index into that lidar array = the frame number), and the control
    targets ``steer``/``throttle``/``brake``.
    """
    index = []
    for ep in episodes:
        lidar_path = os.path.join(ep, "lidar.npy")
        for row in _read_labels(ep):
            index.append({
                "image": os.path.join(ep, "frames", row["frame"] + ".jpg"),
                "lidar": lidar_path,
                "row": int(row["frame"]),
                "steer": float(row["steer"]),
                "throttle": float(row["throttle"]),
                "brake": float(row["brake"]),
            })
    return index


def sample_weights(steers, straight_thr=0.05, straight_w=1.0, curve_w=3.0):
    """Per-sample weights for a WeightedRandomSampler: upweight curves."""
    s = np.abs(np.asarray(steers, dtype=np.float64))
    return np.where(s < straight_thr, straight_w, curve_w).astype(np.float64)
