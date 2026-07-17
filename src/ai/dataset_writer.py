"""Write a behavioral-cloning dataset to disk, one episode at a time.

Layout (see the plan):

    dataset_vN/
        meta.json
        ep_0001/
            frames/000000.jpg      # BGR, as captured
            lidar.npy              # (N, 72) float32, metres (free = max_range)
            labels.csv             # one row per frame, same index as the JPG

Everything is index-aligned so image, LiDAR and control come from the same
synchronous tick.
"""
import csv
import json
import os

import cv2
import numpy as np

LABEL_COLUMNS = ["frame", "steer", "throttle", "brake", "v", "x", "y", "yaw", "noise_active"]


class EpisodeWriter:
    """Accumulate one episode and flush ``lidar.npy`` + ``labels.csv`` on close."""

    def __init__(self, episode_dir):
        self._dir = episode_dir
        self._frames_dir = os.path.join(episode_dir, "frames")
        os.makedirs(self._frames_dir, exist_ok=True)
        self._lidar_rows = []
        self._labels = []
        self._count = 0

    def add(self, image_bgr, lidar_sectors_m, label):
        """Write one frame's JPG and buffer its LiDAR row + label. Returns the frame index."""
        idx = self._count
        name = "%06d" % idx
        cv2.imwrite(os.path.join(self._frames_dir, name + ".jpg"), image_bgr)
        self._lidar_rows.append(np.asarray(lidar_sectors_m, dtype=np.float32))
        row = {"frame": name}
        for key in LABEL_COLUMNS[1:]:
            row[key] = label.get(key, "")
        self._labels.append(row)
        self._count += 1
        return idx

    def close(self):
        """Flush the buffered LiDAR array and labels CSV."""
        if self._lidar_rows:
            lidar = np.stack(self._lidar_rows).astype(np.float32)
        else:
            lidar = np.zeros((0, 0), dtype=np.float32)
        np.save(os.path.join(self._dir, "lidar.npy"), lidar)
        with open(os.path.join(self._dir, "labels.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
            writer.writeheader()
            writer.writerows(self._labels)

    def __len__(self):
        return self._count


def write_meta(dataset_dir, meta):
    """Write ``meta.json`` (camera/LiDAR config, stage/scale, pipeline version)."""
    os.makedirs(dataset_dir, exist_ok=True)
    with open(os.path.join(dataset_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
