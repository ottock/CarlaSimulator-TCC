"""CARLA LiDAR point cloud -> sector vector (sim only).

CARLA's ray-cast LiDAR returns 3D points in the sensor frame; the real COIN-D6
gives polar readings directly. This adapter bridges the sim side: drop ground,
overhead and near-ego returns, project to the ground plane, then reuse the
shared sector encoding. Output is per-sector minimum distance in METRES (free
sectors = ``max_range``), matching what the dataset stores.
"""
import numpy as np

from ai.shared.lidar_pipeline import scan_to_sectors_m


def points_to_sectors_m(points_xyz, n_sectors=72, max_range=12.0,
                        z_min=-1.7, z_max=2.0, min_range=0.5):
    """Convert a CARLA LiDAR cloud to a per-sector minimum-distance vector (metres).

    Args:
        points_xyz: ``(N, 3)`` array of sensor-frame points (x forward, y right).
        n_sectors: number of angular sectors over 360 deg.
        max_range: sector cap; returns beyond this are treated as free.
        z_min, z_max: keep points whose height is in this band (drops the ground
            plane and overhead structures).
        min_range: drop returns closer than this (ego-vehicle artifacts).

    Returns:
        ``np.ndarray`` of shape ``(n_sectors,)``, float32, metres.
    """
    pts = np.asarray(points_xyz, dtype=np.float64)
    free = np.full(n_sectors, max_range, dtype=np.float32)
    if pts.size == 0:
        return free

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    r = np.hypot(x, y)
    keep = (z >= z_min) & (z <= z_max) & (r >= min_range) & (r <= max_range)
    if not np.any(keep):
        return free

    angles_deg = np.degrees(np.arctan2(y[keep], x[keep]))
    return scan_to_sectors_m(angles_deg, r[keep], n_sectors=n_sectors, max_range=max_range)
