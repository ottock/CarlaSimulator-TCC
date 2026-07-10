"""Polar LiDAR scan -> fixed-length sector vector.

Single source of truth for how a 2D scan becomes model input. Runs identically
in the simulator and (later) on the Jetson, so keep this Python 3.6-compatible.

Convention:
    - ``angles_deg``: 0 deg is the front of the car; the exact rotation direction
      only has to be consistent between the sim adapter and the real sensor.
    - Each sector holds the MINIMUM distance seen in that angular slice (the
      nearest thing in that direction), which is robust to point-density
      differences between the CARLA ray-cast and the real COIN-D6.
    - Output is normalized to [0, 1]; 1.0 means "free" (no return within range).
"""
import numpy as np

N_SECTORS = 72
MAX_RANGE_M = 12.0


def scan_to_sectors_m(angles_deg, dist_m, n_sectors=N_SECTORS, max_range=MAX_RANGE_M):
    """Reduce a polar scan to per-sector minimum distances, in METRES.

    Free sectors (no return within range) hold ``max_range``. This is the form
    stored in the dataset (``lidar.npy``) so training can re-normalize with any
    ``max_range`` later.

    Args:
        angles_deg: iterable of ray angles in degrees (0 = front).
        dist_m: iterable of ray distances in metres, aligned with ``angles_deg``.
            Values <= 0 or > ``max_range`` are treated as "no return".
        n_sectors: number of angular sectors covering the full 360 deg.
        max_range: cap; distances at/after this count as free.

    Returns:
        ``np.ndarray`` of shape ``(n_sectors,)``, dtype float32, in [0, max_range].
    """
    sectors = np.full(n_sectors, max_range, dtype=np.float32)

    a = np.asarray(angles_deg, dtype=np.float64)
    d = np.asarray(dist_m, dtype=np.float64)

    if a.size > 0:
        a = np.mod(a, 360.0)
        valid = (d > 0.0) & (d <= max_range)
        if np.any(valid):
            sector_width = 360.0 / n_sectors
            idx = (a[valid] / sector_width).astype(np.int64) % n_sectors
            np.minimum.at(sectors, idx, d[valid].astype(np.float32))

    return sectors


def scan_to_sectors(angles_deg, dist_m, n_sectors=N_SECTORS, max_range=MAX_RANGE_M):
    """Same as :func:`scan_to_sectors_m` but normalized to [0, 1] (1.0 = free).

    This is the model-input form.
    """
    meters = scan_to_sectors_m(angles_deg, dist_m, n_sectors=n_sectors, max_range=max_range)
    return (meters / max_range).astype(np.float32)
