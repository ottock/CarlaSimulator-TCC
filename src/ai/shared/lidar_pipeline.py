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


def apply_fov_mask(sectors_m, fov_deg=180.0, max_range=MAX_RANGE_M):
    """Blind the sectors outside a frontal field of view, in METRES.

    On the real car the chassis occludes the rear of the LiDAR, so the network
    can only ever see a frontal arc. Masking only at inference time would hand
    the net an input it never saw in training (25% of the vector changes, mean
    error 0.177 -- the very condition that made the LiDAR ablation crash), so the
    SAME mask is applied when building the training set and when driving.

    Sector ``i`` spans ``[i*360/n, (i+1)*360/n)``, so its centre is
    ``(i+0.5)*360/n``. The sector is kept when that centre, wrapped to
    ``[-180, +180]``, satisfies ``|angle| <= fov_deg/2``; otherwise it reads
    ``max_range``, i.e. "free / nothing there" -- the same value an unoccupied
    sector holds, so the masked input stays inside the distribution the model
    already knows.

    Args:
        sectors_m: per-sector minimum distances in metres (see
            :func:`scan_to_sectors_m`).
        fov_deg: total frontal field of view in degrees. ``>= 360`` is a no-op.
        max_range: value written into the blinded sectors ("free").

    Returns:
        A NEW ``np.ndarray`` (the caller's array is never mutated, so the raw
        ``lidar.npy`` stays intact), shape unchanged, dtype float32.
    """
    out = np.asarray(sectors_m, dtype=np.float32).copy()
    if fov_deg >= 360.0:
        return out

    n = out.shape[-1]
    centers = (np.arange(n, dtype=np.float64) + 0.5) * (360.0 / n)
    centers = np.mod(centers + 180.0, 360.0) - 180.0   # -> [-180, +180)
    out[np.abs(centers) > fov_deg / 2.0] = np.float32(max_range)
    return out


def normalize_sectors_m(sectors_m, max_range=MAX_RANGE_M):
    """Turn stored per-sector metres into model input in [0, 1] (near=0, free=1).

    Single source of truth used by both training (``DrivingDataset``) and
    inference (``DrivingPolicy``). Python 3.6-safe (runs on the Jetson too).
    """
    x = np.asarray(sectors_m, dtype=np.float32) / float(max_range)
    return np.clip(x, 0.0, 1.0).astype(np.float32)
