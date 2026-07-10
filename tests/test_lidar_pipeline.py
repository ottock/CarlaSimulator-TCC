"""Tests for the shared LiDAR sector-encoding pipeline.

The encoding must be identical in the simulator and on the real car, so these
tests pin down the exact contract: a polar scan (angles + distances) becomes a
fixed-length vector of per-sector minimum distances, normalized to [0, 1] where
1.0 means "free / no return".
"""
import numpy as np
import pytest

from ai.shared.lidar_pipeline import scan_to_sectors


def test_empty_scan_is_all_free():
    sectors = scan_to_sectors([], [])
    assert sectors.shape == (72,)
    assert sectors.dtype == np.float32
    assert np.allclose(sectors, 1.0)


def test_front_point_normalized_min_distance():
    # angle 0 deg = front of the car -> sector 0; 6 m of 12 m -> 0.5
    sectors = scan_to_sectors([0.0], [6.0], n_sectors=72, max_range=12.0)
    assert sectors[0] == pytest.approx(0.5)
    assert np.allclose(np.delete(sectors, 0), 1.0)


def test_keeps_minimum_distance_within_sector():
    # both angles fall in the first 5-deg sector; the nearer one wins
    sectors = scan_to_sectors([1.0, 3.0], [8.0, 4.0], n_sectors=72, max_range=12.0)
    assert sectors[0] == pytest.approx(4.0 / 12.0)


def test_distance_beyond_range_is_free():
    sectors = scan_to_sectors([0.0], [20.0], n_sectors=72, max_range=12.0)
    assert sectors[0] == pytest.approx(1.0)


def test_nonpositive_distance_is_free():
    sectors = scan_to_sectors([0.0, 10.0], [0.0, -3.0], n_sectors=72, max_range=12.0)
    assert np.allclose(sectors, 1.0)


def test_angle_wraps_around_360():
    # 365 deg -> 5 deg -> sector 1 (5-deg sectors)
    sectors = scan_to_sectors([365.0], [6.0], n_sectors=72, max_range=12.0)
    assert sectors[1] == pytest.approx(0.5)
    assert sectors[0] == pytest.approx(1.0)


def test_negative_angle_wraps():
    # -1 deg -> 359 deg -> sector 71
    sectors = scan_to_sectors([-1.0], [6.0], n_sectors=72, max_range=12.0)
    assert sectors[71] == pytest.approx(0.5)


def test_custom_sector_count():
    # 4 sectors of 90 deg each, one point per quadrant at 3 m of 12 m -> 0.25
    sectors = scan_to_sectors(
        [0.0, 90.0, 180.0, 270.0], [3.0, 3.0, 3.0, 3.0], n_sectors=4, max_range=12.0
    )
    assert sectors.shape == (4,)
    assert np.allclose(sectors, 0.25)


def test_values_within_unit_interval():
    rng = np.random.default_rng(0)
    angles = rng.uniform(0.0, 360.0, 500)
    dists = rng.uniform(0.0, 15.0, 500)
    sectors = scan_to_sectors(angles, dists, n_sectors=72, max_range=12.0)
    assert sectors.shape == (72,)
    assert sectors.min() >= 0.0
    assert sectors.max() <= 1.0
