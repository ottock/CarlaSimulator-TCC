"""Tests for the CARLA point-cloud -> sector-vector adapter.

CARLA's ray-cast LiDAR returns 3D points in the sensor frame. On the real car
the COIN-D6 gives polar readings directly, so this xyz->sectors conversion is
sim-only: it drops ground/overhead/ego returns, projects to the ground plane,
and reuses the shared sector encoding.
"""
import numpy as np

from ai.sim_lidar import points_to_sectors_m


def test_front_point_maps_to_sector_zero_in_meters():
    sectors = points_to_sectors_m([[5.0, 0.0, 0.0]], n_sectors=72, max_range=12.0)
    assert sectors.shape == (72,)
    assert sectors[0] == np.float32(5.0)
    assert np.allclose(np.delete(sectors, 0), 12.0)


def test_left_point_maps_to_90_degree_sector():
    # +y is 90 deg -> sector 18 with 5-deg sectors
    sectors = points_to_sectors_m([[0.0, 3.0, 0.0]], n_sectors=72, max_range=12.0)
    assert sectors[18] == np.float32(3.0)


def test_ground_points_are_filtered():
    sectors = points_to_sectors_m([[5.0, 0.0, -2.0]], z_min=-1.7, z_max=2.0)
    assert np.allclose(sectors, 12.0)


def test_overhead_points_are_filtered():
    sectors = points_to_sectors_m([[5.0, 0.0, 3.0]], z_min=-1.7, z_max=2.0)
    assert np.allclose(sectors, 12.0)


def test_near_ego_returns_are_filtered():
    sectors = points_to_sectors_m([[0.2, 0.0, 0.0]], min_range=0.5)
    assert np.allclose(sectors, 12.0)


def test_empty_cloud_is_all_free():
    sectors = points_to_sectors_m(np.zeros((0, 3), dtype=np.float32))
    assert sectors.shape == (72,)
    assert np.allclose(sectors, 12.0)
