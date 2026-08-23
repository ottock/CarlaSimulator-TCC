"""Tests for the shared LiDAR sector-encoding pipeline.

The encoding must be identical in the simulator and on the real car, so these
tests pin down the exact contract: a polar scan (angles + distances) becomes a
fixed-length vector of per-sector minimum distances, normalized to [0, 1] where
1.0 means "free / no return".
"""
import numpy as np
import pytest

from ai.shared.lidar_pipeline import apply_fov_mask, scan_to_sectors, scan_to_sectors_m


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


def test_sectors_in_meters_keeps_distance():
    # meters variant stores the raw minimum distance, free sectors = max_range
    sectors = scan_to_sectors_m([0.0], [6.0], n_sectors=72, max_range=12.0)
    assert sectors[0] == pytest.approx(6.0)
    assert np.allclose(np.delete(sectors, 0), 12.0)


def test_normalized_is_meters_over_max_range():
    angles = [0.0, 30.0, 200.0]
    dists = [3.0, 9.0, 1.5]
    meters = scan_to_sectors_m(angles, dists, n_sectors=72, max_range=12.0)
    norm = scan_to_sectors(angles, dists, n_sectors=72, max_range=12.0)
    assert np.allclose(norm, meters / 12.0)


def test_normalize_maps_metres_to_unit_range():
    from ai.shared.lidar_pipeline import normalize_sectors_m
    out = normalize_sectors_m(np.array([0.0, 6.0, 12.0], dtype=np.float32), max_range=12.0)
    assert np.allclose(out, [0.0, 0.5, 1.0])
    assert out.dtype == np.float32


def test_normalize_clips_beyond_range():
    from ai.shared.lidar_pipeline import normalize_sectors_m
    out = normalize_sectors_m(np.array([-1.0, 24.0], dtype=np.float32), max_range=12.0)
    assert np.allclose(out, [0.0, 1.0])  # clipa abaixo de 0 e acima de max_range


# --- FOV mask (Fase 6a) -------------------------------------------------------
# A carroceria do carro real oclui a traseira do LiDAR. A mesma mascara roda no
# treino e na inferencia, senao o modelo recebe uma entrada que nunca viu.


def test_fov_180_keeps_front_half_and_frees_the_rear():
    # 72 setores de 5 deg; o centro do setor i e' (i+0.5)*5 deg. Com FOV 180 sobram
    # os centros em [-90, +90]: setores 0..17 (2.5..87.5) e 54..71 (272.5..357.5).
    sectors = np.full(72, 3.0, dtype=np.float32)
    out = apply_fov_mask(sectors, fov_deg=180.0, max_range=12.0)
    kept = list(range(0, 18)) + list(range(54, 72))
    freed = list(range(18, 54))
    assert np.allclose(out[kept], 3.0)
    assert np.allclose(out[freed], 12.0)


def test_fov_mask_preserves_kept_values_exactly():
    sectors = np.arange(72, dtype=np.float32) / 10.0
    out = apply_fov_mask(sectors, fov_deg=180.0, max_range=12.0)
    assert out[0] == pytest.approx(sectors[0])
    assert out[17] == pytest.approx(sectors[17])
    assert out[71] == pytest.approx(sectors[71])


def test_fov_360_is_a_noop():
    sectors = np.arange(72, dtype=np.float32) / 10.0
    assert np.allclose(apply_fov_mask(sectors, fov_deg=360.0, max_range=12.0), sectors)
    assert np.allclose(apply_fov_mask(sectors, fov_deg=400.0, max_range=12.0), sectors)


def test_fov_zero_frees_everything():
    sectors = np.full(72, 3.0, dtype=np.float32)
    out = apply_fov_mask(sectors, fov_deg=0.0, max_range=12.0)
    assert np.allclose(out, 12.0)


def test_fov_mask_does_not_mutate_input():
    sectors = np.full(72, 3.0, dtype=np.float32)
    apply_fov_mask(sectors, fov_deg=180.0, max_range=12.0)
    assert np.allclose(sectors, 3.0)  # o dado cru do dataset tem de ficar intacto


def test_fov_mask_keeps_shape_and_dtype():
    out = apply_fov_mask(np.full(72, 3.0, dtype=np.float64), fov_deg=180.0, max_range=12.0)
    assert out.shape == (72,)
    assert out.dtype == np.float32


def test_fov_mask_works_for_any_sector_count():
    # 4 setores de 90 deg, centros 45/135/225/315 -> so 45 e 315 caem em [-90, +90]
    out = apply_fov_mask(np.full(4, 3.0, dtype=np.float32), fov_deg=180.0, max_range=12.0)
    assert np.allclose(out, [3.0, 12.0, 12.0, 3.0])


def test_fov_mask_uses_the_given_max_range():
    # no carro a escala 1:12 vira max_range=1.0; o "livre" tem de acompanhar
    out = apply_fov_mask(np.full(72, 0.3, dtype=np.float32), fov_deg=180.0, max_range=1.0)
    assert out[36] == pytest.approx(1.0)
    assert out[0] == pytest.approx(0.3)


def test_narrow_fov_keeps_only_the_frontmost_sectors():
    # FOV 20 deg -> |centro| <= 10 -> setores 0 (2.5), 1 (7.5), 70 (352.5), 71 (357.5)
    out = apply_fov_mask(np.full(72, 3.0, dtype=np.float32), fov_deg=20.0, max_range=12.0)
    kept = [i for i in range(72) if out[i] == pytest.approx(3.0)]
    assert kept == [0, 1, 70, 71]


def test_masked_vector_normalizes_to_free():
    # a traseira mascarada tem de virar 1.0 ("livre") depois do normalize
    from ai.shared.lidar_pipeline import normalize_sectors_m
    masked = apply_fov_mask(np.full(72, 3.0, dtype=np.float32), fov_deg=180.0, max_range=12.0)
    norm = normalize_sectors_m(masked, max_range=12.0)
    assert np.allclose(norm[18:54], 1.0)
    assert np.allclose(norm[0:18], 0.25)
