"""Referencia de pista (centerline) + desvio lateral — matematica pura, sem CARLA.

Substitui o map.get_waypoint() (que nao existe na pista custom) por uma linha de
centro derivada da geometria da pista e o desvio lateral perpendicular a ela.
"""
import math

from ai.track_ref import track_centerline, track_width, deviation_from_centerline


def test_deviation_zero_on_center_and_matches_offset():
    cl = [(float(i), 0.0, 0.0) for i in range(10)]      # reta ao longo de +x, heading 0
    dev, idx = deviation_from_centerline(cl, 4.0, 0.0)
    assert dev < 1e-6 and idx == 4
    dev, idx = deviation_from_centerline(cl, 4.0, 0.3)  # 0,3 m pro lado
    assert abs(dev - 0.3) < 1e-6


def test_deviation_perpendicular_uses_heading():
    cl = [(0.0, float(i), math.pi / 2.0) for i in range(10)]  # reta ao longo de +y
    dev, _ = deviation_from_centerline(cl, 0.5, 4.0)            # offset em x = lateral
    assert abs(dev - 0.5) < 1e-6


def test_window_search_stays_local():
    cl = [(float(i), 0.0, 0.0) for i in range(100)]
    dev, idx = deviation_from_centerline(cl, 55.0, 0.0, start_idx=50, window=10)
    assert idx == 55


def test_track_centerline_and_width_real():
    cfg = {"preset": "pista2", "escala": "real", "flip_y": -1, "flip_yaw": -1}
    cl = track_centerline(cfg)
    assert len(cl) > 50
    assert all(len(p) == 3 for p in cl)
    x, y, _ = cl[10]                                    # um ponto do proprio centro
    dev, _ = deviation_from_centerline(cl, x, y)
    assert dev < 0.05
    assert abs(track_width(cfg) - 0.53 * 12.0) < 1e-6   # LARGURA_PISTA * fator
