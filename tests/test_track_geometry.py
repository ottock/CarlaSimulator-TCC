"""Geometria das pistas custom (matematica pura, sem CARLA).

Garante que as pistas do estande FECHAM (o laco volta ao inicio) — em especial a
pista1, cujo residuo de fechamento escala com o `fator`: o `fechar_gap` precisa
distribuir esse residuo tambem em escala real (nao so a 1/12).
"""
import math

from core.carlaClient import track_builder as tb


def _centerline_closure(preset, fator):
    """Distancia entre o ULTIMO e o PRIMEIRO waypoint da linha de centro densa."""
    wps = tb.gerar_waypoints(preset, fator, 0.5)
    return math.hypot(wps[-1]["x"] - wps[0]["x"], wps[-1]["y"] - wps[0]["y"])


def test_stand_tracks_close_at_real_scale():
    # Em escala real (fator 12) as 3 pistas do estande fecham (< 15 cm).
    for preset in ("pista1", "pista2", "pista3"):
        gap = _centerline_closure(preset, 12.0)
        assert gap < 0.15, "%s nao fecha em escala real: gap=%.3f m" % (preset, gap)


def test_pista1_closes_both_scales():
    # pista1 fecha tanto a 1/12 quanto em escala real (o fix do limiar por fator).
    assert _centerline_closure("pista1", 1.0) < 0.05
    assert _centerline_closure("pista1", 12.0) < 0.15


def test_geometric_presets_close():
    for preset in ("oval", "quadrado", "octogono"):
        assert _centerline_closure(preset, 12.0) < 0.05


def test_curva_esquerda_vs_direita():
    # A mesma malha vira p/ lados opostos conforme o conector (esq +90, dir -90).
    c = tb._conectores(1.0)
    dh_esq = c["tcc_curva90"]["h_out"] - c["tcc_curva90"]["h_in"]
    dh_dir = c["tcc_curva90_r"]["h_out"] - c["tcc_curva90_r"]["h_in"]
    assert dh_esq > 0 and dh_dir < 0


def test_waypoints_tem_campos_e_densidade():
    wps = tb.gerar_waypoints("pista2", 12.0, 0.5)
    assert len(wps) > 50
    assert all({"x", "y", "heading", "s"} <= set(w) for w in wps)
    assert wps[-1]["s"] > wps[0]["s"]  # distancia acumulada cresce
