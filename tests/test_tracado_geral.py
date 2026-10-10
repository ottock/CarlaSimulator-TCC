"""Traçado geral (2026-10-09): o minimax de curvatura tem de reencontrar o estadio
no oval e dar folga real em pistas que o estadio nao resolve."""
import numpy as np
import pytest

from ai.racing_line import distance_to_centerline, min_radius_m, stadium_path
from ai.steer_scale import CAR_LENGTH_M, CAR_MIN_RADIUS_M, CAR_WIDTH_M, SCALE
from ai.tracado_geral import folga_exata, sobra, tracado_geral
from core.carlaClient.track_builder import LARGURA_PISTA, gerar_waypoints

HW = LARGURA_PISTA * SCALE / 2
L, W = CAR_LENGTH_M * SCALE, CAR_WIDTH_M * SCALE
R_CARRO = CAR_MIN_RADIUS_M * SCALE
MARGEM = 0.36                                   # a do v4: 3 cm no carro


def _eixo(preset, espac=0.25):
    return np.array([(w["x"], w["y"], w["heading"])
                     for w in gerar_waypoints(preset, SCALE, espac)])


def _uso(caminho, fechado):
    return R_CARRO / min_radius_m(caminho[:, :2], fechado=fechado)


@pytest.fixture(scope="module")
def oval():
    eixo = _eixo("oval_tcc")
    geral = tracado_geral(eixo, True, HW, W, L, MARGEM, raio_ref=R_CARRO)
    est = stadium_path([tuple(r) for r in eixo], HW, W, L, margin=MARGEM, n_points=600)
    return eixo, geral, est


def test_on_the_oval_it_finds_the_stadium_again(oval):
    # O estadio do v4 e a resposta exata no oval: raio 7,66 m (74% do esterco) e
    # 1,30 m do eixo. Sem saber nada de ovais, o minimax tem de chegar la.
    eixo, geral, est = oval
    assert _uso(geral, True) == pytest.approx(_uso(est, True), abs=0.03)
    assert distance_to_centerline(geral, eixo).max() == pytest.approx(
        distance_to_centerline(est, eixo).max(), abs=0.05)


def test_on_the_oval_the_real_clearance_is_the_margin(oval):
    eixo, geral, _ = oval
    f = folga_exata(geral, eixo, True, HW, W, L)
    assert f.min() == pytest.approx(MARGEM, abs=0.02)


def test_the_overhang_matches_the_stadiums_formula():
    from ai.racing_line import swept_overhang
    assert sobra(1 / 7.66, L, W) == pytest.approx(swept_overhang(7.66, L, W))
    assert sobra(0.0, L, W) == pytest.approx(W / 2)


@pytest.mark.parametrize("codigo", ["SDSES", "SESDSDS", "SESESESE"])
def test_tracks_the_stadium_cannot_draw(codigo):
    # S (sentidos opostos), U encadeado, e o quadrado 3 x 3 fechado.
    fechado = codigo == "SESESESE"
    eixo = _eixo("grade:" + codigo)
    c = tracado_geral(eixo, fechado, HW, W, L, MARGEM, raio_ref=R_CARRO)
    assert _uso(c, fechado) < 0.90
    f = folga_exata(c, eixo, fechado, HW, W, L, ignorar_fim=0.0 if fechado else L)
    assert f.min() > MARGEM - 0.02


def test_an_open_track_starts_on_the_axis_pointing_along_it():
    eixo = _eixo("grade:SDSES")
    c = tracado_geral(eixo, False, HW, W, L, MARGEM, raio_ref=R_CARRO)
    assert np.allclose(c[:2, :2], eixo[:2, :2], atol=1e-9)
    assert len(c) == len(eixo)


def test_an_impossible_corridor_is_refused():
    with pytest.raises(ValueError):
        tracado_geral(_eixo("grade:SSS"), False, 1.0, W, L, MARGEM, raio_ref=R_CARRO)


def test_expert_path_offers_the_general_line_for_open_tracks():
    from ai.racing_line import expert_path
    eixo = [tuple(r) for r in _eixo("grade:SDSES", espac=0.5)]
    c = expert_path(eixo, HW, W, L, margin=MARGEM, min_radius_required=R_CARRO,
                    modo="geral", fechado=False)
    assert len(c) == len(eixo)
    with pytest.raises(ValueError):
        expert_path(eixo, HW, W, L, margin=MARGEM, modo="estadio", fechado=False)
