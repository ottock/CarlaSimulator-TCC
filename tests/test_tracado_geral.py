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
    assert np.allclose(c[0, :2], eixo[0, :2], atol=1e-9)
    assert abs(c[1, 1] - eixo[0, 1]) < 1e-9                 # segundo ponto ainda no eixo


def test_an_impossible_corridor_is_refused():
    with pytest.raises(ValueError):
        tracado_geral(_eixo("grade:SSS"), False, 1.0, W, L, MARGEM, raio_ref=R_CARRO)


def test_expert_path_offers_the_general_line_for_open_tracks():
    from ai.racing_line import expert_path
    eixo = [tuple(r) for r in _eixo("grade:SDSES", espac=0.5)]
    c = expert_path(eixo, HW, W, L, margin=MARGEM, min_radius_required=R_CARRO,
                    modo="geral", fechado=False)
    assert len(c) > len(eixo)                    # reamostrado de 0,5 m para ~0,26 m
    with pytest.raises(ValueError):
        expert_path(eixo, HW, W, L, margin=MARGEM, modo="estadio", fechado=False)


def test_the_line_has_the_stadiums_resolution_whatever_the_axis_spacing():
    # Com o eixo a cada 0,5 m (o da coleta) o traçado saia com metade dos pontos
    # do estadio, e o expert batia bem mais. O passo agora e do traçado.
    grosso = tracado_geral(_eixo("oval_tcc", espac=0.5), True, HW, W, L, MARGEM, raio_ref=R_CARRO)
    fino = tracado_geral(_eixo("oval_tcc", espac=0.25), True, HW, W, L, MARGEM, raio_ref=R_CARRO)
    assert abs(len(grosso) - len(fino)) <= 2
    passo = np.hypot(*np.diff(grosso[:, :2], axis=0).T)
    assert passo.max() < 0.35                  # antes: ate ~0,6 m por fora da curva
    assert _uso(grosso, True) == pytest.approx(_uso(fino, True), abs=0.01)


# ------------------------------------------------- margem so do lado de dentro
# Pedido do Rafael em 2026-10-17: o v5d1 passa perto da parede de DENTRO da
# curva. Mais margem dos dois lados aperta as curvas (a pista mais apertada ja
# usa 84% do esterco com 3 cm); mais margem so do lado de dentro ataca o ponto
# certo. "Dentro" e o lado para onde o EIXO da pista curva naquele ponto.

APERTADA = "SDSESESES"          # a montagem do teste real de 2026-10-12, 4 curvas


def _curva_do_eixo(eixo, fechado):
    from ai.tracado_geral import _curvaturas, _reamostra
    e = _reamostra(eixo, HW / 12.0, fechado)
    return e, _curvaturas(e[:, :2], fechado)


def test_an_inner_margin_equal_to_the_margin_gives_todays_line():
    eixo = _eixo("grade:" + APERTADA)
    hoje = tracado_geral(eixo, False, HW, W, L, MARGEM, raio_ref=R_CARRO)
    igual = tracado_geral(eixo, False, HW, W, L, MARGEM, raio_ref=R_CARRO,
                          margin_dentro=MARGEM)
    assert np.allclose(hoje, igual)


def test_the_inner_margin_keeps_the_car_farther_from_the_inner_wall():
    from ai.tracado_geral import folga_por_lado
    eixo = _eixo("grade:" + APERTADA)
    # 7 cm no carro. Com a margem de 3 cm a linha ja passa a 5,4 cm da parede de
    # dentro nessa pista (medido em 2026-10-17): um valor menor nao distinguiria
    # nada, e o teste passaria com a folga extra do lado errado.
    dentro = 0.84
    c = tracado_geral(eixo, False, HW, W, L, MARGEM, raio_ref=R_CARRO, margin_dentro=dentro)
    esq, dir_ = folga_por_lado(c, eixo, False, HW, W, L, ignorar_fim=L)
    e, k = _curva_do_eixo(eixo, False)
    # Cada ponto do traçado pelo trecho do eixo mais proximo: curva a esquerda
    # (k > 0) tem a parede de dentro a esquerda.
    perto = np.array([np.argmin(np.hypot(*(e[:, :2] - p[:2]).T)) for p in c])
    na_curva = np.abs(k[perto]) > 0.5 * np.abs(k).max()
    interna = np.where(k[perto] > 0, esq, dir_)[na_curva]
    externa = np.where(k[perto] > 0, dir_, esq)[na_curva]
    assert np.isfinite(interna).any()
    assert interna[np.isfinite(interna)].min() > dentro - 0.02
    assert externa[np.isfinite(externa)].min() > MARGEM - 0.02
    # E o lado de FORA continua podendo ir ate a margem de sempre: a folga extra
    # e so de um lado. (Se fosse dos dois, a curva inteira ficaria mais estreita.)
    assert externa[np.isfinite(externa)].min() < dentro - 0.05


def test_the_side_clearances_agree_with_the_exact_clearance(oval):
    from ai.tracado_geral import folga_por_lado
    eixo, geral, _ = oval
    esq, dir_ = folga_por_lado(geral, eixo, True, HW, W, L)
    assert np.allclose(np.minimum(esq, dir_), folga_exata(geral, eixo, True, HW, W, L))


def test_an_inner_margin_smaller_than_the_margin_is_refused():
    with pytest.raises(ValueError):
        tracado_geral(_eixo("grade:SDSES"), False, HW, W, L, MARGEM, raio_ref=R_CARRO,
                      margin_dentro=MARGEM - 0.1)


def test_an_inner_margin_that_leaves_no_corridor_is_refused():
    with pytest.raises(ValueError):
        tracado_geral(_eixo("grade:SDSES"), False, HW, W, L, MARGEM, raio_ref=R_CARRO,
                      margin_dentro=HW)


def test_the_inner_margin_does_not_squeeze_the_straights():
    # Nas retas a linha continua podendo ir ate a margem de sempre (e o que ela
    # faz antes de cada curva, por fora). Se a folga extra valesse na reta, seria
    # a margem maior dos dois lados -- a que aperta as curvas.
    from ai.tracado_geral import folga_por_lado
    eixo = _eixo("grade:" + APERTADA)
    dentro = 0.84
    c = tracado_geral(eixo, False, HW, W, L, MARGEM, raio_ref=R_CARRO, margin_dentro=dentro)
    esq, dir_ = folga_por_lado(c, eixo, False, HW, W, L, ignorar_fim=L)
    e, k = _curva_do_eixo(eixo, False)
    perto = np.array([np.argmin(np.hypot(*(e[:, :2] - p[:2]).T)) for p in c])
    na_reta = np.abs(k[perto]) < 0.1 * np.abs(k).max()
    folga_reta = np.minimum(esq, dir_)[na_reta]
    assert folga_reta[np.isfinite(folga_reta)].min() < dentro - 0.05


def test_the_inner_margin_costs_less_steering_than_the_same_margin_on_both_sides():
    # E o motivo de existir: na pista do teste real, 7 cm so do lado de dentro
    # pedem ~90% do esterco; os mesmos 7 cm dos dois lados, ~94%.
    eixo = _eixo("grade:" + APERTADA)
    so_dentro = tracado_geral(eixo, False, HW, W, L, MARGEM, raio_ref=R_CARRO,
                              margin_dentro=0.84)
    dois_lados = tracado_geral(eixo, False, HW, W, L, 0.84, raio_ref=R_CARRO)
    assert _uso(so_dentro, False) < _uso(dois_lados, False) - 0.02


def test_expert_path_passes_the_inner_margin_to_the_general_line():
    from ai.racing_line import expert_path
    eixo = _eixo("grade:" + APERTADA, espac=0.5)
    c = expert_path([tuple(r) for r in eixo], HW, W, L, margin=MARGEM,
                    min_radius_required=R_CARRO, modo="geral", fechado=False,
                    margin_dentro=0.84)
    direto = tracado_geral(eixo, False, HW, W, L, MARGEM, raio_ref=R_CARRO, margin_dentro=0.84)
    assert np.allclose(np.array(c), direto)


def test_the_stadium_has_no_inner_margin():
    # O estadio (o traçado do v4) nao sabe fazer margem por lado: aceitar e
    # ignorar faria uma coleta "com margem de dentro" sair igual a de sempre.
    from ai.racing_line import expert_path
    eixo = [tuple(r) for r in _eixo("oval_tcc", espac=0.5)]
    with pytest.raises(ValueError):
        expert_path(eixo, HW, W, L, margin=MARGEM, modo="estadio", fechado=True,
                    margin_dentro=0.84)
