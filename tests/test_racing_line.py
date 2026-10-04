"""Traçado viavel dentro do corredor (Fase 6c).

O expert sempre seguiu o eixo da pista e o carro nunca conseguiu segui-lo: raio
minimo medido 0,625 m contra 0,426 m de maior arco concentrico numa faixa de
0,53. Estes testes guardam as duas propriedades que fazem o traçado prestar:
a carroceria nao sai da faixa, e a curvatura cai o bastante para o carro
executar.
"""
import math

import numpy as np
import pytest

from ai.racing_line import (
    fits,
    swept_overhang,
    menger_curvature,
    min_radius_m,
    oval_dims,
    path_curvatures,
    stadium_path,
)

CARRO_SIM_M = 0.21 * 12       # largura do WLtoys na escala do simulador
COMP_SIM_M = 0.36 * 12        # comprimento
RAIO_CARRO_SIM_M = 0.470 * 12  # 5,64 m -- raio minimo medido (circulo de 94 cm)


def _circulo(r, n=120):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([r * np.cos(a), r * np.sin(a)], axis=1)


def _oval():
    from ai.track_ref import track_centerline, track_width
    cfg = {"preset": "oval_tcc", "escala": "real", "flip_y": -1, "flip_yaw": -1,
           "professor": {"espacamento": 0.5}}
    return track_centerline(cfg), track_width(cfg) / 2.0


# ---------------------------------------------------------------------------
# Curvatura
# ---------------------------------------------------------------------------

def test_menger_curvature_is_one_over_the_radius():
    for r in (0.5, 3.18, 50.0):
        c = _circulo(r, 360)
        assert menger_curvature(c[0], c[1], c[2]) == pytest.approx(1.0 / r, rel=1e-3)


def test_collinear_or_coincident_points_have_no_curvature():
    assert menger_curvature((0, 0), (1, 0), (2, 0)) == 0.0
    assert menger_curvature((0, 0), (0, 0), (2, 0)) == 0.0


def test_min_radius_of_a_circle_is_its_radius():
    assert min_radius_m(_circulo(7.5)) == pytest.approx(7.5, rel=1e-3)


def test_the_closing_duplicate_point_does_not_blow_up_the_curvature():
    # track_centerline repete o primeiro ponto no fim. Num calculo circular isso
    # vira um triangulo degenerado e a curvatura dos VIZINHOS explode -- custou
    # um bom tempo de depuracao, entao fica travado por teste.
    c = _circulo(7.5, 120)
    com_repetido = np.vstack([c, c[:1]])
    assert min_radius_m(com_repetido) == pytest.approx(7.5, rel=1e-3)


# ---------------------------------------------------------------------------
# Geometria do oval
# ---------------------------------------------------------------------------

def test_oval_dims_reads_the_semi_axes_and_the_centre():
    eixo, _ = _oval()
    a, b, cx, cy = oval_dims(eixo)
    assert 2 * a == pytest.approx(19.08, abs=0.02)   # 159 cm na escala 1:12
    assert 2 * b == pytest.approx(12.72, abs=0.02)   # 106 cm
    assert a > b


def test_a_square_loop_is_refused_because_the_stadium_has_no_long_axis():
    with pytest.raises(ValueError):
        oval_dims(_circulo(10.0, 80))


# ---------------------------------------------------------------------------
# O traçado
# ---------------------------------------------------------------------------

def test_the_centerline_of_the_oval_is_NOT_drivable_by_this_car():
    # O ponto de partida de tudo.
    eixo, _ = _oval()
    assert min_radius_m(eixo) < RAIO_CARRO_SIM_M


def test_the_stadium_makes_the_oval_drivable():
    eixo, half = _oval()
    p = stadium_path(eixo, half, CARRO_SIM_M, COMP_SIM_M)
    assert min_radius_m(p[:, :2]) >= RAIO_CARRO_SIM_M, (
        "ainda exige raio menor que o do carro: %.2f m" % min_radius_m(p[:, :2]))


def test_the_body_never_leaves_the_lane():
    eixo, half = _oval()
    p = stadium_path(eixo, half, CARRO_SIM_M, COMP_SIM_M)
    assert fits(p[:, :2], eixo, half, CARRO_SIM_M, COMP_SIM_M)


def test_the_stadium_is_two_arcs_and_two_straights():
    # Assinatura do estadio: curvatura ou ~0 (reta) ou no patamar do raio (ponta),
    # quase nada no meio. Se fossem quatro curvas separadas haveria quatro picos.
    eixo, half = _oval()
    p = stadium_path(eixo, half, CARRO_SIM_M, COMP_SIM_M)
    k = path_curvatures(p[:, :2])
    alvo = 1.0 / min_radius_m(p[:, :2])
    reto = (k < 0.05 * alvo).sum()
    ponta = (k > 0.95 * alvo).sum()
    assert reto + ponta > 0.95 * len(k), "curvatura dispersa: nao e um estadio"
    assert reto > 0.15 * len(k) and ponta > 0.4 * len(k)


def test_the_path_uses_the_corridor_instead_of_hugging_the_centerline():
    # Colar no eixo seria a trajetoria impossivel de sempre; colar na parede
    # externa daria so 5,36 m de raio. O estadio tem de fazer melhor que os dois.
    eixo, half = _oval()
    p = stadium_path(eixo, half, CARRO_SIM_M, COMP_SIM_M)
    folga = half - CARRO_SIM_M / 2
    raio_colado_na_parede = min_radius_m(eixo) + folga
    assert min_radius_m(p[:, :2]) > raio_colado_na_parede


def test_the_path_runs_the_same_way_round_as_the_centerline():
    # Invertido, o Pure Pursuit faria o carro percorrer a pista ao contrario.
    eixo, half = _oval()
    p = stadium_path(eixo, half, CARRO_SIM_M, COMP_SIM_M)
    area = lambda q: float(np.sum(q[:, 0] * np.roll(q[:, 1], -1)
                                  - np.roll(q[:, 0], -1) * q[:, 1]))
    c = np.array([(x, y) for x, y, _ in eixo])
    assert np.sign(area(c)) == np.sign(area(p[:, :2]))


def test_the_yaw_column_matches_the_direction_of_travel():
    eixo, half = _oval()
    p = stadium_path(eixo, half, CARRO_SIM_M, COMP_SIM_M)
    for i in range(0, len(p), 17):
        j = (i + 1) % len(p)
        avanco = math.atan2(p[j][1] - p[i][1], p[j][0] - p[i][0])
        erro = abs(math.atan2(math.sin(avanco - p[i][2]), math.cos(avanco - p[i][2])))
        assert erro < 0.08, "yaw nao acompanha a marcha no ponto %d" % i


def test_a_vehicle_wider_than_the_lane_is_refused():
    eixo, half = _oval()
    with pytest.raises(ValueError):
        stadium_path(eixo, half, vehicle_width=4 * half, vehicle_length=COMP_SIM_M)


def test_a_wider_car_gets_a_tighter_path():
    # Menos corredor util -> raio menor. Se nao mudasse, a largura estaria
    # sendo ignorada e o traçado encostaria a carroceria na parede.
    eixo, half = _oval()
    estreito = stadium_path(eixo, half, 1.0, COMP_SIM_M)
    largo = stadium_path(eixo, half, 3.0, COMP_SIM_M)
    assert min_radius_m(estreito[:, :2]) > min_radius_m(largo[:, :2])


# ---------------------------------------------------------------------------
# A porta antes de coletar
# ---------------------------------------------------------------------------

from ai.racing_line import expert_path


def test_the_expert_path_defaults_to_the_stadium():
    eixo, half = _oval()
    p = expert_path(eixo, half, CARRO_SIM_M, COMP_SIM_M)
    assert min_radius_m([(x, y) for x, y, _ in p]) >= RAIO_CARRO_SIM_M


def test_the_centerline_mode_still_exists_for_the_old_tracks():
    eixo, half = _oval()
    p = expert_path(eixo, half, CARRO_SIM_M, COMP_SIM_M, modo="eixo")
    assert len(p) == len(eixo)
    assert p[0][:2] == pytest.approx(tuple(eixo[0][:2]))


def test_an_impossible_path_is_REFUSED_instead_of_collected():
    # Esta e a porta que nao existia. O dataset_track_v1 inteiro foi coletado
    # sobre o eixo, que exige 3,18 m de raio contra os 7,50 do carro, e o
    # problema so apareceu meses depois no asfalto.
    eixo, half = _oval()
    with pytest.raises(ValueError, match="inexecutavel"):
        expert_path(eixo, half, CARRO_SIM_M, COMP_SIM_M,
                    min_radius_required=RAIO_CARRO_SIM_M, modo="eixo")


def test_the_stadium_passes_the_same_gate():
    eixo, half = _oval()
    p = expert_path(eixo, half, CARRO_SIM_M, COMP_SIM_M, min_radius_required=RAIO_CARRO_SIM_M)
    assert len(p) == 240


def test_an_unknown_mode_is_refused():
    eixo, half = _oval()
    with pytest.raises(ValueError):
        expert_path(eixo, half, CARRO_SIM_M, COMP_SIM_M, modo="racing")


def test_every_point_carries_a_yaw():
    eixo, half = _oval()
    for modo in ("estadio", "eixo"):
        assert all(len(p) == 3 for p in expert_path(eixo, half, CARRO_SIM_M, COMP_SIM_M, modo=modo))


# ---------------------------------------------------------------------------
# Varredura: a quina, nao a lateral
# ---------------------------------------------------------------------------

def test_a_long_car_sweeps_wider_than_its_own_body():
    # Foi este o erro que fez o expert bater: usei meia-largura como folga.
    # Para o WLtoys escalado num raio de 8 m sao 1,51 m contra 1,26 -- 25 cm.
    assert swept_overhang(8.0, length=COMP_SIM_M, width=CARRO_SIM_M) == pytest.approx(1.51, abs=0.01)
    assert swept_overhang(8.0, COMP_SIM_M, CARRO_SIM_M) > CARRO_SIM_M / 2


def test_a_point_vehicle_sweeps_nothing_extra():
    assert swept_overhang(8.0, length=0.0, width=0.0) == pytest.approx(0.0)


def test_length_tightens_the_path_and_the_old_model_was_too_optimistic():
    eixo, half = _oval()
    sem = stadium_path(eixo, half, CARRO_SIM_M, vehicle_length=0.0)
    com = stadium_path(eixo, half, CARRO_SIM_M, vehicle_length=COMP_SIM_M)
    assert min_radius_m(com[:, :2]) < min_radius_m(sem[:, :2])
    # e o traçado antigo NAO cabia de verdade -- foi o que o CARLA mostrou
    assert not fits(sem[:, :2], eixo, half, CARRO_SIM_M, COMP_SIM_M)
    assert fits(com[:, :2], eixo, half, CARRO_SIM_M, COMP_SIM_M)


def test_the_corrected_path_is_still_drivable_by_the_car():
    eixo, half = _oval()
    p = stadium_path(eixo, half, CARRO_SIM_M, COMP_SIM_M)
    assert min_radius_m(p[:, :2]) >= RAIO_CARRO_SIM_M


# ---------------------------------------------------------------------------
# Empurrao de recuperacao: o espaco e assimetrico
# ---------------------------------------------------------------------------

from ai.racing_line import nudge_bounds, signed_offsets


def test_the_room_around_the_racing_line_is_asymmetric():
    # O traçado encosta num lado do corredor, entao quase todo o espaco sobra
    # do outro. Tratar como simetrico foi o que pos a carroceria na parede.
    lo, hi = nudge_bounds(path_offset=1.93, half_width=3.18, vehicle_width=2.0)
    assert hi == pytest.approx(0.25, abs=0.01)
    assert lo == pytest.approx(-4.11, abs=0.01)


def test_a_path_on_the_centerline_has_symmetric_room():
    lo, hi = nudge_bounds(0.0, half_width=3.18, vehicle_width=2.0)
    assert lo == pytest.approx(-hi)


def test_the_bounds_keep_the_body_inside_the_lane():
    half, veh = 3.18, 2.0
    for s in (-2.0, -0.5, 0.0, 1.5, 1.93):
        lo, hi = nudge_bounds(s, half, veh)
        for delta in (lo, hi):
            assert abs(s + delta) <= half - veh / 2 + 1e-9


def test_a_vehicle_wider_than_the_lane_is_refused():
    with pytest.raises(ValueError):
        nudge_bounds(0.0, half_width=0.5, vehicle_width=2.0)


def test_signed_offsets_say_which_SIDE_not_just_how_far():
    eixo, half = _oval()
    p = stadium_path(eixo, half, CARRO_SIM_M, COMP_SIM_M)
    s = signed_offsets(p[:, :2], eixo)
    # O estadio fica todo de um lado do eixo: mesmo sinal em todo lugar.
    assert (s > 0).all() or (s < 0).all()
    assert np.abs(s).max() == pytest.approx(
        np.abs(signed_offsets(p[:, :2], eixo)).max())


def test_lateral_offset_is_signed_and_matches_the_normal():
    from ai.racing_line import lateral_offset
    eixo, _ = _oval()
    c0 = eixo[0]
    s, n = lateral_offset(c0[0], c0[1], eixo)
    assert s == pytest.approx(0.0, abs=1e-6)        # no proprio eixo
    # um metro na direcao da normal da deslocamento +1
    s2, _ = lateral_offset(c0[0] + n[0], c0[1] + n[1], eixo)
    assert s2 == pytest.approx(1.0, abs=0.05)


def test_the_nudge_displaces_from_where_the_car_IS():
    # O bug que custou duas rodadas: empurrar a partir do TRACADO teleportava o
    # carro ate la antes de somar o deslocamento. No inicio do episodio ele
    # esta no eixo, a 1,9 m do traçado, e um "empurrao de 0,6" virava um salto
    # de 1,4 m que punha a carroceria 25 cm dentro da parede.
    from ai.racing_line import lateral_offset, nudge_bounds
    eixo, half = _oval()
    c0 = eixo[0]
    s, _ = lateral_offset(c0[0], c0[1], eixo)
    lo, hi = nudge_bounds(s, half, CARRO_SIM_M)
    # a partir do eixo o espaco e simetrico, e um empurrao de 0,6 cabe nos dois
    assert lo <= -0.6 and hi >= 0.6
