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
    menger_curvature,
    min_radius_m,
    oval_dims,
    path_curvatures,
    stadium_path,
)

CARRO_SIM_M = 0.208 * 12      # a largura do WLtoys na escala do simulador
RAIO_CARRO_SIM_M = 0.625 * 12  # 7,5 m -- o raio minimo medido, escalado


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
    p = stadium_path(eixo, half, CARRO_SIM_M)
    assert min_radius_m(p[:, :2]) >= RAIO_CARRO_SIM_M, (
        "ainda exige raio menor que o do carro: %.2f m" % min_radius_m(p[:, :2]))


def test_the_body_never_leaves_the_lane():
    eixo, half = _oval()
    p = stadium_path(eixo, half, CARRO_SIM_M)
    assert fits(p[:, :2], eixo, half, CARRO_SIM_M)


def test_the_stadium_is_two_arcs_and_two_straights():
    # Assinatura do estadio: curvatura ou ~0 (reta) ou no patamar do raio (ponta),
    # quase nada no meio. Se fossem quatro curvas separadas haveria quatro picos.
    eixo, half = _oval()
    p = stadium_path(eixo, half, CARRO_SIM_M)
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
    p = stadium_path(eixo, half, CARRO_SIM_M)
    folga = half - CARRO_SIM_M / 2
    raio_colado_na_parede = min_radius_m(eixo) + folga
    assert min_radius_m(p[:, :2]) > raio_colado_na_parede


def test_the_path_runs_the_same_way_round_as_the_centerline():
    # Invertido, o Pure Pursuit faria o carro percorrer a pista ao contrario.
    eixo, half = _oval()
    p = stadium_path(eixo, half, CARRO_SIM_M)
    area = lambda q: float(np.sum(q[:, 0] * np.roll(q[:, 1], -1)
                                  - np.roll(q[:, 0], -1) * q[:, 1]))
    c = np.array([(x, y) for x, y, _ in eixo])
    assert np.sign(area(c)) == np.sign(area(p[:, :2]))


def test_the_yaw_column_matches_the_direction_of_travel():
    eixo, half = _oval()
    p = stadium_path(eixo, half, CARRO_SIM_M)
    for i in range(0, len(p), 17):
        j = (i + 1) % len(p)
        avanco = math.atan2(p[j][1] - p[i][1], p[j][0] - p[i][0])
        erro = abs(math.atan2(math.sin(avanco - p[i][2]), math.cos(avanco - p[i][2])))
        assert erro < 0.08, "yaw nao acompanha a marcha no ponto %d" % i


def test_a_vehicle_wider_than_the_lane_is_refused():
    eixo, half = _oval()
    with pytest.raises(ValueError):
        stadium_path(eixo, half, vehicle_width=4 * half)


def test_a_wider_car_gets_a_tighter_path():
    # Menos corredor util -> raio menor. Se nao mudasse, a largura estaria
    # sendo ignorada e o traçado encostaria a carroceria na parede.
    eixo, half = _oval()
    estreito = stadium_path(eixo, half, 1.0)
    largo = stadium_path(eixo, half, 3.0)
    assert min_radius_m(estreito[:, :2]) > min_radius_m(largo[:, :2])
