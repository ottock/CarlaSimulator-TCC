"""Calibracao do esterco entre o carro real e o simulador (Fase 6c).

Por que existe: o `steer` normalizado e dividido pelo angulo MAXIMO de roda, e
esse maximo nao estava em escala. O Tesla do CARLA tem 70 graus; o WLtoys medido
tem 27. O mesmo numero significava angulos quase 3x diferentes, entao o modelo
pedia 42 graus e o carro entregava 15 -- subesterco em tudo, sem erro nenhum no
terminal.

O que tem de casar NAO e o angulo, e o RAIO escalado: o Tesla e 14% mais
comprido que o carro multiplicado por 12, entao copiar 27 graus para o sim daria
raios diferentes. Um erro aqui so aparece depois de coletar e treinar.
"""
import math

import pytest

from ai.steer_scale import (
    CAR_MIN_RADIUS_M,
    CAR_WHEELBASE_M,
    CAR_WIDTH_M,
    LANE_WIDTH_M,
    LANE_WIDTH_SIM_M,
    SCALE,
    corner_fits,
    max_usable_radius_m,
    sim_max_steer_deg,
    steer_deg_for_radius,
    turning_radius_m,
)


def test_the_measured_car_constants_are_the_ones_we_measured():
    # 107 cm de diametro tracados no centro do eixo traseiro, com o curso
    # INTEIRO do servo (300 us). Com 260 dava 125 cm -- os 40 us de folga que
    # estavam ali custavam 13% do curso.
    assert CAR_MIN_RADIUS_M == pytest.approx(0.535)
    assert CAR_WHEELBASE_M == pytest.approx(0.21)
    assert CAR_WIDTH_M == pytest.approx(0.21)     # medido com trena, 36 x 21 cm
    assert LANE_WIDTH_M == pytest.approx(0.56)
    assert SCALE == pytest.approx(12.0)


def test_the_radius_is_the_primary_measurement_not_the_angle():
    # Um protractor deu 27 graus, que preveriam 0.412 m -- 34% errado. O carro e
    # 4WD: em trava total as dianteiras arrastam e ele abre. Vale o que ele FAZ.
    equivalente = steer_deg_for_radius(CAR_WHEELBASE_M, CAR_MIN_RADIUS_M)
    assert equivalente == pytest.approx(21.4, abs=0.1)
    assert equivalente < 27.0


def test_turning_radius_is_the_bicycle_model():
    # 45 graus com entre-eixos 1 m da raio 1 m, por definicao de tangente.
    assert turning_radius_m(1.0, 45.0) == pytest.approx(1.0)
    assert turning_radius_m(0.21, 27.0) == pytest.approx(0.4122, abs=1e-4)


def test_radius_and_angle_are_inverses():
    for deg in (5.0, 27.0, 42.0, 70.0):
        r = turning_radius_m(2.875, deg)
        assert steer_deg_for_radius(2.875, r) == pytest.approx(deg)


def test_going_straight_is_not_a_zero_radius():
    # steer 0 e raio INFINITO. Devolver 0 aqui inverteria o significado e o
    # chamador leria "curva fechadissima" onde o carro vai reto.
    assert math.isinf(turning_radius_m(2.875, 0.0))


def test_impossible_geometry_raises_instead_of_returning_nonsense():
    with pytest.raises(ValueError):
        turning_radius_m(0.0, 30.0)
    with pytest.raises(ValueError):
        turning_radius_m(2.875, 90.0)      # roda perpendicular: nao existe
    with pytest.raises(ValueError):
        turning_radius_m(2.875, -5.0)
    with pytest.raises(ValueError):
        steer_deg_for_radius(2.875, 0.0)


def test_the_sim_angle_matches_the_RADIUS_not_the_angle():
    # O numero que vai para o CARLA: igualar o raio escalado (0.535 x 12 = 6.42 m)
    # com o Tesla de 2.875 m de entre-eixos pede ~24.1 graus.
    deg = sim_max_steer_deg()
    assert deg == pytest.approx(24.1, abs=0.1)


def test_the_calibrated_sim_reproduces_the_cars_minimum_radius():
    r_sim = turning_radius_m(2.875, sim_max_steer_deg())
    assert r_sim / SCALE == pytest.approx(CAR_MIN_RADIUS_M, rel=1e-3)


def test_a_vehicle_scaled_exactly_needs_the_cars_own_angle():
    # Controle de sanidade: se o carro do sim fosse o WLtoys x12, o angulo teria
    # de ser o angulo equivalente do proprio carro.
    deg = sim_max_steer_deg(sim_wheelbase_m=CAR_WHEELBASE_M * SCALE)
    assert deg == pytest.approx(steer_deg_for_radius(CAR_WHEELBASE_M, CAR_MIN_RADIUS_M))


def test_the_usable_radius_is_limited_by_the_body_not_the_lane():
    # A curva e 1/4 de disco com raio interno ZERO, entao o limite e so a parede
    # externa: o eixo do carro pode ir ate a largura da faixa menos meia largura.
    assert max_usable_radius_m() == pytest.approx(LANE_WIDTH_M - CAR_WIDTH_M / 2)
    assert max_usable_radius_m() == pytest.approx(0.455, abs=1e-3)


def test_the_measured_car_does_NOT_fit_a_concentric_arc():
    # O resultado duro: mesmo com a faixa real de 0.56 m, seguir a curva por
    # dentro exigiria 0.456 m e o carro faz 0.625. Falta 17 cm. Por isso o
    # expert TEM de cortar o apice -- nao e preferencia de traçado.
    assert corner_fits(CAR_MIN_RADIUS_M) is False
    assert max_usable_radius_m() == pytest.approx(0.455, abs=1e-3)
    assert CAR_MIN_RADIUS_M - max_usable_radius_m() == pytest.approx(0.080, abs=2e-3)


def test_the_extra_three_centimetres_of_real_track_help_but_do_not_save_it():
    assert max_usable_radius_m(lane_width_m=0.53) == pytest.approx(0.425, abs=1e-3)
    assert corner_fits(CAR_MIN_RADIUS_M, lane_width_m=0.53) is False


def test_the_calibrated_sim_car_also_cannot_take_a_concentric_arc():
    # E CORRETO que nao caiba: o simulador agora mente menos. Foi exatamente por
    # isto que o expert travou na verificacao -- ele perseguia a linha de centro.
    r_sim = turning_radius_m(2.875, sim_max_steer_deg())
    assert corner_fits(r_sim, lane_width_m=LANE_WIDTH_SIM_M * SCALE,
                       vehicle_width_m=2.0) is False


# ---------------------------------------------------------------------------
# Quais rodas esterçam
# ---------------------------------------------------------------------------
from ai.steer_scale import steering_wheel_indices


def test_only_the_wheels_that_already_steer_are_touched():
    # Ordem das rodas no CARLA e convencao, nao garantia. Em vez de assumir
    # "as duas primeiras sao as da frente", mexemos nas que JA tem angulo --
    # assumir a ordem errada limitaria rodas traseiras e deixaria as dianteiras
    # em 70 graus, que e o bug original intacto e silencioso.
    assert steering_wheel_indices([70.0, 70.0, 0.0, 0.0]) == [0, 1]
    assert steering_wheel_indices([0.0, 0.0, 70.0, 70.0]) == [2, 3]


def test_a_vehicle_with_no_steering_wheels_is_refused():
    # Devolver lista vazia em silencio faria a calibracao nao acontecer e a
    # coleta rodar inteira com o esterco errado.
    with pytest.raises(ValueError):
        steering_wheel_indices([0.0, 0.0, 0.0, 0.0])


def test_all_wheel_steering_is_handled():
    assert steering_wheel_indices([50.0, 50.0, 10.0, 10.0]) == [0, 1, 2, 3]
