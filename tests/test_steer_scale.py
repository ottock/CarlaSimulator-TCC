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
    CAR_MAX_STEER_DEG,
    CAR_WHEELBASE_M,
    CAR_WIDTH_M,
    LANE_WIDTH_M,
    SCALE,
    corner_fits,
    max_usable_radius_m,
    sim_max_steer_deg,
    steer_deg_for_radius,
    turning_radius_m,
)


def test_the_measured_car_constants_are_the_ones_we_measured():
    assert CAR_MAX_STEER_DEG == pytest.approx(27.0)
    assert CAR_WHEELBASE_M == pytest.approx(0.21)
    assert CAR_WIDTH_M == pytest.approx(0.208)
    assert SCALE == pytest.approx(12.0)


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
    # O numero que vai para o CARLA. Com o Tesla de 2.875 m, igualar o raio
    # escalado do carro (0.412 x 12 = 4.95 m) pede ~30.2 graus, nao os 27.
    deg = sim_max_steer_deg()
    assert deg == pytest.approx(30.2, abs=0.1)
    assert deg > CAR_MAX_STEER_DEG      # o Tesla e mais comprido, precisa de mais


def test_the_calibrated_sim_reproduces_the_cars_minimum_radius():
    deg = sim_max_steer_deg()
    r_sim = turning_radius_m(2.875, deg)
    r_car = turning_radius_m(CAR_WHEELBASE_M, CAR_MAX_STEER_DEG)
    assert r_sim / SCALE == pytest.approx(r_car, rel=1e-3)


def test_a_vehicle_scaled_exactly_needs_the_cars_own_angle():
    # Controle de sanidade da formula: se o carro do sim fosse o WLtoys x12,
    # o angulo teria de ser identico.
    deg = sim_max_steer_deg(sim_wheelbase_m=CAR_WHEELBASE_M * SCALE)
    assert deg == pytest.approx(CAR_MAX_STEER_DEG)


def test_the_usable_radius_is_limited_by_the_body_not_the_lane():
    # A curva e 1/4 de disco com raio interno ZERO, entao o limite e so a parede
    # externa: o eixo do carro pode ir ate a largura da faixa menos meia largura.
    assert max_usable_radius_m() == pytest.approx(LANE_WIDTH_M - CAR_WIDTH_M / 2)
    assert max_usable_radius_m() == pytest.approx(0.426, abs=1e-3)


def test_the_measured_car_fits_the_corner_but_barely():
    r_min = turning_radius_m(CAR_WHEELBASE_M, CAR_MAX_STEER_DEG)
    assert corner_fits(r_min) is True
    folga = max_usable_radius_m() - r_min
    assert folga == pytest.approx(0.014, abs=2e-3)      # 1,4 cm


def test_a_weaker_steering_would_not_fit_and_the_check_says_so():
    # Com 24 graus (o valor aparente medido em velocidade alta) nao caberia.
    # E este teste que impede alguem de coletar um dataset impossivel.
    assert corner_fits(turning_radius_m(CAR_WHEELBASE_M, 24.0)) is False


def test_the_calibrated_sim_car_also_fits_its_own_corner():
    # Se o carro do sim NAO couber na curva do sim, o expert bate e o dataset
    # inteiro nao presta. Tesla: 2.0 m de largura, faixa de 6.36 m.
    r_sim = turning_radius_m(2.875, sim_max_steer_deg())
    assert corner_fits(r_sim, lane_width_m=LANE_WIDTH_M * SCALE,
                       vehicle_width_m=2.0) is True


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
