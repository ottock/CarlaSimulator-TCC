"""Mapeamento da saida do modelo para o servo do carro (Fase 6b).

O modelo devolve steer em [-1, 1]; o servo fala microssegundos. Este mapa e o
ultimo ponto antes do hardware, entao ele tem de ser seguro mesmo recebendo lixo:
um NaN ou um valor fora de faixa NAO pode virar um comando extremo no servo.
"""

from ai.car.control_map import (
    ESC_NEUTRAL_US,
    STEER_CENTER_US,
    STEER_LEFT_US,
    STEER_RIGHT_US,
    FrameWatchdog,
    steer_to_us,
)


def test_zero_steer_is_centre():
    assert steer_to_us(0.0) == 1500


def test_full_right_matches_carla_convention():
    # CARLA: steer +1 = direita. O MASTER do hardware: 1700 us = direita.
    assert steer_to_us(1.0) == STEER_RIGHT_US == 1700


def test_full_left():
    assert steer_to_us(-1.0) == STEER_LEFT_US == 1300


def test_half_right_is_linear():
    assert steer_to_us(0.5) == 1600


def test_out_of_range_is_clamped_not_wrapped():
    # A cabeca de steer usa tanh, entao nao deveria estourar -- mas se estourar,
    # o servo nao pode receber um comando alem do batente mecanico.
    assert steer_to_us(5.0) == STEER_RIGHT_US
    assert steer_to_us(-5.0) == STEER_LEFT_US


def test_nan_is_centre():
    assert steer_to_us(float("nan")) == STEER_CENTER_US


def test_none_is_centre():
    assert steer_to_us(None) == STEER_CENTER_US


def test_positive_infinity_is_hard_right():
    # Infinity should not crash; +inf clamped to 1.0 means hard right.
    assert steer_to_us(float("inf")) == STEER_RIGHT_US


def test_negative_infinity_is_hard_left():
    # Infinity should not crash; -inf clamped to -1.0 means hard left.
    assert steer_to_us(float("-inf")) == STEER_LEFT_US


def test_esc_neutral_constant():
    assert ESC_NEUTRAL_US == 1500


def test_watchdog_does_not_fire_on_the_first_tick():
    wd = FrameWatchdog(timeout_s=0.25)
    assert wd.tick(100.0) is False


def test_watchdog_quiet_when_frames_are_fast():
    wd = FrameWatchdog(timeout_s=0.25)
    wd.tick(100.0)
    assert wd.tick(100.05) is False


def test_watchdog_fires_when_a_frame_stalls():
    wd = FrameWatchdog(timeout_s=0.25)
    wd.tick(100.0)
    assert wd.tick(100.5) is True


def test_watchdog_recovers_after_a_stall():
    wd = FrameWatchdog(timeout_s=0.25)
    wd.tick(100.0)
    wd.tick(100.5)
    assert wd.tick(100.55) is False


# ---------------------------------------------------------------------------
# Velocidade de cruzeiro constante (Fase 6b parte 2)
#
# Decisao do Rafael: velocidade CONSTANTE, sem aceleracao; e se ela nao puder ser
# constante, entao ZERO. O modelo comanda so o esterco -- as cabecas de throttle e
# brake sao ignoradas (a de freio esta inerte, o dataset tem 0% de frenagem).
# ---------------------------------------------------------------------------
import numpy as np
import pytest

from ai.car.control_map import (
    ESC_MAX_US, ESC_MIN_MOVE_US, clamp_cruise_us, front_blocked, front_min,
)


def test_cruise_passes_through_a_valid_value():
    assert clamp_cruise_us(1650) == 1650


def test_cruise_never_exceeds_the_calibrated_maximum():
    assert clamp_cruise_us(1800) == ESC_MAX_US
    assert clamp_cruise_us(9999) == ESC_MAX_US


def test_cruise_inside_the_esc_deadzone_becomes_neutral():
    # O ESC so anda a partir de 1600 us. Pedir 1550 nao moveria o carro; devolver
    # neutro mantem "comandado" e "real" coerentes em vez de fingir um comando.
    assert clamp_cruise_us(1550) == ESC_NEUTRAL_US
    assert clamp_cruise_us(ESC_MIN_MOVE_US) == ESC_MIN_MOVE_US


def test_cruise_never_goes_backwards():
    # Re nao faz parte desta fase: qualquer valor abaixo do neutro vira neutro.
    assert clamp_cruise_us(1400) == ESC_NEUTRAL_US
    assert clamp_cruise_us(0) == ESC_NEUTRAL_US


def test_cruise_with_garbage_is_neutral():
    assert clamp_cruise_us(None) == ESC_NEUTRAL_US
    assert clamp_cruise_us(float("nan")) == ESC_NEUTRAL_US
    assert clamp_cruise_us("rapido") == ESC_NEUTRAL_US
    assert clamp_cruise_us(float("inf")) == ESC_MAX_US


def test_front_min_looks_only_at_the_frontal_cone():
    v = np.ones(72, dtype=np.float32)
    v[0] = 0.2                                  # setor 0 = 2.5 graus, dentro do cone
    assert front_min(v) == pytest.approx(0.2)
    v2 = np.ones(72, dtype=np.float32)
    v2[36] = 0.05                               # 182.5 graus = atras
    assert front_min(v2) == pytest.approx(1.0)


def test_front_blocked_trips_on_something_close_ahead():
    v = np.ones(72, dtype=np.float32)
    v[0] = 0.2
    assert front_blocked(v, threshold=0.25) is True


def test_front_blocked_ignores_the_same_obstacle_behind():
    v = np.ones(72, dtype=np.float32)
    v[36] = 0.05
    assert front_blocked(v, threshold=0.25) is False


def test_front_blocked_is_false_on_a_clear_road():
    assert front_blocked(np.ones(72, dtype=np.float32), threshold=0.25) is False
