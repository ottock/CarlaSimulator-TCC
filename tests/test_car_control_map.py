"""Mapeamento da saida do modelo para o servo do carro (Fase 6b).

O modelo devolve steer em [-1, 1]; o servo fala microssegundos. Este mapa e o
ultimo ponto antes do hardware, entao ele tem de ser seguro mesmo recebendo lixo:
um NaN ou um valor fora de faixa NAO pode virar um comando extremo no servo.
"""
import math

import pytest

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
