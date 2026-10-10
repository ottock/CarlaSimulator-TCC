"""Pure Pursuit numa pista COM FIM (2026-10-09): a mira nao pode dar a volta."""
import math

import pytest

from core.carlaClient.professor import PurePursuit


def _reta(n=50, passo=0.5):
    return [(i * passo, 0.0, 0.0) for i in range(n)]


def test_in_a_loop_the_target_wraps_as_before():
    wps = _reta()
    pp = PurePursuit(wps, lookahead=4.0)
    pp.idx = len(wps) - 2
    alvo = pp._ponto_lookahead(wps[-2][0], 0.0)
    assert alvo[0] < 5.0                         # deu a volta para o comeco


def test_on_an_open_track_the_target_stops_at_the_last_point():
    wps = _reta()
    pp = PurePursuit(wps, lookahead=4.0, fechado=False)
    pp.idx = len(wps) - 2
    alvo = pp._ponto_lookahead(wps[-2][0], 0.0)
    assert alvo == wps[-1]


def test_driving_off_the_end_keeps_steering_straight():
    wps = _reta()
    pp = PurePursuit(wps, lookahead=4.0, fechado=False)
    pp.idx = len(wps) - 3
    steer, _, _ = pp.control(wps[-3][0], 0.0, 0.0, 2.0)
    assert steer == pytest.approx(0.0)


def test_remaining_distance_counts_down_to_the_end():
    wps = _reta(n=11, passo=1.0)
    pp = PurePursuit(wps, fechado=False)
    assert pp.restante() == pytest.approx(10.0)
    pp.control(7.2, 0.0, 0.0, 2.0)
    assert pp.restante() == pytest.approx(3.0)
    assert PurePursuit(wps).restante() == math.inf
