"""Tests for closed-loop route metrics accumulation.

Pure logic (no CARLA): the harness feeds per-step lateral deviation, speed, and
a `departed` flag (it alone knows the lane width and whether the ego is in a
junction). This summarizes a route. Same object measures expert and model.
"""
import pytest

from ai.metrics import RouteMetrics


def test_empty_summary_is_zeroed():
    s = RouteMetrics().summary()
    assert s["steps"] == 0
    assert s["mean_dev"] == 0.0
    assert s["max_dev"] == 0.0
    assert s["offlane"] == 0


def test_accumulates_mean_and_max():
    m = RouteMetrics()
    for d in (0.0, 1.0, 2.0):
        m.add(d, 5.0)  # departed defaults to False
    s = m.summary()
    assert s["steps"] == 3
    assert s["mean_dev"] == pytest.approx(1.0)
    assert s["max_dev"] == pytest.approx(2.0)
    assert s["mean_speed"] == pytest.approx(5.0)
    assert s["offlane"] == 0


def test_counts_departed_steps():
    m = RouteMetrics()
    m.add(0.2, 1.0, departed=False)
    m.add(2.0, 1.0, departed=True)
    m.add(2.5, 1.0, departed=True)
    assert m.summary()["offlane"] == 2


def test_p95_is_high_percentile():
    m = RouteMetrics()
    for d in range(101):  # 0..100
        m.add(float(d), 1.0)
    assert m.summary()["p95_dev"] == pytest.approx(95.0, abs=1.0)


def test_steer_smoothness_zero_for_constant_steering():
    m = RouteMetrics()
    for _ in range(5):
        m.add(0.0, 1.0, steer=0.2)
    s = m.summary()
    assert s["mean_steer_jerk"] == pytest.approx(0.0)
    assert s["steer_std"] == pytest.approx(0.0)


def test_steer_jerk_measures_left_right_oscillation():
    m = RouteMetrics()
    for i in range(6):
        m.add(0.0, 1.0, steer=(0.5 if i % 2 == 0 else -0.5))
    s = m.summary()
    # every consecutive step flips by 1.0 -> mean jerk 1.0; std of +/-0.5 is 0.5
    assert s["mean_steer_jerk"] == pytest.approx(1.0)
    assert s["steer_std"] == pytest.approx(0.5)
