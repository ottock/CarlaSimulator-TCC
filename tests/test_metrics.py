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
