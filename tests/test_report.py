"""Tests for the dataset quality report.

The report answers the two questions that decide if a collected dataset is
usable: is the steering distribution rich enough (not "90% straight"), and are
there braking/recovery frames? Plus basic aggregation across episodes.
"""
import numpy as np
import pytest

from ai.dataset_writer import EpisodeWriter
from ai.report import summarize_labels, steer_histogram, dataset_report


def test_summarize_counts_straight_brake_and_noise():
    labels = [
        {"steer": 0.0, "throttle": 0.5, "brake": 0.0, "v": 2.0, "noise_active": "False"},
        {"steer": 0.5, "throttle": 0.0, "brake": 0.8, "v": 0.0, "noise_active": "True"},
        {"steer": 0.01, "throttle": 0.5, "brake": 0.0, "v": 3.0, "noise_active": "False"},
    ]
    s = summarize_labels(labels)
    assert s["frames"] == 3
    assert s["pct_straight"] == pytest.approx(200.0 / 3)   # 2 of 3 with |steer| < 0.05
    assert s["pct_brake"] == pytest.approx(100.0 / 3)      # 1 of 3 with brake > 0.05
    assert s["pct_noise"] == pytest.approx(100.0 / 3)


def test_histogram_bins_and_total():
    steers = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    edges, counts = steer_histogram(steers, n_bins=4, lo=-1.0, hi=1.0)
    assert len(counts) == 4
    assert counts.sum() == 5


def test_dataset_report_aggregates_episodes(tmp_path):
    for e in range(2):
        w = EpisodeWriter(str(tmp_path / ("ep_%04d" % e)))
        for _ in range(5):
            w.add(
                np.zeros((8, 8, 3), dtype=np.uint8),
                np.zeros(72, dtype=np.float32),
                {"steer": 0.0, "throttle": 0.5, "brake": 0.0, "v": 1.0,
                 "x": 0, "y": 0, "yaw": 0, "noise_active": False},
            )
        w.close()
    rep = dataset_report(str(tmp_path))
    assert rep["num_episodes"] == 2
    assert rep["frames"] == 10
