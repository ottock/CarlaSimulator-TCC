"""Closed-loop route metrics (pure, no CARLA).

The harness feeds per-step lateral deviation (metres from the lane centre) and
speed (m/s); this accumulates them and produces a per-route summary. The same
object measures the expert (Fase 0) and the model (Fase 2+), so both are judged
on identical numbers.
"""
import numpy as np


def mae(pred, target):
    """Mean absolute error between two 1-D sequences."""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if p.size == 0:
        return 0.0
    return float(np.abs(p - t).mean())


def variance_ratio(pred, target):
    """std(pred) / std(target): a collapse detector for "model always goes straight".

    A healthy steering model should keep most of the target's spread (>= ~0.6).
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    ts = t.std()
    if ts == 0:
        return 0.0
    return float(p.std() / ts)


class RouteMetrics:
    """Accumulate per-step measurements and summarize one route."""

    def __init__(self):
        self._devs = []
        self._speeds = []
        self._steers = []
        self._offlane = 0

    def add(self, deviation_m, speed_ms, departed=False, steer=None):
        """Record one control step.

        Args:
            deviation_m: lateral distance from the lane centre (m).
            speed_ms: current speed (m/s).
            departed: True if the ego actually left its lane this step. The
                caller decides this (it knows the lane width and whether the
                ego is in a junction, where centre-line distance is meaningless).
            steer: the applied steering command, if tracking smoothness. Used to
                measure swerving via ``mean_steer_jerk`` (mean absolute change
                between consecutive steering commands).
        """
        self._devs.append(float(deviation_m))
        self._speeds.append(float(speed_ms))
        if departed:
            self._offlane += 1
        if steer is not None:
            self._steers.append(float(steer))

    def summary(self):
        """Return a dict of route statistics."""
        if not self._devs:
            return {
                "steps": 0,
                "mean_dev": 0.0,
                "p95_dev": 0.0,
                "max_dev": 0.0,
                "offlane": 0,
                "mean_speed": 0.0,
                "steer_std": 0.0,
                "mean_steer_jerk": 0.0,
            }
        devs = np.asarray(self._devs, dtype=np.float64)
        speeds = np.asarray(self._speeds, dtype=np.float64)

        steer_std = 0.0
        mean_steer_jerk = 0.0
        if self._steers:
            steers = np.asarray(self._steers, dtype=np.float64)
            steer_std = float(steers.std())
            if len(steers) > 1:
                mean_steer_jerk = float(np.abs(np.diff(steers)).mean())

        return {
            "steps": len(self._devs),
            "mean_dev": float(devs.mean()),
            "p95_dev": float(np.percentile(devs, 95)),
            "max_dev": float(devs.max()),
            "offlane": self._offlane,
            "mean_speed": float(speeds.mean()),
            "steer_std": steer_std,
            "mean_steer_jerk": mean_steer_jerk,
        }
