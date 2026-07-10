"""Steering-noise injection for recovery data (behavioral cloning).

Emits a perturbation added to the APPLIED steering in bouts, covering roughly
``active_fraction`` of the time. Between bouts it is silent. The label written
to the dataset is always the clean expert output, so the model learns the
corrective action for the drifted states these perturbations create.

Deterministic given a seed, so a collection run is reproducible.
"""
import math
import random


class SteeringNoiseInjector:
    """Bout-based sinusoidal steering perturbation source."""

    def __init__(self, dt, active_fraction=0.3, amplitude=0.3,
                 period_range=(2.0, 4.0), bout_range=(2.0, 4.0), seed=0):
        self._dt = dt
        self._active_fraction = active_fraction
        self._amplitude = amplitude
        self._period_range = period_range
        self._bout_range = bout_range
        self._rng = random.Random(seed)

        # Start silent so the ego settles before the first perturbation.
        self._active = False
        self._ticks_left = self._sample_idle_ticks()
        self._t_in_bout = 0.0
        self._period = 1.0
        self._phase = 0.0
        self._bout_amplitude = 0.0

    def _sample_bout_ticks(self):
        seconds = self._rng.uniform(*self._bout_range)
        return max(1, int(round(seconds / self._dt)))

    def _sample_idle_ticks(self):
        # Sized so that, in expectation, active time is `active_fraction`.
        f = self._active_fraction
        seconds = self._rng.uniform(*self._bout_range) * (1.0 - f) / f
        return max(1, int(round(seconds / self._dt)))

    def _begin_active_bout(self):
        self._active = True
        self._ticks_left = self._sample_bout_ticks()
        self._t_in_bout = 0.0
        self._period = self._rng.uniform(*self._period_range)
        self._phase = self._rng.uniform(0.0, 2.0 * math.pi)
        self._bout_amplitude = self._rng.uniform(0.5 * self._amplitude, self._amplitude)

    def _begin_idle(self):
        self._active = False
        self._ticks_left = self._sample_idle_ticks()

    def step(self):
        """Advance one tick; return ``(perturbation, active)``."""
        if self._ticks_left <= 0:
            if self._active:
                self._begin_idle()
            else:
                self._begin_active_bout()

        self._ticks_left -= 1

        if not self._active:
            return 0.0, False

        value = self._bout_amplitude * math.sin(
            2.0 * math.pi * self._t_in_bout / self._period + self._phase
        )
        self._t_in_bout += self._dt
        # Clamp to the configured amplitude for safety.
        if value > self._amplitude:
            value = self._amplitude
        elif value < -self._amplitude:
            value = -self._amplitude
        return value, True
