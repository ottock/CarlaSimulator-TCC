"""Tests for steering-noise injection used during data collection.

Behavioral cloning's #1 failure is never seeing recovery: the expert stays on
the line, so the model never learns "I'm off, how do I get back?". We perturb
the APPLIED steering part of the time (the car drifts, the expert corrects) but
record the CLEAN expert output as the label. This tests the perturbation source.
"""
import numpy as np

from ai.noise import SteeringNoiseInjector


def test_deterministic_with_seed():
    a = SteeringNoiseInjector(dt=0.05, seed=7)
    b = SteeringNoiseInjector(dt=0.05, seed=7)
    seq_a = [a.step() for _ in range(500)]
    seq_b = [b.step() for _ in range(500)]
    assert seq_a == seq_b


def test_perturbation_bounded_by_amplitude():
    inj = SteeringNoiseInjector(dt=0.05, amplitude=0.3, seed=1)
    perts = [abs(inj.step()[0]) for _ in range(20000)]
    assert max(perts) <= 0.3 + 1e-9


def test_inactive_steps_have_zero_perturbation():
    inj = SteeringNoiseInjector(dt=0.05, seed=2)
    for _ in range(20000):
        pert, active = inj.step()
        if not active:
            assert pert == 0.0


def test_active_fraction_approximates_target():
    inj = SteeringNoiseInjector(dt=0.05, active_fraction=0.3, seed=3)
    active = [inj.step()[1] for _ in range(40000)]
    assert abs(np.mean(active) - 0.3) < 0.05
