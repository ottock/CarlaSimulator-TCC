"""Tests for the trained model wrapped as a closed-loop policy (Fase 2).

Same call signature as the expert: ``policy(obs) -> (steer, throttle, brake)``,
so it drops into the existing harness. Camera-only: the model sets steering; a
fixed throttle keeps the car moving so lane-keeping can be judged.
"""
import numpy as np
import torch

from ai.model import CameraSteeringNet
from ai.model_policy import ModelSteeringPolicy


def _save_ckpt(path):
    net = CameraSteeringNet()
    net(torch.zeros(1, 3, 66, 200))  # init lazy layer
    torch.save({"model_state_dict": net.state_dict()}, str(path))


def test_policy_returns_control_tuple(tmp_path):
    ckpt = tmp_path / "m.pt"
    _save_ckpt(ckpt)
    policy = ModelSteeringPolicy(str(ckpt), throttle=0.3, device="cpu")
    steer, throttle, brake = policy({"image": np.zeros((360, 640, 3), dtype=np.uint8)})
    assert -1.0 <= steer <= 1.0
    assert throttle == 0.3
    assert brake == 0.0


def test_policy_handles_missing_image(tmp_path):
    ckpt = tmp_path / "m.pt"
    _save_ckpt(ckpt)
    policy = ModelSteeringPolicy(str(ckpt), device="cpu")
    assert policy({"image": None}) == (0.0, 0.0, 0.0)
