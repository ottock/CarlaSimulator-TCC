"""Tests for the trained model wrapped as a closed-loop policy (Fase 2).

Same call signature as the expert: ``policy(obs) -> (steer, throttle, brake)``,
so it drops into the existing harness. Camera-only: the model sets steering; a
fixed throttle keeps the car moving so lane-keeping can be judged.
"""
import numpy as np
import pytest
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


from ai.model import DrivingNet
from ai.model_policy import DrivingPolicy


def _save_driving_ckpt(path):
    import torch
    net = DrivingNet(); net(torch.zeros(1, 3, 66, 200), torch.zeros(1, 72))
    torch.save({"model_state_dict": net.state_dict(), "arch": "DrivingNet"}, path)


def _fake_obs():
    return {"image": np.zeros((360, 640, 3), dtype=np.uint8),
            "lidar": {"points": np.array([[5.0, 0.0, 0.0], [3.0, 1.0, 0.0]], dtype=np.float32)}}


def test_driving_policy_returns_three_controls_in_range(tmp_path):
    ckpt = tmp_path / "driving.pt"; _save_driving_ckpt(ckpt)
    pol = DrivingPolicy(str(ckpt), device="cpu")
    steer, throttle, brake = pol(_fake_obs())
    assert -1.0 <= steer <= 1.0
    assert 0.0 <= throttle <= 1.0
    assert 0.0 <= brake <= 1.0


def test_driving_policy_no_image_is_safe(tmp_path):
    ckpt = tmp_path / "driving.pt"; _save_driving_ckpt(ckpt)
    pol = DrivingPolicy(str(ckpt), device="cpu")
    assert pol({"image": None, "lidar": None}) == (0.0, 0.0, 0.0)


def test_driving_policy_throttle_floor_applied_when_not_braking(tmp_path):
    ckpt = tmp_path / "driving.pt"; _save_driving_ckpt(ckpt)
    pol = DrivingPolicy(str(ckpt), device="cpu", throttle_floor=0.4, brake_eps=0.05)
    pol.model = lambda xt, lt: torch.tensor([[0.1, 0.0, 0.0]])  # brake=0.0 < eps
    steer, throttle, brake = pol(_fake_obs())
    assert brake < 0.05
    assert throttle == pytest.approx(0.4)  # floor applied


def test_driving_policy_throttle_floor_not_applied_when_braking(tmp_path):
    ckpt = tmp_path / "driving.pt"; _save_driving_ckpt(ckpt)
    pol = DrivingPolicy(str(ckpt), device="cpu", throttle_floor=0.4, brake_eps=0.05)
    pol.model = lambda xt, lt: torch.tensor([[0.0, 0.0, 0.9]])  # brake=0.9 >= eps
    steer, throttle, brake = pol(_fake_obs())
    assert throttle == pytest.approx(0.0)  # floor NOT applied


def test_driving_policy_ablate_lidar_zeros_vector(tmp_path):
    ckpt = tmp_path / "driving.pt"; _save_driving_ckpt(ckpt)
    pol = DrivingPolicy(str(ckpt), device="cpu", ablate_lidar=True)
    vec = pol._lidar_vector(_fake_obs())
    assert vec.shape == (72,)
    assert float(vec.max()) == 0.0  # LiDAR zeroed under ablation
