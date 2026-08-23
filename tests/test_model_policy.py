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


def test_driving_policy_residual_brake_deadzoned(tmp_path):
    # The brake head never regresses exactly to 0; a tiny residual brake applied
    # together with throttle holds the car at a standstill. Below the deadzone the
    # brake must be dropped so throttle can move the car.
    ckpt = tmp_path / "driving.pt"; _save_driving_ckpt(ckpt)
    pol = DrivingPolicy(str(ckpt), device="cpu", brake_deadzone=0.1)
    pol.model = lambda xt, lt: torch.tensor([[0.0, 0.35, 0.03]])  # residual brake < deadzone
    steer, throttle, brake = pol(_fake_obs())
    assert brake == 0.0                       # residual brake dropped
    assert throttle == pytest.approx(0.35)    # throttle preserved so the car moves


def test_driving_policy_throttle_floor_applied_when_not_braking(tmp_path):
    ckpt = tmp_path / "driving.pt"; _save_driving_ckpt(ckpt)
    pol = DrivingPolicy(str(ckpt), device="cpu", throttle_floor=0.4, brake_deadzone=0.1)
    pol.model = lambda xt, lt: torch.tensor([[0.1, 0.0, 0.0]])  # brake=0.0 < deadzone
    steer, throttle, brake = pol(_fake_obs())
    assert brake == 0.0
    assert throttle == pytest.approx(0.4)     # floor applied when not braking


def test_driving_policy_braking_cuts_throttle(tmp_path):
    # Real braking (>= deadzone) and throttle are mutually exclusive.
    ckpt = tmp_path / "driving.pt"; _save_driving_ckpt(ckpt)
    pol = DrivingPolicy(str(ckpt), device="cpu", throttle_floor=0.4, brake_deadzone=0.1)
    pol.model = lambda xt, lt: torch.tensor([[0.0, 0.6, 0.9]])  # brake=0.9 >= deadzone
    steer, throttle, brake = pol(_fake_obs())
    assert brake == pytest.approx(0.9)
    assert throttle == pytest.approx(0.0)     # throttle cut during real braking


def test_driving_policy_ablate_lidar_free_vector(tmp_path):
    ckpt = tmp_path / "driving.pt"; _save_driving_ckpt(ckpt)
    pol = DrivingPolicy(str(ckpt), device="cpu", ablate_lidar=True)
    vec = pol._lidar_vector(_fake_obs())
    assert vec.shape == (72,)
    assert bool(np.all(vec == 1.0))  # ablation feeds "clear road" (free), not "obstacle everywhere"


# --- FOV limitado (Fase 6a) ---------------------------------------------------
# O fov_deg viaja no checkpoint porque um descasamento silencioso entre a mascara
# do treino e a da inferencia entrega a rede a mesma entrada da ablacao (0/3 pistas).


def _save_driving_ckpt_fov(path, fov_deg):
    net = DrivingNet(); net(torch.zeros(1, 3, 66, 200), torch.zeros(1, 72))
    torch.save({"model_state_dict": net.state_dict(), "arch": "DrivingNet",
                "fov_deg": fov_deg}, path)


def _rear_obs():
    """Um ponto na frente (5 m) e outro atras (2 m, x negativo)."""
    return {"image": np.zeros((360, 640, 3), dtype=np.uint8),
            "lidar": {"points": np.array([[5.0, 0.0, 0.0], [-2.0, 0.0, 0.0]], dtype=np.float32)}}


def test_driving_policy_reads_fov_from_checkpoint(tmp_path):
    ckpt = tmp_path / "driving180.pt"; _save_driving_ckpt_fov(ckpt, 180.0)
    pol = DrivingPolicy(str(ckpt), device="cpu")
    assert pol.fov_deg == 180.0


def test_driving_policy_legacy_checkpoint_has_no_fov(tmp_path):
    # driving_track_v1.pt (Fase 4) nao tem a chave: tem de continuar rodando 360
    ckpt = tmp_path / "driving.pt"; _save_driving_ckpt(ckpt)
    pol = DrivingPolicy(str(ckpt), device="cpu")
    assert pol.fov_deg is None
    vec = pol._lidar_vector(_rear_obs())
    assert vec[36] < 1.0  # o ponto de tras continua visivel


def test_driving_policy_applies_the_checkpoint_fov_to_the_lidar_vector(tmp_path):
    ckpt = tmp_path / "driving180.pt"; _save_driving_ckpt_fov(ckpt, 180.0)
    pol = DrivingPolicy(str(ckpt), device="cpu")
    vec = pol._lidar_vector(_rear_obs())
    assert np.allclose(vec[18:54], 1.0)  # traseira cega
    assert vec[0] < 1.0                  # o ponto da frente sobrevive


def test_driving_policy_explicit_fov_overrides_the_checkpoint(tmp_path):
    ckpt = tmp_path / "driving180.pt"; _save_driving_ckpt_fov(ckpt, 180.0)
    pol = DrivingPolicy(str(ckpt), device="cpu", fov_deg=360.0)
    assert pol.fov_deg == 360.0
    vec = pol._lidar_vector(_rear_obs())
    assert vec[36] < 1.0  # 360 explicito desliga a mascara
