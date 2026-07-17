"""Tests for the camera-only steering network (Fase 2).

PilotNet-style CNN: image (B,3,66,200) -> steer (B,1) in [-1,1]. Kept small so it
exports cleanly and (later) runs on the Jetson.
"""
import torch

from ai.model import CameraSteeringNet, DrivingNet


def test_forward_output_shape():
    net = CameraSteeringNet()
    out = net(torch.zeros(4, 3, 66, 200))
    assert out.shape == (4, 1)


def test_output_in_tanh_range():
    net = CameraSteeringNet()
    out = net(torch.randn(8, 3, 66, 200))
    assert float(out.min()) >= -1.0
    assert float(out.max()) <= 1.0


def test_model_is_small():
    net = CameraSteeringNet()
    net(torch.zeros(1, 3, 66, 200))  # initialize any lazy layers
    n_params = sum(p.numel() for p in net.parameters())
    assert n_params < 2_000_000  # tiny, Jetson-friendly


def test_dual_forward_output_shape():
    net = DrivingNet()
    out = net(torch.zeros(4, 3, 66, 200), torch.zeros(4, 72))
    assert out.shape == (4, 3)


def test_dual_output_ranges():
    net = DrivingNet()
    out = net(torch.randn(8, 3, 66, 200), torch.rand(8, 72))
    steer, throttle, brake = out[:, 0], out[:, 1], out[:, 2]
    assert float(steer.min()) >= -1.0 and float(steer.max()) <= 1.0
    assert float(throttle.min()) >= 0.0 and float(throttle.max()) <= 1.0
    assert float(brake.min()) >= 0.0 and float(brake.max()) <= 1.0


def test_dual_lidar_changes_output():
    # Prova que o LiDAR está ligado à saída (relevante para a ablation).
    torch.manual_seed(0)
    net = DrivingNet().eval()
    img = torch.randn(2, 3, 66, 200)
    a = net(img, torch.zeros(2, 72))
    b = net(img, torch.ones(2, 72))
    assert not torch.allclose(a, b)


def test_dual_zero_lidar_forward_ok():
    # Caminho da ablation: LiDAR zerado não pode quebrar.
    net = DrivingNet()
    out = net(torch.zeros(1, 3, 66, 200), torch.zeros(1, 72))
    assert out.shape == (1, 3)


def test_dual_camera_backbone_matches_camera_net():
    # Mesmas chaves e shapes de conv que CameraSteeringNet (pré-requisito do warm-start).
    cam = CameraSteeringNet(); cam(torch.zeros(1, 3, 66, 200))
    dual = DrivingNet(); dual(torch.zeros(1, 3, 66, 200), torch.zeros(1, 72))
    cam_cnn = {k: v.shape for k, v in cam.state_dict().items() if k.startswith("cnn.")}
    dual_cnn = {k: v.shape for k, v in dual.state_dict().items() if k.startswith("cnn.")}
    assert cam_cnn == dual_cnn


def test_dual_model_is_small():
    net = DrivingNet()
    net(torch.zeros(1, 3, 66, 200), torch.zeros(1, 72))
    assert sum(p.numel() for p in net.parameters()) < 2_000_000
