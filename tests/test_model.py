"""Tests for the camera-only steering network (Fase 2).

PilotNet-style CNN: image (B,3,66,200) -> steer (B,1) in [-1,1]. Kept small so it
exports cleanly and (later) runs on the Jetson.
"""
import torch

from ai.model import CameraSteeringNet


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
