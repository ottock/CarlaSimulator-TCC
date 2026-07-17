"""Warm-start: copiar o backbone de câmera do checkpoint da Fase 2 para o DrivingNet."""
import torch

from ai.model import CameraSteeringNet, DrivingNet
from ai.warmstart import load_camera_backbone


def _save_camera_ckpt(path):
    cam = CameraSteeringNet(); cam(torch.zeros(1, 3, 66, 200))
    torch.save({"model_state_dict": cam.state_dict(), "arch": "CameraSteeringNet"}, path)
    return cam


def test_load_camera_backbone_copies_conv_weights(tmp_path):
    ckpt = tmp_path / "cam.pt"
    cam = _save_camera_ckpt(ckpt)
    dual = DrivingNet(); dual(torch.zeros(1, 3, 66, 200), torch.zeros(1, 72))
    n = load_camera_backbone(dual, str(ckpt), device="cpu")
    assert n > 0
    for k, v in cam.state_dict().items():
        if k.startswith("cnn."):
            assert torch.allclose(dual.state_dict()[k], v)


def test_load_camera_backbone_leaves_lidar_untouched(tmp_path):
    ckpt = tmp_path / "cam.pt"
    _save_camera_ckpt(ckpt)
    dual = DrivingNet(); dual(torch.zeros(1, 3, 66, 200), torch.zeros(1, 72))
    before = {k: v.clone() for k, v in dual.state_dict().items() if k.startswith("lidar.")}
    load_camera_backbone(dual, str(ckpt), device="cpu")
    for k, v in before.items():
        assert torch.allclose(dual.state_dict()[k], v)  # braço de LiDAR intacto
