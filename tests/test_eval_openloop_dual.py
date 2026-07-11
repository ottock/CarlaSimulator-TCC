"""Open-loop dual: roda o DrivingNet sobre um dataset sintético e devolve MAE por eixo."""
import numpy as np
import torch

from ai.dataset_writer import EpisodeWriter
from ai.model import DrivingNet
from ai.eval_openloop import evaluate_dual


def _episode(path, n):
    w = EpisodeWriter(str(path))
    for i in range(n):
        w.add(np.zeros((360, 640, 3), dtype=np.uint8),
              np.full(72, float(i % 12), dtype=np.float32),
              {"steer": 0.1, "throttle": 0.5, "brake": 0.0, "v": 1.0,
               "x": 0, "y": 0, "yaw": 0, "noise_active": False})
    w.close()


def test_evaluate_dual_returns_axis_maes(tmp_path):
    _episode(tmp_path / "ep_0001", 6)
    _episode(tmp_path / "ep_0002", 6)
    net = DrivingNet(); net(torch.zeros(1, 3, 66, 200), torch.zeros(1, 72))
    ckpt = tmp_path / "driving.pt"
    torch.save({"model_state_dict": net.state_dict(), "arch": "DrivingNet"}, ckpt)
    out = evaluate_dual(str(ckpt), str(tmp_path), split="all", workers=0, device="cpu")
    assert {"mae_steer", "mae_throttle", "mae_brake", "var_ratio", "frames"} <= set(out)
    assert out["frames"] == 12
