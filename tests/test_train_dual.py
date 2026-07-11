"""Smoke test do treino dual: 2 episódios sintéticos, 1 época, CPU -> gera checkpoint."""
import numpy as np
import torch

from ai.dataset_writer import EpisodeWriter
from ai.train import train_dual


def _episode(path, n):
    w = EpisodeWriter(str(path))
    for i in range(n):
        s = 0.4 if i % 2 else -0.4
        br = 0.8 if i % 3 == 0 else 0.0
        w.add(np.zeros((360, 640, 3), dtype=np.uint8),
              np.full(72, float(i % 12), dtype=np.float32),
              {"steer": s, "throttle": 0.5, "brake": br, "v": 1.0,
               "x": 0, "y": 0, "yaw": 0, "noise_active": False})
    w.close()


def test_train_dual_writes_checkpoint(tmp_path):
    _episode(tmp_path / "ep_0001", 8)
    _episode(tmp_path / "ep_0002", 8)
    out = tmp_path / "driving_smoke.pt"
    train_dual(str(tmp_path), str(out), epochs=1, batch=4, workers=0, device="cpu")
    assert out.exists()
    state = torch.load(str(out), map_location="cpu", weights_only=False)
    assert state["arch"] == "DrivingNet"
    assert {"mae_steer", "mae_throttle", "mae_brake"} <= set(state)
