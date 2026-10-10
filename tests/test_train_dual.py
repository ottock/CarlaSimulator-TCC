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


def test_train_dual_records_fov_in_the_checkpoint(tmp_path):
    # O FOV do treino TEM de viajar no checkpoint: se a inferencia usar outro, a
    # rede recebe uma entrada fora da distribuicao (= a condicao da ablacao).
    _episode(tmp_path / "ep_0001", 8)
    _episode(tmp_path / "ep_0002", 8)
    out = tmp_path / "driving_smoke_180.pt"
    train_dual(str(tmp_path), str(out), epochs=1, batch=4, workers=0, device="cpu", fov_deg=180.0)
    state = torch.load(str(out), map_location="cpu", weights_only=False)
    assert state["fov_deg"] == 180.0


def test_train_dual_records_no_fov_when_full_360(tmp_path):
    _episode(tmp_path / "ep_0001", 8)
    _episode(tmp_path / "ep_0002", 8)
    out = tmp_path / "driving_smoke_360.pt"
    train_dual(str(tmp_path), str(out), epochs=1, batch=4, workers=0, device="cpu")
    state = torch.load(str(out), map_location="cpu", weights_only=False)
    assert state["fov_deg"] is None


def test_train_dual_records_the_photometric_ranges(tmp_path):
    # O checkpoint diz com que faixas foi treinado: "v2" ou "v3" (ai.augment.FAIXAS).
    _episode(tmp_path / "ep_0001", 8)
    _episode(tmp_path / "ep_0002", 8)
    out = tmp_path / "driving_faixas.pt"
    train_dual(str(tmp_path), str(out), epochs=1, batch=4, workers=0, device="cpu",
               photometric=True, faixas_fotometricas="v2")
    state = torch.load(str(out), map_location="cpu", weights_only=False)
    assert state["photometric"] is True
    assert state["photometric_faixas"] == "v2"


def test_unknown_photometric_ranges_are_refused(tmp_path):
    import pytest
    _episode(tmp_path / "ep_0001", 4)
    with pytest.raises(SystemExit):
        train_dual(str(tmp_path), str(tmp_path / "x.pt"), epochs=1, batch=4, workers=0,
                   device="cpu", photometric=True, faixas_fotometricas="v9")


def test_fine_tuning_starts_from_the_whole_trained_net(tmp_path):
    # Rodadas de DAgger (2026-10-10): a rede inteira parte do modelo anterior.
    _episode(tmp_path / "ep_0001", 8)
    _episode(tmp_path / "ep_0002", 8)
    base = tmp_path / "base.pt"
    train_dual(str(tmp_path), str(base), epochs=1, batch=4, workers=0, device="cpu", fov_deg=180.0)
    sd_base = torch.load(base, weights_only=False)["model_state_dict"]
    fino = tmp_path / "fino.pt"
    train_dual(str(tmp_path), str(fino), epochs=1, batch=4, workers=0, device="cpu", fov_deg=180.0,
               lr=0.0, continuar_de=str(base))
    sd_fino = torch.load(fino, weights_only=False)["model_state_dict"]
    assert all(torch.allclose(sd_base[k].float(), sd_fino[k].float()) for k in sd_base)


def test_fine_tuning_refuses_a_different_fov(tmp_path):
    import pytest
    _episode(tmp_path / "ep_0001", 8)
    _episode(tmp_path / "ep_0002", 8)
    base = tmp_path / "base.pt"
    train_dual(str(tmp_path), str(base), epochs=1, batch=4, workers=0, device="cpu", fov_deg=180.0)
    with pytest.raises(SystemExit):
        train_dual(str(tmp_path), str(tmp_path / "x.pt"), epochs=1, batch=4, workers=0,
                   device="cpu", fov_deg=120.0, continuar_de=str(base))
