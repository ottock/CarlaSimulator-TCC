"""Testes do export ONNX (Fase 6a).

O Jetson Nano nao roda PyTorch: o modelo vai como ONNX -> TensorRT. Estes testes
prendem o contrato do arquivo exportado (opset, nomes, batch dinamico, metadados)
e, principalmente, a PARIDADE numerica: se o ONNX divergir do checkpoint, o carro
dirige com uma rede diferente da que foi validada em malha fechada.
"""
import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from ai.export_onnx import export_onnx, check_parity, synthetic_samples
from ai.model import DrivingNet


def _ckpt(path, fov_deg=None):
    net = DrivingNet()
    net(torch.zeros(1, 3, 66, 200), torch.zeros(1, 72))
    torch.save({"model_state_dict": net.state_dict(), "arch": "DrivingNet",
                "fov_deg": fov_deg}, str(path))
    return str(path)


def test_export_writes_a_valid_onnx_file(tmp_path):
    out = str(tmp_path / "m.onnx")
    export_onnx(_ckpt(tmp_path / "m.pt"), out, opset=11)
    onnx.checker.check_model(onnx.load(out))  # levanta se o grafo for invalido


def test_export_uses_the_requested_opset(tmp_path):
    # opset 11: validado contra o TensorRT 8.x do JetPack 4.6 (cobre ate 13-14)
    out = str(tmp_path / "m.onnx")
    export_onnx(_ckpt(tmp_path / "m.pt"), out, opset=11)
    model = onnx.load(out)
    default = [o for o in model.opset_import if o.domain in ("", "ai.onnx")]
    assert default and default[0].version == 11


def test_export_names_the_inputs_and_output(tmp_path):
    out = str(tmp_path / "m.onnx")
    export_onnx(_ckpt(tmp_path / "m.pt"), out, opset=11)
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == ["image", "lidar"]
    assert [o.name for o in sess.get_outputs()] == ["control"]


def test_export_accepts_a_batch_larger_than_one(tmp_path):
    # batch dinamico: a paridade roda em lote e o runtime pode variar
    out = str(tmp_path / "m.onnx")
    export_onnx(_ckpt(tmp_path / "m.pt"), out, opset=11)
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    y = sess.run(["control"], {"image": np.zeros((4, 3, 66, 200), dtype=np.float32),
                               "lidar": np.ones((4, 72), dtype=np.float32)})[0]
    assert y.shape == (4, 3)


def test_export_records_the_training_fov_in_the_metadata(tmp_path):
    # a Fase 6b le isto para mascarar o scan real com o MESMO fov do treino,
    # sem depender de alguem lembrar de repetir a flag
    out = str(tmp_path / "m.onnx")
    export_onnx(_ckpt(tmp_path / "m.pt", fov_deg=180.0), out, opset=11)
    meta = {p.key: p.value for p in onnx.load(out).metadata_props}
    assert meta["fov_deg"] == "180.0"
    assert meta["n_sectors"] == "72"
    assert meta["max_range_m"] == "12.0"
    assert meta["arch"] == "DrivingNet"


def test_metadata_reports_360_for_a_checkpoint_without_fov(tmp_path):
    out = str(tmp_path / "m.onnx")
    export_onnx(_ckpt(tmp_path / "m.pt", fov_deg=None), out, opset=11)
    meta = {p.key: p.value for p in onnx.load(out).metadata_props}
    assert meta["fov_deg"] == "360.0"  # sem mascara = visao completa


def test_parity_between_torch_and_onnx_is_within_tolerance(tmp_path):
    ckpt = _ckpt(tmp_path / "m.pt")
    out = str(tmp_path / "m.onnx")
    export_onnx(ckpt, out, opset=11)
    max_diff = check_parity(ckpt, out, synthetic_samples(8, seed=0))
    assert max_diff < 1e-4


def test_parity_catches_a_mismatched_onnx(tmp_path):
    # se a paridade nao reprovasse um ONNX de outros pesos, ela nao provaria nada
    ckpt = _ckpt(tmp_path / "m.pt")
    other = _ckpt(tmp_path / "other.pt")
    out = str(tmp_path / "other.onnx")
    export_onnx(other, out, opset=11)
    assert check_parity(ckpt, out, synthetic_samples(8, seed=0)) > 1e-4


def test_synthetic_samples_have_the_model_input_shapes(tmp_path):
    imgs, lidars = synthetic_samples(5, seed=0)
    assert imgs.shape == (5, 3, 66, 200) and imgs.dtype == np.float32
    assert lidars.shape == (5, 72) and lidars.dtype == np.float32
    assert lidars.min() >= 0.0 and lidars.max() <= 1.0   # LiDAR normalizado
    assert imgs.min() >= -1.0 and imgs.max() <= 1.0      # imagem em [-1, 1]


def test_export_writes_a_json_sidecar_next_to_the_onnx(tmp_path):
    # a engine TensorRT nao carrega os metadata_props; o Jetson le este JSON
    import json
    out = str(tmp_path / "m.onnx")
    export_onnx(_ckpt(tmp_path / "m.pt", fov_deg=180.0), out, opset=11)
    with open(str(tmp_path / "m.json")) as fh:
        side = json.load(fh)
    assert side["fov_deg"] == 180.0
    assert side["n_sectors"] == 72
    assert side["max_range_m"] == 12.0
    assert side["arch"] == "DrivingNet"


def test_sidecar_holds_numbers_not_strings(tmp_path):
    import json
    out = str(tmp_path / "m.onnx")
    export_onnx(_ckpt(tmp_path / "m.pt", fov_deg=180.0), out, opset=11)
    with open(str(tmp_path / "m.json")) as fh:
        side = json.load(fh)
    assert isinstance(side["fov_deg"], float)
    assert isinstance(side["n_sectors"], int)
