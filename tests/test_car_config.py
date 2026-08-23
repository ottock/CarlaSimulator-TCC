"""Config do modelo lida no carro (Fase 6b).

A engine TensorRT NAO carrega os metadata_props do ONNX -- o fov_deg se perderia
no trtexec. O sidecar JSON e lido no Jetson com a stdlib, sem exigir o pacote onnx
instalado la.
"""
import io
import json

import pytest

from ai.car.config import car_max_range, load_model_config


def _write(path, data):
    with io.open(str(path), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(data))
    return str(path)


def _valid():
    return {"arch": "DrivingNet", "fov_deg": 180.0, "n_sectors": 72,
            "max_range_m": 12.0, "opset": 11}


def test_reads_the_training_fov(tmp_path):
    cfg = load_model_config(_write(tmp_path / "m.json", _valid()))
    assert cfg["fov_deg"] == 180.0
    assert cfg["n_sectors"] == 72
    assert cfg["max_range_m"] == 12.0


def test_types_are_numbers_not_strings(tmp_path):
    # os metadata_props do ONNX sao strings; o sidecar tem de entregar numeros,
    # senao um "180.0" viraria comparacao de texto la na frente
    cfg = load_model_config(_write(tmp_path / "m.json", _valid()))
    assert isinstance(cfg["fov_deg"], float)
    assert isinstance(cfg["n_sectors"], int)


def test_missing_file_fails_loudly(tmp_path):
    # um default silencioso aqui = mascara de FOV errada = a condicao da ablacao
    with pytest.raises(IOError):
        load_model_config(str(tmp_path / "nao_existe.json"))


def test_missing_key_fails_loudly(tmp_path):
    incomplete = _valid()
    del incomplete["fov_deg"]
    with pytest.raises(KeyError):
        load_model_config(_write(tmp_path / "m.json", incomplete))


def test_car_max_range_is_the_training_range_over_the_scale(tmp_path):
    # A normalizacao divide por max_range, entao usar 12.0/12 = 1.0 no carro e
    # matematicamente identico a multiplicar as leituras reais por 12.
    cfg = load_model_config(_write(tmp_path / "m.json", _valid()))
    assert car_max_range(cfg, scale=12.0) == pytest.approx(1.0)


def test_car_max_range_follows_a_different_training_range(tmp_path):
    cfg = _valid()
    cfg["max_range_m"] = 24.0
    loaded = load_model_config(_write(tmp_path / "m.json", cfg))
    assert car_max_range(loaded, scale=12.0) == pytest.approx(2.0)
