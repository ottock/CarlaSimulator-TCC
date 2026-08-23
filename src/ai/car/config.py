"""Read the model's sidecar config on the car (Fase 6b).

A TensorRT engine does NOT carry the ONNX ``metadata_props`` -- ``fov_deg`` would be
lost in ``trtexec``. ``export_onnx.py`` writes a small JSON next to the ``.onnx``,
which the Jetson reads with the stdlib, no ``onnx`` package required there.

Fails loudly on a missing file or key: a silent default here means the wrong FOV
mask, which is exactly the LiDAR-ablation condition (0/3 tracks).

Python 3.6-safe.
"""
import io
import json
import os

REQUIRED_KEYS = ("arch", "fov_deg", "n_sectors", "max_range_m")


def load_model_config(path):
    """Load the sidecar JSON, with numeric types and every required key present."""
    if not os.path.exists(path):
        raise IOError("model config not found: {0}".format(path))
    with io.open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    for key in REQUIRED_KEYS:
        if key not in raw:
            raise KeyError("model config {0} is missing '{1}'".format(path, key))
    cfg = dict(raw)
    cfg["fov_deg"] = float(raw["fov_deg"])
    cfg["n_sectors"] = int(raw["n_sectors"])
    cfg["max_range_m"] = float(raw["max_range_m"])
    return cfg


def car_max_range(model_cfg, scale=12.0):
    """The car's ``max_range``: the training range divided by the 1:12 scale.

    Normalisation divides by ``max_range``, so feeding the pipeline 12.0/12 = 1.0 m
    on the car is *mathematically identical* to multiplying the real readings by 12.
    Derived rather than hardcoded, so a model retrained with a different range
    still works without anyone remembering to edit a constant.
    """
    return float(model_cfg["max_range_m"]) / float(scale)
