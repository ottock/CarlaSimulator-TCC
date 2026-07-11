"""Warm-start the DrivingNet camera arm from a CameraSteeringNet checkpoint (Fase 3).

The two nets share the exact same ``cnn.*`` submodule, so transferring the proven
PilotNet features is a filtered state_dict copy. The LiDAR arm and head keep their
fresh initialization.
"""
import torch


def load_camera_backbone(model, ckpt_path, device="cpu"):
    """Copy ``cnn.*`` tensors from a camera checkpoint into ``model`` in place.

    Args:
        model: an initialized ``DrivingNet`` (run one forward first so lazy layers exist).
        ckpt_path: path to a checkpoint dict containing ``model_state_dict``.
        device: map_location for loading.

    Returns:
        Number of tensors copied.
    """
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    src = state["model_state_dict"]
    cnn = {k: v for k, v in src.items() if k.startswith("cnn.")}
    missing, unexpected = model.load_state_dict(cnn, strict=False)
    # strict=False: 'missing' lists the lidar/fc/head keys we intentionally skip.
    return len(cnn)
