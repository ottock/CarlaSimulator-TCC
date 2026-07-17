"""Trained steering model wrapped as a closed-loop policy (Fase 2).

Exposes ``policy(obs) -> (steer, throttle, brake)`` so it drops into the same
harness slot the expert used. Camera-only: the model outputs steering; a fixed
throttle keeps the car moving so lane-keeping can be evaluated. Preprocessing is
the SAME shared pipeline used in training (and later on the car).
"""
import numpy as np
import torch

from ai.model import CameraSteeringNet, DrivingNet
from ai.shared.image_pipeline import preprocess
from ai.sim_lidar import points_to_sectors_m
from ai.shared.lidar_pipeline import normalize_sectors_m


class ModelSteeringPolicy:
    def __init__(self, ckpt_path, throttle=0.35, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.throttle = throttle
        self.model = CameraSteeringNet().to(self.device)
        self.model(torch.zeros(1, 3, 66, 200, device=self.device))  # init lazy layer
        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()

    def __call__(self, obs):
        img = obs.get("image") if obs else None
        if img is None:
            return 0.0, 0.0, 0.0
        x = preprocess(img)
        with torch.no_grad():
            tensor = torch.from_numpy(x).unsqueeze(0).to(self.device)
            steer = float(self.model(tensor).item())
        return steer, self.throttle, 0.0


class DrivingPolicy:
    """Dual-input trained model as a closed-loop policy (Fase 3).

    Consumes camera + LiDAR from ``obs`` and outputs (steer, throttle, brake).
    LiDAR points are converted with the SAME sim adapter used at collection time,
    then normalized with the SAME function used in training.
    """

    def __init__(self, ckpt_path, device=None, throttle_floor=0.0, brake_deadzone=0.1,
                 n_sectors=72, max_range=12.0, ablate_lidar=False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.throttle_floor = throttle_floor
        self.brake_deadzone = brake_deadzone
        self.n_sectors = n_sectors
        self.max_range = max_range
        self.ablate_lidar = ablate_lidar
        self.model = DrivingNet().to(self.device)
        self.model(torch.zeros(1, 3, 66, 200, device=self.device),
                   torch.zeros(1, n_sectors, device=self.device))  # init lazy
        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()

    def _lidar_vector(self, obs):
        data = obs.get("lidar") if obs else None
        if self.ablate_lidar or not data or data.get("points") is None:
            # normalize_sectors_m uses near=0/free=1: an all-ones vector means
            # "clear road in every direction", the neutral no-LiDAR signal for
            # both the ablation experiment and the missing-cloud fallback.
            # Zeros would instead mean "obstacle in every direction", which is
            # wrong here and inconsistent with how empty clouds are stored at
            # collection time (see ai.sim_lidar / ai.shared.lidar_pipeline).
            return np.ones(self.n_sectors, dtype=np.float32)
        sectors_m = points_to_sectors_m(data["points"], n_sectors=self.n_sectors,
                                        max_range=self.max_range)
        return normalize_sectors_m(sectors_m, self.max_range)

    def __call__(self, obs):
        img = obs.get("image") if obs else None
        if img is None:
            return 0.0, 0.0, 0.0
        x = preprocess(img)
        lidar = self._lidar_vector(obs)
        with torch.no_grad():
            xt = torch.from_numpy(x).unsqueeze(0).to(self.device)
            lt = torch.from_numpy(lidar).unsqueeze(0).to(self.device)
            out = self.model(xt, lt).cpu().numpy().ravel()
        steer, throttle, brake = float(out[0]), float(out[1]), float(out[2])
        # Throttle and brake are mutually exclusive. The brake head never regresses
        # exactly to 0, and even a tiny residual brake applied together with throttle
        # holds the car at a standstill (confirmed in closed loop). Below the deadzone
        # drop the brake (and honour the throttle floor); at/above it brake for real
        # and cut throttle.
        if brake < self.brake_deadzone:
            brake = 0.0
            throttle = max(throttle, self.throttle_floor)
        else:
            throttle = 0.0
        return steer, throttle, brake
