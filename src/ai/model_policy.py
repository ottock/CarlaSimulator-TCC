"""Trained steering model wrapped as a closed-loop policy (Fase 2).

Exposes ``policy(obs) -> (steer, throttle, brake)`` so it drops into the same
harness slot the expert used. Camera-only: the model outputs steering; a fixed
throttle keeps the car moving so lane-keeping can be evaluated. Preprocessing is
the SAME shared pipeline used in training (and later on the car).
"""
import numpy as np
import torch

from ai.model import CameraSteeringNet
from ai.shared.image_pipeline import preprocess


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
