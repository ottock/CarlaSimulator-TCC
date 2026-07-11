"""Behavioral-cloning networks.

Fase 2: ``CameraSteeringNet`` — a PilotNet/DAVE-2-style CNN that maps a front
camera frame to a steering command. Kept small (~0.3 M params) so it exports to
ONNX and later runs on the Jetson. Fase 3 will add a LiDAR arm + a longitudinal
output on top of the same vision backbone.
"""
import torch.nn as nn


class CameraSteeringNet(nn.Module):
    """Input: img (B, 3, 66, 200) in [-1, 1]. Output: steer (B, 1) via tanh."""

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.cnn = nn.Sequential(                       # DAVE-2 / PilotNet
            nn.Conv2d(3, 24, 5, stride=2), nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2), nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2), nn.ELU(),
            nn.Conv2d(48, 64, 3), nn.ELU(),
            nn.Conv2d(64, 64, 3), nn.ELU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.LazyLinear(100), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(100, 50), nn.ELU(),
            nn.Linear(50, 1), nn.Tanh(),
        )

    def forward(self, img):
        return self.head(self.cnn(img))
