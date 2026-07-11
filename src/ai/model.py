"""Behavioral-cloning networks.

Fase 2: ``CameraSteeringNet`` — a PilotNet/DAVE-2-style CNN that maps a front
camera frame to a steering command. Kept small (~0.3 M params) so it exports to
ONNX and later runs on the Jetson. Fase 3 will add a LiDAR arm + a longitudinal
output on top of the same vision backbone.
"""
import torch
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


class DrivingNet(nn.Module):
    """Dual-input BC net (Fase 3).

    Camera arm = the SAME PilotNet CNN as CameraSteeringNet (so its weights can be
    warm-started from cam_v2.pt). LiDAR arm = small Conv1D with circular padding
    (sector 71 neighbours sector 0). Outputs [steer, throttle, brake].

    forward(img (B,3,66,200) in [-1,1], lidar (B,n_sectors) in [0,1]) -> (B,3).
    """

    def __init__(self, n_sectors: int = 72, dropout: float = 0.3):
        super().__init__()
        self.cnn = nn.Sequential(                       # IDÊNTICO à CameraSteeringNet
            nn.Conv2d(3, 24, 5, stride=2), nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2), nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2), nn.ELU(),
            nn.Conv2d(48, 64, 3), nn.ELU(),
            nn.Conv2d(64, 64, 3), nn.ELU(),
            nn.Flatten(),
        )
        self.lidar = nn.Sequential(                     # padding circular = setores dão a volta
            nn.Conv1d(1, 16, 5, padding=2, padding_mode="circular"), nn.ELU(),
            nn.Conv1d(16, 32, 3, padding=1, padding_mode="circular"), nn.ELU(),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.LazyLinear(100), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(100, 50), nn.ELU(),
        )
        self.steer_head = nn.Sequential(nn.Linear(50, 1), nn.Tanh())
        self.throttle_head = nn.Sequential(nn.Linear(50, 1), nn.Sigmoid())
        self.brake_head = nn.Sequential(nn.Linear(50, 1), nn.Sigmoid())

    def forward(self, img, lidar):
        cam = self.cnn(img)                             # (B, Fcam)
        lid = self.lidar(lidar.unsqueeze(1))            # (B, Flid)
        h = self.fc(torch.cat([cam, lid], dim=1))       # (B, 50)
        return torch.cat(
            [self.steer_head(h), self.throttle_head(h), self.brake_head(h)], dim=1
        )                                               # (B, 3)
