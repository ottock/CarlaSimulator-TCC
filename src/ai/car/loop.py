"""The car's control loop, with its hardware injected (Fase 6b).

Every dependency is passed in, so the whole safety envelope is testable on the PC
with fakes: servo centred when data is missing, ESC never leaving neutral. The
Jetson wiring lives in ``hardware/jetson_runtime.py`` and only supplies the real
camera, LiDAR, engine and actuator.

The perception chain reuses ``ai.shared.*`` unchanged -- the same functions the
simulator ran. Reimplementing any of them here would be a silent inference bug.

Python 3.6-safe.
"""
import time

import numpy as np

from ai.car.control_map import ESC_NEUTRAL_US, STEER_CENTER_US, FrameWatchdog, steer_to_us
from ai.car.image_crop import prepare_frame
from ai.car.scan_assembly import ScanAssembler
from ai.shared.image_pipeline import preprocess
from ai.shared.lidar_pipeline import apply_fov_mask, normalize_sectors_m, scan_to_sectors_m


class DriveLoop:
    """One ``step()`` = one control cycle."""

    def __init__(self, camera, lidar, engine, actuator, logger, fov_deg, max_range,
                 crop_frac, n_sectors=72, clock=None, watchdog=None):
        self.camera = camera
        self.lidar = lidar
        self.engine = engine
        self.actuator = actuator
        self.logger = logger
        self.fov_deg = fov_deg
        self.max_range = max_range
        self.crop_frac = crop_frac
        self.n_sectors = n_sectors
        self.clock = clock or time.monotonic
        self.watchdog = watchdog or FrameWatchdog()
        self.assembler = ScanAssembler()
        self._last_vec = None
        self._t_prev = None

    def _sectors_from_scan(self, scan):
        angles = [a for a, _ in scan]
        dists = [d for _, d in scan]
        sectors_m = scan_to_sectors_m(angles, dists, n_sectors=self.n_sectors,
                                      max_range=self.max_range)
        sectors_m = apply_fov_mask(sectors_m, self.fov_deg, self.max_range)
        return normalize_sectors_m(sectors_m, self.max_range)

    def step(self):
        """Read sensors, run the model, command the servo, log. Returns telemetry."""
        now = self.clock()
        stalled = self.watchdog.tick(now)
        dt = 0.0 if self._t_prev is None else (now - self._t_prev)
        self._t_prev = now

        for scan in self.assembler.feed(self.lidar.read_points()):
            self._last_vec = self._sectors_from_scan(scan)
            self.logger.log_scan(t=now, points=scan)

        frame = self.camera.read()
        control = (0.0, 0.0, 0.0)
        # Sem uma volta completa o vetor teria buracos que a rede leria como "livre";
        # sem imagem nao ha o que inferir; um frame estourado significa laco travado.
        # Nos tres casos o comando seguro e o mesmo: servo ao centro.
        can_drive = (self._last_vec is not None) and (frame is not None) and (not stalled)
        if can_drive:
            img = preprocess(prepare_frame(frame, self.crop_frac))
            control = self.engine.infer(img, self._last_vec)
        servo_us = steer_to_us(control[0]) if can_drive else STEER_CENTER_US

        self.actuator.set_servo_us(servo_us)
        self.actuator.set_esc_us(ESC_NEUTRAL_US)   # nunca sai de neutro nesta fase

        vec = self._last_vec
        self.logger.log_frame(
            t=now,
            sectors=vec if vec is not None else np.ones(self.n_sectors, dtype=np.float32),
            control=control, servo_us=servo_us, dt=dt, frame_bgr=frame)
        return {"t": now, "steer": control[0], "throttle": control[1],
                "brake": control[2], "servo_us": servo_us, "dt": dt,
                "stalled": stalled, "has_scan": self._last_vec is not None,
                "lidar_vec": vec}
