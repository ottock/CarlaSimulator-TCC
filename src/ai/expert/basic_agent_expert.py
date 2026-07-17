"""Stage-A expert: CARLA BasicAgent driving to rotating random destinations.

BasicAgent computes control on the client from the map/waypoints (not from our
camera/LiDAR), which is exactly what we want for a clean reference label and,
later, for noise injection. It exposes the same call signature the model will
use, so the closed-loop harness treats expert and model identically.
"""
import logging

from ai.paths import ensure_carla_agents_on_path

# CARLA's `agents` package is not on the default path; add it before importing.
ensure_carla_agents_on_path()
from agents.navigation.basic_agent import BasicAgent  # noqa: E402

logger = logging.getLogger(__name__)

# BasicAgent's default lateral PID (K_P=1.95, K_I=0.05, K_D=0.2) is under-damped:
# the high proportional gain on the angle to a near waypoint makes it swerve on
# straights and the integral term winds up. A lower-gain, no-integral tune drives
# smoothly, which matters because the model clones the expert's behavior.
DEFAULT_LATERAL_PID = {"K_P": 0.6, "K_I": 0.0, "K_D": 0.15}


class ExpertPolicy:
    """Wrap a BasicAgent as a ``policy(obs) -> (steer, throttle, brake)`` callable.

    The agent drives toward a destination; when it arrives, a new random spawn
    point is chosen so the ego keeps driving indefinitely. ``obs`` is ignored
    (the agent navigates by waypoints), but the harness still reads the sensors
    every step so the full data path is exercised.
    """

    def __init__(self, world, vehicle, target_speed_kmh=25.0, seed=None, lateral_pid=None):
        import random

        self._vehicle = vehicle
        self._map = world.get_map()
        self._spawn_points = self._map.get_spawn_points()
        self._rng = random.Random(seed)

        dt = world.get_settings().fixed_delta_seconds or 0.05
        pid = dict(lateral_pid if lateral_pid is not None else DEFAULT_LATERAL_PID)
        pid["dt"] = dt
        self._agent = BasicAgent(
            vehicle, target_speed=target_speed_kmh,
            opt_dict={"lateral_control_dict": pid},
        )
        self._set_new_destination()

    def _set_new_destination(self):
        destination = self._rng.choice(self._spawn_points).location
        self._agent.set_destination(destination)
        logger.info("Expert destination set to (%.1f, %.1f)", destination.x, destination.y)

    def __call__(self, obs=None):
        """Return ``(steer, throttle, brake)`` for the current vehicle state."""
        if self._agent.done():
            self._set_new_destination()
        control = self._agent.run_step()
        return float(control.steer), float(control.throttle), float(control.brake)
