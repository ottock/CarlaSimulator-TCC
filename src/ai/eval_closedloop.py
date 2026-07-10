"""Closed-loop evaluation harness (Fase 0 deliverable).

Runs the control loop end-to-end in CARLA: read sensors -> policy -> apply
control -> world.tick() -> measure. The policy is injectable; today it is the
BasicAgent expert, and from Fase 2 the trained model plugs into the SAME point
without changing the loop.

Usage (from the repo root):
    python src/ai/eval_closedloop.py --settings settings/baseSettings.json \
        --routes 3 --seconds 120

Add --realtime to watch it in the spectator window at 1x speed; --no-launch to
attach to an already-running CARLA server.
"""
# --- make `src` importable no matter how this script is launched ---
import os
import sys

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import argparse
import logging
import subprocess
import time
from pathlib import Path

import carla

from ai.expert.basic_agent_expert import ExpertPolicy
from ai.metrics import RouteMetrics
from core.carlaClient.world_manager import (
    connect_to_carla,
    simulation_context,
    spawn_actor_vehicle,
    setup_spectator_follow_vehicle,
    update_spectator_position,
)
from utils.config import load_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("eval_closedloop")


def _launch_server(carla_config, quality="Low"):
    """Launch CarlaUE4.exe (mirrors main.launch_carla_server but with a quality knob)."""
    repo_root = Path(__file__).resolve().parents[2]
    exe_path = repo_root / "CARLA_0.9.16" / "CarlaUE4.exe"
    if not exe_path.exists():
        raise RuntimeError("CARLA executable not found at %s" % exe_path)
    port = carla_config.get("port", 2000)
    cmd = [str(exe_path), "-carla-port=%d" % port, "-quality-level=%s" % quality, "-nosound"]
    logger.info("Launching CARLA server (quality=%s) on port %d", quality, port)
    return subprocess.Popen(cmd, cwd=str(exe_path.parent))


def _terminate_server(server):
    """Kill the CARLA server reliably on Windows.

    CarlaUE4.exe is a stub launcher that spawns CarlaUE4-Win64-Shipping.exe and
    exits, so a plain ``terminate()`` on our PID leaves the real engine holding
    port 2000. Kill the process tree, then best-effort the orphaned shipping
    process by image name.
    """
    if server is None:
        return
    logger.info("Terminating CARLA server (pid %s) and any child engine process", server.pid)
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(server.pid)], capture_output=True)
        subprocess.run(["taskkill", "/F", "/IM", "CarlaUE4-Win64-Shipping.exe"], capture_output=True)
    else:
        server.terminate()


def _speed_ms(vehicle):
    v = vehicle.get_velocity()
    return (v.x * v.x + v.y * v.y + v.z * v.z) ** 0.5


def lane_reference(carla_map, vehicle):
    """Measure the ego against the nearest driving lane.

    Returns ``(deviation_m, lane_width_m, is_junction)``. Deviation is the planar
    distance from the vehicle to the nearest driving-lane centre; the caller
    treats it as a genuine lane departure only when the ego is NOT in a junction
    and the deviation exceeds half the lane width.
    """
    loc = vehicle.get_transform().location
    wp = carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp is None:
        return 0.0, 0.0, False
    c = wp.transform.location
    dx, dy = loc.x - c.x, loc.y - c.y
    deviation = (dx * dx + dy * dy) ** 0.5
    return deviation, float(wp.lane_width), bool(wp.is_junction)


def read_observation(vehicle, sensors):
    """Assemble the observation dict the policy receives (mirrors what the model will get)."""
    obs = {"speed": _speed_ms(vehicle), "image": None, "lidar": None}
    camera = sensors.get("camera")
    if camera is not None:
        frame = camera.get_latest_frame()
        if frame is not None:
            obs["image"] = frame["image"]
    lidar = sensors.get("lidar")
    if lidar is not None:
        obs["lidar"] = lidar.get_latest_point_cloud()
    return obs


def run_closed_loop(settings_path, routes, seconds, target_speed_kmh,
                    realtime=False, follow=True, launch=True, quality="Low", seed=0):
    """Drive the ego with the expert in closed loop and report per-route metrics."""
    settings = load_settings(settings_path)
    carla_config = settings.get("carla_client", {})
    world_config = settings.get("world", {})
    actor_cfg = world_config.get("actor_vehicle", {})

    server = None
    try:
        if launch:
            server = _launch_server(carla_config, quality=quality)
            boot = carla_config.get("server_boot_time", 30)
            logger.info("Waiting %ss for CARLA to boot", boot)
            time.sleep(boot)

        client = connect_to_carla(carla_config)

        with simulation_context(client, world_config) as (world, actor_list):
            fixed_delta = world.get_settings().fixed_delta_seconds or 0.05

            # Spawn ego + sensors WITHOUT autopilot (traffic_manager=None) so the
            # policy is the only thing driving.
            vehicle, sensors = spawn_actor_vehicle(world, actor_list, actor_cfg, traffic_manager=None)

            # Let the spawn settle and the first sensor frames arrive.
            for _ in range(10):
                world.tick()

            policy = ExpertPolicy(world, vehicle, target_speed_kmh=target_speed_kmh, seed=seed)
            spectator_state = (
                setup_spectator_follow_vehicle(world, vehicle, mode="behind") if follow else {}
            )

            steps_per_route = int(seconds / fixed_delta)
            all_summaries = []
            for route in range(1, routes + 1):
                metrics = RouteMetrics()
                logger.info("=== Route %d/%d (%d steps) ===", route, routes, steps_per_route)
                for _ in range(steps_per_route):
                    tick_start = time.perf_counter()

                    obs = read_observation(vehicle, sensors)
                    steer, throttle, brake = policy(obs)
                    vehicle.apply_control(
                        carla.VehicleControl(steer=steer, throttle=throttle, brake=brake)
                    )
                    world.tick()
                    if spectator_state:
                        update_spectator_position(world, spectator_state)

                    deviation, lane_width, is_junction = lane_reference(world.get_map(), vehicle)
                    departed = (not is_junction) and (deviation > lane_width / 2.0)
                    metrics.add(deviation, _speed_ms(vehicle), departed)

                    if realtime:
                        remaining = fixed_delta - (time.perf_counter() - tick_start)
                        if remaining > 0:
                            time.sleep(remaining)

                summary = metrics.summary()
                all_summaries.append(summary)
                logger.info(
                    "Route %d done: mean_dev=%.2fm p95=%.2fm max=%.2fm offlane=%d mean_speed=%.1fm/s",
                    route, summary["mean_dev"], summary["p95_dev"], summary["max_dev"],
                    summary["offlane"], summary["mean_speed"],
                )

            _print_verdict(all_summaries)
            return all_summaries
    finally:
        _terminate_server(server)


def _print_verdict(summaries):
    """Print a Fase-0 pass/fail verdict against the approval criteria."""
    clean = sum(1 for s in summaries if s["offlane"] == 0)
    worst = max((s["max_dev"] for s in summaries), default=0.0)
    logger.info("---------------------------------------------")
    logger.info("SUMMARY: %d/%d routes with zero off-lane steps; worst max deviation %.2f m",
                clean, len(summaries), worst)
    if summaries and clean == len(summaries):
        logger.info("FASE 0 CRITERION MET: expert drove every route without leaving the lane.")
    else:
        logger.info("FASE 0 CRITERION NOT MET: some routes had off-lane steps (see above).")


def main():
    parser = argparse.ArgumentParser(description="CARLA closed-loop evaluation harness")
    parser.add_argument("--settings", default="settings/baseSettings.json", help="Path to settings JSON")
    parser.add_argument("--routes", type=int, default=3, help="Number of routes to drive")
    parser.add_argument("--seconds", type=float, default=120.0, help="Seconds per route")
    parser.add_argument("--target-speed", type=float, default=25.0, help="Expert target speed (km/h)")
    parser.add_argument("--realtime", action="store_true", help="Pace the loop to 1x for watching")
    parser.add_argument("--no-follow", action="store_true", help="Do not move the spectator camera")
    parser.add_argument("--no-launch", action="store_true", help="Attach to an already-running server")
    parser.add_argument("--quality", default="Low", help="CARLA quality level when launching")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for destinations")
    args = parser.parse_args()

    run_closed_loop(
        settings_path=args.settings,
        routes=args.routes,
        seconds=args.seconds,
        target_speed_kmh=args.target_speed,
        realtime=args.realtime,
        follow=not args.no_follow,
        launch=not args.no_launch,
        quality=args.quality,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
