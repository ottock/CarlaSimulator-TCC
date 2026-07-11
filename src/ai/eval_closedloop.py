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


def _attach_collision_sensor(world, vehicle, actor_list):
    """Attach a collision sensor to the ego. Returns a list that fills with events."""
    bp = world.get_blueprint_library().find("sensor.other.collision")
    sensor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
    actor_list.append(sensor)
    events = []
    sensor.listen(lambda e: events.append(e))
    return events


def run_closed_loop(settings_path, routes, seconds, target_speed_kmh,
                    realtime=False, follow=True, launch=True, quality="Low", seed=0,
                    autopilot=False, model_ckpt=None, model_throttle=0.35):
    """Drive the ego in closed loop and report per-route metrics.

    Driver precedence: ``autopilot=True`` uses CARLA's Traffic Manager (smooth
    reference); else ``model_ckpt`` drives with the trained model (Fase 2); else
    the ``ExpertPolicy`` drives. Model and expert share the same policy slot.
    """
    settings = load_settings(settings_path)
    carla_config = settings.get("carla_client", {})
    world_config = settings.get("world", {})
    actor_cfg = world_config.get("actor_vehicle", {})
    tm_port = world_config.get("traffic_manager", {}).get("port", 8000)

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

            if autopilot:
                # Reference driver: CARLA's Traffic Manager controls the ego.
                traffic_manager = client.get_trafficmanager(tm_port)
                traffic_manager.set_synchronous_mode(True)
                vehicle, sensors = spawn_actor_vehicle(
                    world, actor_list, actor_cfg, traffic_manager, ignore_traffic_lights=True)
                policy = None
            else:
                # Injectable policy drives; ego spawned WITHOUT autopilot.
                vehicle, sensors = spawn_actor_vehicle(world, actor_list, actor_cfg, traffic_manager=None)

            # Let the spawn settle and the first sensor frames arrive.
            for _ in range(10):
                world.tick()

            collision_events = _attach_collision_sensor(world, vehicle, actor_list)

            if not autopilot:
                if model_ckpt:
                    from ai.model_policy import ModelSteeringPolicy
                    policy = ModelSteeringPolicy(model_ckpt, throttle=model_throttle)
                    logger.info("Driving with trained model: %s", model_ckpt)
                else:
                    policy = ExpertPolicy(world, vehicle, target_speed_kmh=target_speed_kmh, seed=seed)
            spectator_state = (
                setup_spectator_follow_vehicle(world, vehicle, mode="behind") if follow else {}
            )

            steps_per_route = int(seconds / fixed_delta)
            all_summaries = []
            for route in range(1, routes + 1):
                metrics = RouteMetrics()
                col_start = len(collision_events)
                first_col_step = None
                logger.info("=== Route %d/%d (%d steps) ===", route, routes, steps_per_route)
                for step in range(steps_per_route):
                    tick_start = time.perf_counter()

                    if autopilot:
                        world.tick()
                        steer = vehicle.get_control().steer
                    else:
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
                    metrics.add(deviation, _speed_ms(vehicle), departed, steer=steer)

                    if first_col_step is None and len(collision_events) > col_start:
                        first_col_step = step

                    if realtime:
                        remaining = fixed_delta - (time.perf_counter() - tick_start)
                        if remaining > 0:
                            time.sleep(remaining)

                summary = metrics.summary()
                summary["collisions"] = len(collision_events) - col_start
                summary["first_col_s"] = first_col_step * fixed_delta if first_col_step is not None else -1.0
                all_summaries.append(summary)
                logger.info(
                    "Route %d done: mean_dev=%.2fm p95=%.2fm max=%.2fm offlane=%d collisions=%d "
                    "mean_speed=%.1fm/s steer_std=%.3f%s",
                    route, summary["mean_dev"], summary["p95_dev"], summary["max_dev"],
                    summary["offlane"], summary["collisions"], summary["mean_speed"],
                    summary["steer_std"],
                    ("  <-- FIRST COLLISION @ %.1fs" % summary["first_col_s"]) if summary["collisions"] else "",
                )

            _print_verdict(all_summaries)
            return all_summaries
    finally:
        _terminate_server(server)


def _print_verdict(summaries):
    """Print a pass/fail verdict: a clean route has no lane departures AND no collisions."""
    clean = sum(1 for s in summaries if s["offlane"] == 0 and s.get("collisions", 0) == 0)
    total_col = sum(s.get("collisions", 0) for s in summaries)
    worst = max((s["max_dev"] for s in summaries), default=0.0)
    logger.info("---------------------------------------------")
    logger.info("SUMMARY: %d/%d clean routes (no off-lane, no collision); worst max deviation %.2f m; "
                "total collision events %d", clean, len(summaries), worst, total_col)
    if summaries and clean == len(summaries):
        logger.info("CRITERION MET: drove every route with no lane departures and NO collisions.")
    else:
        logger.info("CRITERION NOT MET: %d route(s) had a lane departure or collision (see above).",
                    len(summaries) - clean)


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
    parser.add_argument("--autopilot", action="store_true",
                        help="Use CARLA's Traffic Manager (smooth reference driver) instead of the policy")
    parser.add_argument("--model", default=None,
                        help="Checkpoint .pt to drive with (the trained model) instead of the expert")
    parser.add_argument("--model-throttle", type=float, default=0.35,
                        help="Fixed throttle when driving with the camera-only model")
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
        autopilot=args.autopilot,
        model_ckpt=args.model,
        model_throttle=args.model_throttle,
    )


if __name__ == "__main__":
    main()
