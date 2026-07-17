"""Behavioral-cloning data collection (Fase 1).

Drives the ego with CARLA's Traffic Manager autopilot (a smooth, well-centered
driver) and records, on every synchronous tick, the camera frame + LiDAR sectors
+ the control the autopilot is applying (``vehicle.get_control()``). Frames where
the ego is genuinely out of lane are dropped.

The TM autopilot was chosen over the client-side BasicAgent after measuring that
it stays ~9x tighter to the lane centre on straights and ~2x tighter through
turns (BasicAgent swerved and clipped curbs). Recovery / noise-injection data
(see ``ai.noise``) will be added later as a separate mode.

Usage (from the repo root):
    python src/ai/collect.py --out D:/tcc_data/dataset_v1 --episodes 10 --seconds 120

Report an existing dataset without driving:
    python src/ai/collect.py --report D:/tcc_data/dataset_v1
"""
import os
import sys

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import argparse
import logging
import random
import time

import carla
import numpy as np

from ai.dataset_writer import EpisodeWriter, write_meta, LABEL_COLUMNS
from ai.report import dataset_report, print_report
from ai.sim_lidar import points_to_sectors_m
from ai.eval_closedloop import (
    _launch_server,
    _terminate_server,
    _speed_ms,
    lane_reference,
    read_observation,
)
from core.carlaClient.world_manager import (
    connect_to_carla,
    simulation_context,
    spawn_actor_vehicle,
    spawn_random_vehicles,
    setup_spectator_follow_vehicle,
    update_spectator_position,
)
from utils.config import load_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("collect")

PIPELINE_VERSION = 1
LIDAR_MAX_RANGE_M = 12.0
LIDAR_N_SECTORS = 72


def _episode_dir(out_dir, index):
    return os.path.join(out_dir, "ep_%04d" % index)


def _lidar_points(obs):
    data = obs.get("lidar")
    if data and "points" in data and data["points"] is not None:
        return data["points"]
    return np.zeros((0, 3), dtype=np.float32)


def _teleport_offcenter(world, vehicle, rng, max_lat, max_yaw, min_speed=2.0):
    """Nudge the ego off the lane centre (lateral + heading) on straight road.

    Only fires while the ego is MOVING (speed >= ``min_speed`` m/s): a stopped
    car — e.g. braking behind traffic — must stay put, otherwise the nudge
    corrupts the braking example and fights the stop. The autopilot then steers
    back to centre, so the frames right after become recovery examples. Returns
    True if a teleport happened (skipped when stopped or inside junctions).
    """
    if _speed_ms(vehicle) < min_speed:
        return False
    wp = world.get_map().get_waypoint(
        vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp is None or wp.is_junction:
        return False
    tf = wp.transform
    right = tf.get_right_vector()
    # Signed magnitude with a floor, so every nudge creates a meaningful off-centre
    # state (avoids wasted near-zero teleports).
    lat = rng.choice((-1.0, 1.0)) * rng.uniform(0.35, max_lat)
    yaw = rng.choice((-1.0, 1.0)) * rng.uniform(6.0, max_yaw)
    loc = carla.Location(
        x=tf.location.x + right.x * lat,
        y=tf.location.y + right.y * lat,
        z=vehicle.get_location().z,  # keep current height (avoid ground clipping)
    )
    rot = carla.Rotation(pitch=tf.rotation.pitch, yaw=tf.rotation.yaw + yaw, roll=tf.rotation.roll)
    vehicle.set_transform(carla.Transform(loc, rot))
    return True


def collect(settings_path, out_dir, episodes, seconds, slow_pct=0.0,
            realtime=False, follow=True, launch=True, quality="Low", traffic=0,
            recovery=False, recovery_every=5.0, recovery_lat=0.8, recovery_yaw=18.0, seed=0):
    """Run the collection loop (TM autopilot expert) and write a dataset to ``out_dir``.

    With ``recovery=True``, the ego is periodically nudged off-centre so the
    autopilot's return-to-centre is recorded (fixes the covariate-shift crash).
    """
    settings = load_settings(settings_path)
    carla_config = settings.get("carla_client", {})
    world_config = settings.get("world", {})
    actor_cfg = world_config.get("actor_vehicle", {})
    tm_port = world_config.get("traffic_manager", {}).get("port", 8000)

    os.makedirs(out_dir, exist_ok=True)
    write_meta(out_dir, {
        "pipeline_version": PIPELINE_VERSION,
        "stage": "A",
        "expert": "tm_autopilot",
        "map": world_config.get("map_name"),
        "camera": actor_cfg.get("camera", {}),
        "lidar_sectors": {"n_sectors": LIDAR_N_SECTORS, "max_range_m": LIDAR_MAX_RANGE_M,
                          "z_min": -1.7, "z_max": 2.0, "min_range": 0.5},
        "recovery": ({"every_s": recovery_every, "lat_m": recovery_lat, "yaw_deg": recovery_yaw}
                     if recovery else None),
        "traffic": traffic,
        "label_columns": LABEL_COLUMNS,
    })

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

            traffic_manager = client.get_trafficmanager(tm_port)
            traffic_manager.set_synchronous_mode(True)
            if slow_pct:
                traffic_manager.global_percentage_speed_difference(slow_pct)

            vehicle, sensors = spawn_actor_vehicle(
                world, actor_list, actor_cfg, traffic_manager, ignore_traffic_lights=True)
            for _ in range(10):
                world.tick()

            if traffic:
                spawned = spawn_random_vehicles(world, actor_list, traffic, traffic_manager)
                for _ in range(10):
                    world.tick()  # deixa o tráfego assentar e o autopilot assumir
                logger.info("Traffic: %d vehicles spawned", len(spawned))

            spectator_state = (
                setup_spectator_follow_vehicle(world, vehicle, mode="behind") if follow else {}
            )
            steps_per_episode = int(seconds / fixed_delta)
            recovery_interval = max(1, int(recovery_every / fixed_delta))
            recover_window = int(1.5 / fixed_delta)  # ticks after a nudge tagged as recovery
            rng = random.Random(seed)

            for ep in range(1, episodes + 1):
                writer = EpisodeWriter(_episode_dir(out_dir, ep))
                kept = dropped = recovered = 0
                last_teleport = -10 ** 9
                logger.info("=== Episode %d/%d -> %s (%d steps) ===",
                            ep, episodes, _episode_dir(out_dir, ep), steps_per_episode)
                try:
                    for step in range(steps_per_episode):
                        tick_start = time.perf_counter()

                        if recovery and (step - last_teleport) >= recovery_interval:
                            if _teleport_offcenter(world, vehicle, rng, recovery_lat, recovery_yaw):
                                last_teleport = step

                        world.tick()
                        if spectator_state:
                            update_spectator_position(world, spectator_state)

                        obs = read_observation(vehicle, sensors)
                        control = vehicle.get_control()
                        deviation, lane_width, _ = lane_reference(world.get_map(), vehicle)
                        # Drop only fully-out frames; keep off-centre (recovery) ones.
                        out_of_lane = deviation > lane_width
                        recovering = recovery and (step - last_teleport) < recover_window

                        if obs["image"] is not None and not out_of_lane:
                            tf = vehicle.get_transform()
                            lidar_m = points_to_sectors_m(
                                _lidar_points(obs), n_sectors=LIDAR_N_SECTORS, max_range=LIDAR_MAX_RANGE_M)
                            writer.add(obs["image"], lidar_m, {
                                "steer": control.steer, "throttle": control.throttle,
                                "brake": control.brake, "v": _speed_ms(vehicle),
                                "x": tf.location.x, "y": tf.location.y, "yaw": tf.rotation.yaw,
                                "noise_active": recovering,
                            })
                            kept += 1
                            recovered += 1 if recovering else 0
                        else:
                            dropped += 1

                        if realtime:
                            remaining = fixed_delta - (time.perf_counter() - tick_start)
                            if remaining > 0:
                                time.sleep(remaining)
                finally:
                    writer.close()
                logger.info("Episode %d done: kept %d frames (%d recovery), dropped %d",
                            ep, kept, recovered, dropped)
    finally:
        _terminate_server(server)

    logger.info("Collection finished. Report:")
    print_report(dataset_report(out_dir))


def main():
    parser = argparse.ArgumentParser(description="Behavioral-cloning data collection (TM autopilot)")
    parser.add_argument("--report", metavar="DATASET_DIR",
                        help="Print a report for an existing dataset and exit")
    parser.add_argument("--settings", default="settings/baseSettings.json")
    parser.add_argument("--out", default="data/dataset_v1", help="Dataset output dir")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--slow", type=float, default=0.0,
                        help="TM speed reduction percent (e.g. 30 = 30%% slower than the limit)")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--no-follow", action="store_true")
    parser.add_argument("--no-launch", action="store_true")
    parser.add_argument("--quality", default="Low")
    parser.add_argument("--traffic", type=int, default=0,
                        help="Spawn N autopilot vehicles to create braking events")
    parser.add_argument("--recovery", action="store_true",
                        help="Periodically nudge the ego off-centre and record the autopilot recovering")
    parser.add_argument("--recovery-every", type=float, default=5.0,
                        help="Seconds between nudges (only while the ego is moving)")
    parser.add_argument("--recovery-lat", type=float, default=0.8, help="Max lateral nudge (m)")
    parser.add_argument("--recovery-yaw", type=float, default=18.0, help="Max heading nudge (deg)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.report:
        print_report(dataset_report(args.report))
        return

    collect(
        settings_path=args.settings, out_dir=args.out, episodes=args.episodes,
        seconds=args.seconds, slow_pct=args.slow, realtime=args.realtime,
        follow=not args.no_follow, launch=not args.no_launch, quality=args.quality,
        traffic=args.traffic,
        recovery=args.recovery, recovery_every=args.recovery_every,
        recovery_lat=args.recovery_lat, recovery_yaw=args.recovery_yaw, seed=args.seed,
    )


if __name__ == "__main__":
    main()
