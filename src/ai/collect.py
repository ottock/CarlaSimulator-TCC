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
import time

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


def collect(settings_path, out_dir, episodes, seconds, slow_pct=0.0,
            realtime=False, follow=True, launch=True, quality="Low"):
    """Run the collection loop (TM autopilot expert) and write a dataset to ``out_dir``."""
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

            spectator_state = (
                setup_spectator_follow_vehicle(world, vehicle, mode="behind") if follow else {}
            )
            steps_per_episode = int(seconds / fixed_delta)

            for ep in range(1, episodes + 1):
                writer = EpisodeWriter(_episode_dir(out_dir, ep))
                kept = dropped = 0
                logger.info("=== Episode %d/%d -> %s (%d steps) ===",
                            ep, episodes, _episode_dir(out_dir, ep), steps_per_episode)
                try:
                    for _ in range(steps_per_episode):
                        tick_start = time.perf_counter()

                        world.tick()
                        if spectator_state:
                            update_spectator_position(world, spectator_state)

                        obs = read_observation(vehicle, sensors)
                        control = vehicle.get_control()
                        deviation, lane_width, is_junction = lane_reference(world.get_map(), vehicle)
                        departed = (not is_junction) and (deviation > lane_width / 2.0)

                        if obs["image"] is not None and not departed:
                            tf = vehicle.get_transform()
                            lidar_m = points_to_sectors_m(
                                _lidar_points(obs), n_sectors=LIDAR_N_SECTORS, max_range=LIDAR_MAX_RANGE_M)
                            writer.add(obs["image"], lidar_m, {
                                "steer": control.steer, "throttle": control.throttle,
                                "brake": control.brake, "v": _speed_ms(vehicle),
                                "x": tf.location.x, "y": tf.location.y, "yaw": tf.rotation.yaw,
                                "noise_active": False,
                            })
                            kept += 1
                        else:
                            dropped += 1

                        if realtime:
                            remaining = fixed_delta - (time.perf_counter() - tick_start)
                            if remaining > 0:
                                time.sleep(remaining)
                finally:
                    writer.close()
                logger.info("Episode %d done: kept %d frames, dropped %d (out-of-lane)",
                            ep, kept, dropped)
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
    args = parser.parse_args()

    if args.report:
        print_report(dataset_report(args.report))
        return

    collect(
        settings_path=args.settings, out_dir=args.out, episodes=args.episodes,
        seconds=args.seconds, slow_pct=args.slow, realtime=args.realtime,
        follow=not args.no_follow, launch=not args.no_launch, quality=args.quality,
    )


if __name__ == "__main__":
    main()
