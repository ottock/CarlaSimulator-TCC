"""Coleta de Behavior Cloning na PISTA CUSTOM (Fase 4).

Junta os dois lados do projeto:
  - expert: o Pure Pursuit do colega (`core.carlaClient.professor.PurePursuit`)
    seguindo a linha de centro da pista (`ai.track_ref.track_centerline`);
  - formato: o NOSSO `EpisodeWriter` (JPG + lidar.npy de 72 setores + labels.csv),
    com os NOSSOS sensores (`spawn_actor_vehicle`), pra o `train.py --dual` treinar
    sem nenhuma mudanca.

Para cada pista (pista1/2/3): monta a pista, spawna o ego no 1o waypoint, dirige
com Pure Pursuit gravando o dataset e, opcionalmente, injeta recuperacao (cutuca
o ego pra fora do centro em movimento -> o Pure Pursuit volta). Roda varias pistas
num run so, num unico dataset.

Uso (da raiz do repo):
    python src/ai/collect_track.py --out D:/tcc_data/dataset_track_v1 \
        --pistas pista1,pista2,pista3 --episodes-por-pista 4 --seconds 60 --recovery
"""
import os
import sys

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import argparse
import logging
import math
import random
import time

import carla
import numpy as np

from ai.dataset_writer import EpisodeWriter, write_meta, LABEL_COLUMNS
from ai.recovery_schedule import RecoveryScheduler
from ai.report import dataset_report, print_report
from ai.sim_lidar import points_to_sectors_m
from ai.track_ref import track_centerline, track_width, deviation_from_centerline
from ai.eval_closedloop import _launch_server, _terminate_server, _speed_ms, read_observation
from core.carlaClient.track_builder import build_track
from core.carlaClient.professor import PurePursuit, _wheelbase
from core.carlaClient.world_manager import (
    connect_to_carla, simulation_context, spawn_actor_vehicle,
)
from utils.config import load_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("collect_track")

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


def _teleport_offcenter_track(vehicle, centerline, rng, max_lat, max_yaw, z):
    """Cutuca o ego pra fora do centro, relativo a linha de centro (nao a get_waypoint).

    Acha o waypoint da centerline mais proximo, monta um transform nele e desloca
    pela direita dele + gira o heading. O Pure Pursuit volta ao centro -> os frames
    seguintes viram exemplos de recuperacao. Retorna True se teleportou."""
    loc = vehicle.get_location()
    _, idx = deviation_from_centerline(centerline, loc.x, loc.y)
    wx, wy, wyaw = centerline[idx]
    base = carla.Transform(carla.Location(wx, wy, z), carla.Rotation(yaw=math.degrees(wyaw)))
    right = base.get_right_vector()
    lat = rng.choice((-1.0, 1.0)) * rng.uniform(0.35, max_lat)
    yaw = rng.choice((-1.0, 1.0)) * rng.uniform(6.0, max_yaw)
    new_loc = carla.Location(x=wx + right.x * lat, y=wy + right.y * lat, z=loc.z)
    new_rot = carla.Rotation(yaw=math.degrees(wyaw) + yaw)
    vehicle.set_transform(carla.Transform(new_loc, new_rot))
    return True


def collect_track(settings_path, out_dir, pistas, episodes_por_pista=4, seconds=60.0,
                  launch=True, quality="Low", recovery=False, recovery_every=5.0,
                  recovery_lat=0.8, recovery_yaw=18.0, recovery_min_speed=1.0, seed=0):
    settings = load_settings(settings_path)
    cc = settings.get("carla_client", {})
    wc = settings.get("world", {})
    track_cfg0 = wc.get("track", {})
    actor_cfg = wc.get("actor_vehicle", {})
    prof = track_cfg0.get("professor", {})

    os.makedirs(out_dir, exist_ok=True)
    write_meta(out_dir, {
        "pipeline_version": PIPELINE_VERSION,
        "stage": "B",
        "expert": "pure_pursuit",
        "map": wc.get("map_name"),
        "pistas": pistas,
        "escala": track_cfg0.get("escala"),
        "camera": actor_cfg.get("camera", {}),
        "lidar_sectors": {"n_sectors": LIDAR_N_SECTORS, "max_range_m": LIDAR_MAX_RANGE_M},
        "recovery": ({"every_s": recovery_every, "lat_m": recovery_lat, "yaw_deg": recovery_yaw}
                     if recovery else None),
        "label_columns": LABEL_COLUMNS,
    })

    server = _launch_server(cc, quality=quality) if launch else None
    try:
        if launch:
            boot = cc.get("server_boot_time", 30)
            logger.info("Waiting %ss for CARLA to boot", boot)
            time.sleep(boot)
        client = connect_to_carla(cc)
        with simulation_context(client, wc) as (world, _actor_list):
            fixed = world.get_settings().fixed_delta_seconds or 0.05
            steps_per_ep = int(seconds / fixed)
            sched = RecoveryScheduler(interval_steps=max(1, int(recovery_every / fixed)),
                                      window_steps=int(1.5 / fixed), enabled=recovery)
            rng = random.Random(seed)
            fator = 12.0 if str(track_cfg0.get("escala", "meio")).lower() == "real" else 1.0
            z_spawn = float(track_cfg0.get("z", 0.05)) * fator + 0.3

            global_ep = 0
            for pista in pistas:
                track_cfg = dict(track_cfg0)
                track_cfg["preset"] = pista
                # obstaculos fora na coleta (Pure Pursuit nao desvia deles)
                track_cfg["obstacles"] = {"count": 0}
                pista_actors = []
                logger.info("=== PISTA %s ===", pista)
                build_track(world, track_cfg, pista_actors)
                centerline = track_centerline(track_cfg)
                lim_fora = track_width(track_cfg)   # desvio > largura util = fora da pista

                x0, y0, yaw0 = centerline[0]
                spawn_tf = carla.Transform(carla.Location(x0, y0, z_spawn),
                                           carla.Rotation(yaw=math.degrees(yaw0)))
                ego, sensors = spawn_actor_vehicle(
                    world, pista_actors, actor_cfg, spawn_transform=spawn_tf)
                for _ in range(10):
                    world.tick()

                pp = PurePursuit(
                    centerline, wheelbase=_wheelbase(ego),
                    lookahead=float(prof.get("lookahead", 4.0)),
                    target_speed=float(prof.get("target_speed", 3.0)),
                    k_throttle=float(prof.get("k_throttle", 0.5)),
                    max_steer_deg=float(prof.get("max_steer_deg", 70.0)))

                for _ep in range(episodes_por_pista):
                    writer = EpisodeWriter(_episode_dir(out_dir, global_ep))
                    global_ep += 1
                    kept = dropped = recovered = 0
                    sched.reset()   # o `step` recomeca em 0: o relogio tem que recomecar junto
                    try:
                        for step in range(steps_per_ep):
                            moving = _speed_ms(ego) >= recovery_min_speed
                            if sched.should_teleport(step, moving):
                                _teleport_offcenter_track(ego, centerline, rng,
                                                          recovery_lat, recovery_yaw, z_spawn)
                                sched.mark(step)

                            tf = ego.get_transform()
                            speed = _speed_ms(ego)
                            steer, throttle, brake = pp.control(
                                tf.location.x, tf.location.y, math.radians(tf.rotation.yaw), speed)
                            ego.apply_control(carla.VehicleControl(
                                steer=steer, throttle=throttle, brake=brake))
                            world.tick()

                            obs = read_observation(ego, sensors)
                            dev, _ = deviation_from_centerline(centerline, tf.location.x, tf.location.y)
                            recovering = sched.is_recovering(step)
                            if obs["image"] is not None and dev <= lim_fora:
                                lidar_m = points_to_sectors_m(
                                    _lidar_points(obs), n_sectors=LIDAR_N_SECTORS,
                                    max_range=LIDAR_MAX_RANGE_M)
                                writer.add(obs["image"], lidar_m, {
                                    "steer": steer, "throttle": throttle, "brake": brake,
                                    "v": speed, "x": tf.location.x, "y": tf.location.y,
                                    "yaw": tf.rotation.yaw, "noise_active": recovering})
                                kept += 1
                                recovered += 1 if recovering else 0
                            else:
                                dropped += 1
                    finally:
                        writer.close()
                    logger.info("  %s ep%d: kept %d (%d recovery), dropped %d",
                                pista, global_ep - 1, kept, recovered, dropped)

                for a in reversed(pista_actors):
                    try:
                        a.destroy()
                    except Exception:
                        pass
    finally:
        _terminate_server(server)

    logger.info("Coleta finalizada. Relatorio:")
    print_report(dataset_report(out_dir))


def main():
    p = argparse.ArgumentParser(description="Coleta BC na pista custom (Pure Pursuit -> nosso formato)")
    p.add_argument("--report", metavar="DATASET_DIR", help="So imprime o relatorio e sai")
    p.add_argument("--settings", default="settings/pistaTCC.json")
    p.add_argument("--out", default="D:/tcc_data/dataset_track_v1")
    p.add_argument("--pistas", default="pista1,pista2,pista3",
                   help="Lista de presets separados por virgula")
    p.add_argument("--episodes-por-pista", type=int, default=4)
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--no-launch", action="store_true")
    p.add_argument("--quality", default="Low")
    p.add_argument("--recovery", action="store_true")
    p.add_argument("--recovery-every", type=float, default=5.0)
    p.add_argument("--recovery-lat", type=float, default=0.8)
    p.add_argument("--recovery-yaw", type=float, default=18.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    if a.report:
        print_report(dataset_report(a.report))
        return

    pistas = [s.strip() for s in a.pistas.split(",") if s.strip()]
    collect_track(
        settings_path=a.settings, out_dir=a.out, pistas=pistas,
        episodes_por_pista=a.episodes_por_pista, seconds=a.seconds,
        launch=not a.no_launch, quality=a.quality, recovery=a.recovery,
        recovery_every=a.recovery_every, recovery_lat=a.recovery_lat,
        recovery_yaw=a.recovery_yaw, seed=a.seed)


if __name__ == "__main__":
    main()
