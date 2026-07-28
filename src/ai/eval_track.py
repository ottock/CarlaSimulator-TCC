"""Loop fechado na PISTA CUSTOM (Fase 4) — valida o modelo dirigindo as pistas.

Mesmo espirito do `eval_closedloop`, mas na pista custom: como ela nao tem malha
OpenDRIVE, o desvio e' medido contra a NOSSA linha de centro (`ai.track_ref`) em vez
do `map.get_waypoint()`. Monta a pista, spawna o ego no 1o waypoint, dirige com o
modelo (`DrivingPolicy`) e mede desvio + colisoes por pista.

Ablacao do LiDAR: rode com e sem `--ablate-lidar`. Numa pista COM paredes o LiDAR
enxerga o corredor, entao aqui a diferenca com/sem LiDAR deve finalmente aparecer
(ver §5.7 do docs/ESTADO_IA.md).

Uso (da raiz do repo):
    python src/ai/eval_track.py --model D:/tcc_data/runs/driving_track_v1.pt \
        --pistas pista1,pista2,pista3 --seconds 120 --realtime
    python src/ai/eval_track.py --model ... --ablate-lidar     # a ablacao
"""
import os
import sys

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import argparse
import logging
import math
import time

import carla

from ai.metrics import RouteMetrics
from ai.model_policy import DrivingPolicy
from ai.track_ref import track_centerline, track_width, deviation_from_centerline
from ai.eval_closedloop import (
    _launch_server, _terminate_server, _attach_collision_sensor, read_observation, _speed_ms,
)
from core.carlaClient.track_builder import build_track
from core.carlaClient.world_manager import (
    connect_to_carla, simulation_context, spawn_actor_vehicle,
    setup_spectator_follow_vehicle, update_spectator_position,
)
from utils.config import load_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("eval_track")


def run_track_eval(settings_path, model_ckpt, pistas, seconds=120.0, obstacles=0,
                   ablate_lidar=False, realtime=False, follow=True, launch=True,
                   quality="Low", model_throttle=0.35):
    settings = load_settings(settings_path)
    cc = settings.get("carla_client", {})
    wc = settings.get("world", {})
    track_cfg0 = wc.get("track", {})
    actor_cfg = wc.get("actor_vehicle", {})
    fator = 12.0 if str(track_cfg0.get("escala", "meio")).lower() == "real" else 1.0
    z_spawn = float(track_cfg0.get("z", 0.05)) * fator + 0.3

    server = _launch_server(cc, quality=quality) if launch else None
    summaries = []
    try:
        if launch:
            boot = cc.get("server_boot_time", 30)
            logger.info("Waiting %ss for CARLA to boot", boot)
            time.sleep(boot)
        client = connect_to_carla(cc)
        with simulation_context(client, wc) as (world, _actor_list):
            fixed = world.get_settings().fixed_delta_seconds or 0.05
            steps = int(seconds / fixed)

            for pista in pistas:
                track_cfg = dict(track_cfg0)
                track_cfg["preset"] = pista
                track_cfg["obstacles"] = {"count": int(obstacles),
                                          "types": track_cfg0.get("obstacles", {}).get(
                                              "types", ["tcc_cone", "tcc_mureta", "tcc_pessoa2d"])}
                pista_actors = []
                build_track(world, track_cfg, pista_actors)
                centerline = track_centerline(track_cfg)
                half_w = track_width(track_cfg) / 2.0

                x0, y0, yaw0 = centerline[0]
                spawn_tf = carla.Transform(carla.Location(x0, y0, z_spawn),
                                           carla.Rotation(yaw=math.degrees(yaw0)))
                ego, sensors = spawn_actor_vehicle(
                    world, pista_actors, actor_cfg, spawn_transform=spawn_tf)
                for _ in range(10):
                    world.tick()
                collisions = _attach_collision_sensor(world, ego, pista_actors)
                policy = DrivingPolicy(model_ckpt, ablate_lidar=ablate_lidar)
                spec = setup_spectator_follow_vehicle(world, ego, mode="behind") if follow else {}

                logger.info("=== PISTA %s%s (%d passos) ===",
                            pista, " [LiDAR ABLADO]" if ablate_lidar else "", steps)
                metrics = RouteMetrics()
                idx = 0
                first_col = None
                for step in range(steps):
                    t0 = time.perf_counter()
                    obs = read_observation(ego, sensors)
                    steer, throttle, brake = policy(obs)
                    ego.apply_control(carla.VehicleControl(steer=steer, throttle=throttle, brake=brake))
                    world.tick()
                    if spec:
                        update_spectator_position(world, spec)
                    loc = ego.get_transform().location
                    dev, idx = deviation_from_centerline(centerline, loc.x, loc.y,
                                                         start_idx=idx, window=60)
                    departed = dev > half_w
                    metrics.add(dev, _speed_ms(ego), departed, steer=steer)
                    if first_col is None and len(collisions) > 0:
                        first_col = step
                    if realtime:
                        rem = fixed - (time.perf_counter() - t0)
                        if rem > 0:
                            time.sleep(rem)

                s = metrics.summary()
                s["pista"] = pista
                s["collisions"] = len(collisions)
                s["first_col_s"] = first_col * fixed if first_col is not None else -1.0
                summaries.append(s)
                logger.info("PISTA %s: mean_dev=%.2fm p95=%.2fm max=%.2fm offlane=%d collisions=%d "
                            "mean_speed=%.1fm/s%s", pista, s["mean_dev"], s["p95_dev"], s["max_dev"],
                            s["offlane"], s["collisions"], s["mean_speed"],
                            ("  <-- 1a COLISAO @ %.1fs" % s["first_col_s"]) if s["collisions"] else "")

                for a in reversed(pista_actors):
                    try:
                        a.destroy()
                    except Exception:
                        pass
    finally:
        _terminate_server(server)

    logger.info("---------------------------------------------")
    clean = sum(1 for s in summaries if s["offlane"] == 0 and s["collisions"] == 0)
    tag = " (LiDAR ABLADO)" if ablate_lidar else ""
    logger.info("RESUMO%s: %d/%d pistas limpas (sem sair da pista, sem colisao)",
                tag, clean, len(summaries))
    return summaries


def main():
    p = argparse.ArgumentParser(description="Loop fechado na pista custom (Fase 4)")
    p.add_argument("--settings", default="settings/pistaTCC.json")
    p.add_argument("--model", required=True)
    p.add_argument("--pistas", default="pista1,pista2,pista3")
    p.add_argument("--seconds", type=float, default=120.0)
    p.add_argument("--obstacles", type=int, default=0, help="Espalha N obstaculos na pista")
    p.add_argument("--ablate-lidar", action="store_true", help="Neutraliza o LiDAR (ablacao)")
    p.add_argument("--realtime", action="store_true")
    p.add_argument("--no-follow", action="store_true")
    p.add_argument("--no-launch", action="store_true")
    p.add_argument("--quality", default="Low")
    a = p.parse_args()

    pistas = [s.strip() for s in a.pistas.split(",") if s.strip()]
    run_track_eval(
        settings_path=a.settings, model_ckpt=a.model, pistas=pistas, seconds=a.seconds,
        obstacles=a.obstacles, ablate_lidar=a.ablate_lidar, realtime=a.realtime,
        follow=not a.no_follow, launch=not a.no_launch, quality=a.quality)


if __name__ == "__main__":
    main()
