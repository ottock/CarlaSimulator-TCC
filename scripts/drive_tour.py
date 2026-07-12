"""Tour do modelo por vários pontos de Town01, para observar retas/curvas em
partes diferentes do mapa e separar falhas de CRUZAMENTO das de pista.

Teleporta o ego por N pontos de spawn espalhados, dirige `--seconds` em cada,
segue com a câmera (realtime, dá para assistir) e imprime um resumo por local:
distância, desvio médio FORA de cruzamento, e — se bater — o tempo e se a
colisão foi dentro de um cruzamento (junction) ou não.

Uso (do diretório raiz do repo):
    .venv\\Scripts\\python.exe scripts\\drive_tour.py --model D:/tcc_data/runs/driving_v2conv.pt --spawns 8 --seconds 20
Acrescente --fast para rodar sem realtime (mais rápido, sem assistir).
"""
import os
import sys
import time
import argparse
import random

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import carla

from ai.model_policy import DrivingPolicy
from ai.eval_closedloop import (
    _launch_server, _terminate_server, _attach_collision_sensor,
    lane_reference, read_observation, _speed_ms,
)
from core.carlaClient.world_manager import (
    connect_to_carla, simulation_context, spawn_actor_vehicle, spawn_random_vehicles,
    setup_spectator_follow_vehicle, update_spectator_position,
)
from utils.config import load_settings

REPO = os.path.dirname(_SRC)


def main():
    ap = argparse.ArgumentParser(description="Tour do modelo por pontos diferentes de Town01")
    ap.add_argument("--model", default="D:/tcc_data/runs/driving_v2conv.pt")
    ap.add_argument("--settings", default=os.path.join(REPO, "settings/baseSettings.json"))
    ap.add_argument("--spawns", type=int, default=8, help="Quantos pontos de spawn visitar")
    ap.add_argument("--seconds", type=float, default=20.0, help="Segundos dirigindo em cada ponto")
    ap.add_argument("--traffic", type=int, default=0, help="Spawna N carros no transito (autopilot)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fast", action="store_true", help="Sem realtime (não dá para assistir, roda rápido)")
    a = ap.parse_args()

    s = load_settings(a.settings)
    cc, wc, ac = s["carla_client"], s["world"], s["world"]["actor_vehicle"]
    fixed = 0.05

    srv = _launch_server(cc, quality="Low")
    try:
        time.sleep(cc.get("server_boot_time", 30))
        client = connect_to_carla(cc)
        with simulation_context(client, wc) as (world, actors):
            v, sensors = spawn_actor_vehicle(world, actors, ac, traffic_manager=None)
            for _ in range(10):
                world.tick()
            col = _attach_collision_sensor(world, v, actors)
            pol = DrivingPolicy(a.model)
            cmap = world.get_map()
            spectator = setup_spectator_follow_vehicle(world, v, mode="behind")

            if a.traffic:
                tm = client.get_trafficmanager(wc.get("traffic_manager", {}).get("port", 8000))
                tm.set_synchronous_mode(True)
                spawn_random_vehicles(world, actors, a.traffic, tm)
                for _ in range(10):
                    world.tick()

            points = cmap.get_spawn_points()
            random.Random(a.seed).shuffle(points)
            points = points[:a.spawns]
            steps = int(a.seconds / fixed)

            print("\n=== TOUR: %d pontos, %.0fs cada, modelo=%s ===" % (len(points), a.seconds, a.model))
            print(" #   dist(m)  dev_reta(m)  resultado")
            for i, tf in enumerate(points):
                # teleporta e zera a velocidade
                v.set_transform(tf)
                v.set_target_velocity(carla.Vector3D(0, 0, 0))
                for _ in range(15):
                    world.tick()
                col_base = len(col)
                prev = v.get_location()
                dist = 0.0
                road_devs = []
                crash_t = None
                crash_junc = None
                for step in range(steps):
                    t0 = time.perf_counter()
                    obs = read_observation(v, sensors)
                    steer, thr, brk = pol(obs)
                    v.apply_control(carla.VehicleControl(steer=steer, throttle=thr, brake=brk))
                    world.tick()
                    update_spectator_position(world, spectator)
                    loc = v.get_location()
                    dist += loc.distance(prev)
                    prev = loc
                    dev, lw, junc = lane_reference(cmap, v)
                    if not junc:
                        road_devs.append(dev)
                    if crash_t is None and len(col) > col_base:
                        crash_t, crash_junc = step * fixed, junc
                        break
                    if not a.fast:
                        rem = fixed - (time.perf_counter() - t0)
                        if rem > 0:
                            time.sleep(rem)
                dev_reta = (sum(road_devs) / len(road_devs)) if road_devs else 0.0
                if crash_t is None:
                    res = "OK (sem colisao em %.0fs)" % a.seconds
                else:
                    onde = "CRUZAMENTO" if crash_junc else "PISTA (reta/curva)"
                    res = "bateu @ %.1fs em %s" % (crash_t, onde)
                print(" %-3d %-8.1f %-12.2f %s" % (i, dist, dev_reta, res))
    finally:
        _terminate_server(srv)


if __name__ == "__main__":
    main()
