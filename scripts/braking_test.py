"""Teste de frenagem (ablacao LiDAR): carro PARADO a frente do ego numa reta.

Para cada trial: teleporta o ego para o centro de uma faixa reta, coloca um
veiculo parado `--gap` metros a frente (na mesma faixa, fisica desligada =
obstaculo estatico) e dirige o ego com o modelo. Mede se o ego PARA antes de
bater (dist minima ao obstaculo) ou COLIDE.

Rode duas vezes para a ablacao:
    ... braking_test.py --model D:/tcc_data/runs/driving_v3.pt              (com LiDAR)
    ... braking_test.py --model D:/tcc_data/runs/driving_v3.pt --ablate-lidar   (sem LiDAR)

Uso (do diretorio raiz do repo):
    .venv\\Scripts\\python.exe scripts\\braking_test.py --model D:/tcc_data/runs/driving_v3.pt --trials 8 --gap 18
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
    read_observation, _speed_ms,
)
from core.carlaClient.world_manager import (
    connect_to_carla, simulation_context, spawn_actor_vehicle,
    setup_spectator_follow_vehicle, update_spectator_position,
)
from utils.config import load_settings

REPO = os.path.dirname(_SRC)


def main():
    ap = argparse.ArgumentParser(description="Teste de frenagem atras de veiculo parado (ablacao LiDAR)")
    ap.add_argument("--model", default="D:/tcc_data/runs/driving_v3.pt")
    ap.add_argument("--settings", default=os.path.join(REPO, "settings/baseSettings.json"))
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--gap", type=float, default=18.0, help="Distancia (m) do obstaculo a frente")
    ap.add_argument("--seconds", type=float, default=12.0, help="Tempo max por trial")
    ap.add_argument("--ablate-lidar", action="store_true", help="Zera/neutraliza o LiDAR (ablacao)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fast", action="store_true", help="Sem realtime")
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
            pol = DrivingPolicy(a.model, ablate_lidar=a.ablate_lidar)
            cmap = world.get_map()
            spectator = setup_spectator_follow_vehicle(world, v, mode="behind")
            lead_bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]

            pts = cmap.get_spawn_points()
            random.Random(a.seed).shuffle(pts)

            print("\n=== FRENAGEM  modelo=%s  LiDAR=%s  gap=%.0fm ===" % (
                a.model, "ABLADO" if a.ablate_lidar else "ON", a.gap))
            print(" #   dist_min(m)  v_final  resultado")
            stopped = collided = trials = 0
            for sp in pts:
                if trials >= a.trials:
                    break
                wp = cmap.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
                if wp is None or wp.is_junction:
                    continue
                ahead = wp.next(a.gap)
                if not ahead or ahead[0].is_junction:
                    continue  # queremos uma reta livre, sem cruzamento no caminho
                lead_tf = ahead[0].transform
                lead_tf.location.z += 0.3
                # posiciona o ego no centro da faixa
                v.set_transform(wp.transform)
                v.set_target_velocity(carla.Vector3D(0, 0, 0))
                for _ in range(10):
                    world.tick()
                lead = world.try_spawn_actor(lead_bp, lead_tf)
                if lead is None:
                    continue  # ponto ocupado/invalido
                lead.set_simulate_physics(False)  # obstaculo estatico
                trials += 1
                col_base = len(col)
                dist_min = a.gap
                crashed = False
                for step in range(int(a.seconds / fixed)):
                    t0 = time.perf_counter()
                    obs = read_observation(v, sensors)
                    steer, thr, brk = pol(obs)
                    v.apply_control(carla.VehicleControl(steer=steer, throttle=thr, brake=brk))
                    world.tick()
                    update_spectator_position(world, spectator)
                    d = v.get_location().distance(lead.get_location())
                    dist_min = min(dist_min, d)
                    if len(col) > col_base:
                        crashed = True
                        break
                    if _speed_ms(v) < 0.3 and d < a.gap:
                        break  # parou antes do obstaculo
                    if not a.fast:
                        rem = fixed - (time.perf_counter() - t0)
                        if rem > 0:
                            time.sleep(rem)
                v_final = _speed_ms(v)
                lead.destroy()
                if crashed:
                    collided += 1
                    res = "COLIDIU"
                else:
                    stopped += 1
                    res = "parou OK"
                print(" %-3d %-12.2f %-8.2f %s" % (trials - 1, dist_min, v_final, res))
            print("--------------------------------------------")
            print("RESULTADO (LiDAR %s): parou sem bater %d/%d  |  colidiu %d/%d" % (
                "ABLADO" if a.ablate_lidar else "ON", stopped, trials, collided, trials))
    finally:
        _terminate_server(srv)


if __name__ == "__main__":
    main()
