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

from ai.car.control_map import dead_end, front_blocked
from ai.eval_criterio import corrida_limpa, motivo
from ai.metrics import RouteMetrics
from ai.model_policy import DrivingPolicy
from ai.pistas_grade import expande_pistas
from ai.steer_scale import sim_physical_max_steer_deg
from ai.track_ref import (track_centerline, track_width, deviation_from_centerline,
                          ponto_de_largada)
from ai.collect_track import RECUO_LARGADA_M
from ai.eval_closedloop import (
    _launch_server, _terminate_server, _attach_collision_sensor, read_observation, _speed_ms,
)
from core.carlaClient.track_builder import build_track, pista_fechada
from core.carlaClient.world_manager import (
    connect_to_carla, simulation_context, spawn_actor_vehicle,
    setup_spectator_follow_vehicle, update_spectator_position,
)
from utils.config import load_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("eval_track")


def run_track_eval(settings_path, model_ckpt, pistas, seconds=120.0, obstacles=0,
                   ablate_lidar=False, realtime=False, follow=True, launch=True,
                   quality="Low", throttle_fixo=0.30, beco=0.5, parada=0.06,
                   fim_max_m=8.0, velocidade_alvo=None):
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
                # Pista com FIM (2026-10-09): o carro tem de parar sozinho, como no
                # carro real -- velocidade constante, e as MESMAS duas regras do
                # runtime (control_map), sobre o MESMO vetor que o modelo recebe:
                # parada de emergencia no cone frontal e o beco, que trava. Os
                # limiares sao normalizados pelo max_range, entao valem igual no
                # sim (12 m) e no carro (1 m).
                fechado = pista_fechada(pista, fator)
                s_eixo = [0.0]
                for i in range(1, len(centerline)):
                    s_eixo.append(s_eixo[-1] + math.hypot(
                        centerline[i][0] - centerline[i - 1][0],
                        centerline[i][1] - centerline[i - 1][1]))

                x0, y0, yaw0 = (centerline[0] if fechado
                                else ponto_de_largada(centerline, RECUO_LARGADA_M))
                spawn_tf = carla.Transform(carla.Location(x0, y0, z_spawn),
                                           carla.Rotation(yaw=math.degrees(yaw0)))
                # O mesmo limite de esterco do treino. Avaliar com 70 graus um
                # modelo treinado com 30 mediria um carro que nao existe.
                ego, sensors = spawn_actor_vehicle(
                    world, pista_actors, actor_cfg, spawn_transform=spawn_tf,
                    max_steer_deg=sim_physical_max_steer_deg(float(
                        track_cfg0.get("professor", {}).get("max_steer_deg", 70.0))))
                for _ in range(10):
                    world.tick()
                collisions = _attach_collision_sensor(world, ego, pista_actors)
                policy = DrivingPolicy(model_ckpt, ablate_lidar=ablate_lidar)
                spec = setup_spectator_follow_vehicle(world, ego, mode="behind") if follow else {}

                # O fov_deg vem do checkpoint (Fase 6a). Logar deixa registrado na
                # corrida com que campo de visao o modelo foi treinado -- um
                # descasamento treino x inferencia e' silencioso e catastrofico.
                fov_txt = "360 (sem mascara)" if policy.fov_deg is None else ("%.0f" % policy.fov_deg)
                logger.info("=== PISTA %s%s (%d passos, LiDAR FOV %s) ===",
                            pista, " [LiDAR ABLADO]" if ablate_lidar else "", steps, fov_txt)
                metrics = RouteMetrics()
                percorrido, ant = 0.0, None
                idx = 0
                first_col = None
                fim = False
                for step in range(steps):
                    t0 = time.perf_counter()
                    obs = read_observation(ego, sensors)
                    steer, throttle, brake = policy(obs)
                    if velocidade_alvo is not None:
                        # O mesmo P de velocidade do expert (professor.PurePursuit):
                        # avaliar no ritmo em que o modelo foi treinado. Com o
                        # acelerador fixo em 0,30 o Tesla chegava a 3-5 m/s, contra
                        # 1,74 do treino (2026-10-10), e a avaliacao media o atraso
                        # do esterco, nao o modelo.
                        err = float(velocidade_alvo) - _speed_ms(ego)
                        throttle = min(1.0, 0.5 * err) if err >= 0 else 0.0
                        brake = min(1.0, -0.5 * err) if err < -0.5 else 0.0
                    elif throttle_fixo is not None:
                        # No carro o acelerador e PWM CONSTANTE e a cabeca de
                        # throttle do modelo e ignorada (o longitudinal ficou
                        # fora do escopo da Fase 6). Avaliar com o throttle da
                        # rede mede um carro que nao existe -- e com o dataset
                        # novo, que descarta os quadros de arrancada, ela
                        # aprendeu so o acelerador de cruzeiro (0,125) e nem
                        # sai do lugar.
                        throttle, brake = float(throttle_fixo), 0.0
                    if not fechado:
                        vec = policy._lidar_vector(obs)
                        if beco > 0 and not fim and dead_end(vec, beco):
                            fim = True
                        if fim or front_blocked(vec, parada):
                            throttle, brake = 0.0, 1.0
                    ego.apply_control(carla.VehicleControl(steer=steer, throttle=throttle, brake=brake))
                    world.tick()
                    if spec:
                        update_spectator_position(world, spec)
                    loc = ego.get_transform().location
                    if ant is not None:
                        percorrido += math.hypot(loc.x - ant[0], loc.y - ant[1])
                    ant = (loc.x, loc.y)
                    dev, idx = deviation_from_centerline(centerline, loc.x, loc.y,
                                                         start_idx=idx, window=60)
                    departed = dev > half_w
                    metrics.add(dev, _speed_ms(ego), departed, steer=steer)
                    if first_col is None and len(collisions) > 0:
                        first_col = step
                    if fim and _speed_ms(ego) < 0.05:
                        break                    # parou no fim: corrida encerrada
                    if realtime:
                        rem = fixed - (time.perf_counter() - t0)
                        if rem > 0:
                            time.sleep(rem)

                s = metrics.summary()
                s["pista"] = pista
                s["collisions"] = len(collisions)
                s["first_col_s"] = first_col * fixed if first_col is not None else -1.0
                s["distance_m"] = percorrido
                if not fechado:
                    # "Chegou" = parou pela regra do beco ja na ultima peca, nao
                    # no meio da pista por um falso positivo.
                    s["restante_m"] = s_eixo[-1] - s_eixo[idx]
                    s["chegou_ao_fim"] = bool(fim and s["restante_m"] <= fim_max_m)
                summaries.append(s)
                logger.info("PISTA %s: mean_dev=%.2fm p95=%.2fm max=%.2fm offlane=%d collisions=%d "
                            "mean_speed=%.1fm/s%s%s", pista, s["mean_dev"], s["p95_dev"], s["max_dev"],
                            s["offlane"], s["collisions"], s["mean_speed"],
                            ("  <-- 1a COLISAO @ %.1fs" % s["first_col_s"]) if s["collisions"] else "",
                            "" if fechado else ("  | parou a %.1f m do fim%s" % (
                                s["restante_m"], "" if s["chegou_ao_fim"] else " -- NAO CHEGOU")))
                if not corrida_limpa(s):
                    logger.warning("PISTA %s REPROVOU: %s (andou %.1f m)",
                                   pista, motivo(s), s["distance_m"])

                for a in reversed(pista_actors):
                    try:
                        a.destroy()
                    except Exception:
                        pass
    finally:
        _terminate_server(server)

    logger.info("---------------------------------------------")
    # O criterio exige que o carro tenha ANDADO: um carro parado nao bate e nao
    # sai da pista, e a versao anterior disto aprovou uma corrida com
    # mean_speed=0.0 como "1/1 limpa".
    clean = sum(1 for s in summaries if corrida_limpa(s))
    tag = " (LiDAR ABLADO)" if ablate_lidar else ""
    logger.info("RESUMO%s: %d/%d pistas limpas (dirigiu, sem sair da pista, sem colisao)",
                tag, clean, len(summaries))
    for s in summaries:
        if not corrida_limpa(s):
            logger.info("   %s: %s", s["pista"], motivo(s))
    return summaries


def main():
    p = argparse.ArgumentParser(description="Loop fechado na pista custom (Fase 4)")
    p.add_argument("--settings", default="settings/pistaTCC.json")
    p.add_argument("--model", required=True)
    p.add_argument("--pistas", default="pista1,pista2,pista3",
                   help="presets, codigos grade:SDSES ou grupos grade:teste etc. "
                        "(ver ai/pistas_grade.py). grade:teste so bate com a coleta "
                        "se --frac-teste/--semente-split forem os mesmos.")
    p.add_argument("--frac-teste", type=float, default=0.15)
    p.add_argument("--semente-split", type=int, default=0)
    p.add_argument("--curva-final", action="store_true")
    p.add_argument("--velocidade-alvo", type=float, default=None, metavar="V",
                   help="segura V m/s com o P de velocidade do expert (2.0 = o ritmo "
                        "da coleta), no lugar do acelerador fixo")
    p.add_argument("--beco", type=float, default=0.5,
                   help="limiar normalizado da regra do beco nas pistas com fim "
                        "(0.5 = 50 cm no carro). 0 desliga.")
    p.add_argument("--seconds", type=float, default=120.0)
    p.add_argument("--obstacles", type=int, default=0, help="Espalha N obstaculos na pista")
    p.add_argument("--ablate-lidar", action="store_true", help="Neutraliza o LiDAR (ablacao)")
    p.add_argument("--realtime", action="store_true")
    p.add_argument("--no-follow", action="store_true")
    p.add_argument("--no-launch", action="store_true")
    p.add_argument("--quality", default="Low")
    p.add_argument("--throttle-fixo", type=float, default=0.30, metavar="T",
                   help="acelerador constante, como o PWM fixo do carro. "
                        "Passe -1 para usar a cabeca de throttle do modelo.")
    a = p.parse_args()

    pistas, _ = expande_pistas(a.pistas, frac_teste=a.frac_teste,
                               semente=a.semente_split, curva_final=a.curva_final)
    run_track_eval(
        settings_path=a.settings, model_ckpt=a.model, pistas=pistas, seconds=a.seconds,
        obstacles=a.obstacles, ablate_lidar=a.ablate_lidar, realtime=a.realtime,
        follow=not a.no_follow, launch=not a.no_launch, quality=a.quality,
        throttle_fixo=(None if a.throttle_fixo < 0 else a.throttle_fixo), beco=a.beco,
        velocidade_alvo=a.velocidade_alvo)


if __name__ == "__main__":
    main()
