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
from ai.noise import SteeringNoiseInjector
from ai.recovery_schedule import RecoveryScheduler
from ai.stuck import JanelaDeColisao, StuckDetector
from ai.report import dataset_report, print_report
from ai.sim_lidar import points_to_sectors_m
from ai.racing_line import (MODOS, expert_path, lateral_offset, min_radius_m,
                            nudge_bounds)
from ai.steer_scale import (CAR_LENGTH_M, CAR_MIN_RADIUS_M, CAR_WIDTH_M, SCALE,
                            sim_physical_max_steer_deg)
from ai.track_ref import (track_centerline, track_width, deviation_from_centerline,
                          ponto_de_largada)
from ai.pistas_grade import expande_pistas
from ai.eval_closedloop import _launch_server, _terminate_server, _speed_ms, read_observation
from ai.eval_closedloop import _attach_collision_sensor
from core.carlaClient.track_builder import build_track, pista_fechada
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
# Pista com fim: o carro nasce com o centro a 3 m (sim) da borda de entrada --
# meio Tesla (2,35 m) mais folga, ou ~25 cm no carro real. Ver ponto_de_largada.
RECUO_LARGADA_M = 3.0


def _episode_dir(out_dir, index):
    return os.path.join(out_dir, "ep_%04d" % index)


def _lidar_points(obs):
    data = obs.get("lidar")
    if data and "points" in data and data["points"] is not None:
        return data["points"]
    return np.zeros((0, 3), dtype=np.float32)


def _trajeto_do_expert(modo, centerline, track_cfg, margem, fechado=True):
    """Caminho que o Pure Pursuit vai perseguir, ja conferido contra o carro.

    O eixo da pista exige raio de 3,18 m e o carro faz 5,64 (0,470 medidos x12).
    Seguir o eixo sempre foi impossivel -- e daí veio o subesterco em tudo. A
    conferencia aqui recusa coletar de novo sobre um caminho inexecutavel.
    """
    half = track_width(track_cfg) / 2.0
    caminho = expert_path(centerline, half_width=half,
                          vehicle_width=CAR_WIDTH_M * SCALE,
                          vehicle_length=CAR_LENGTH_M * SCALE,
                          margin=margem,
                          min_radius_required=CAR_MIN_RADIUS_M * SCALE,
                          modo=modo, fechado=fechado)
    logger.info("Traçado '%s': raio minimo %.2f m (carro faz %.2f) | %d pontos",
                modo, min_radius_m([(x, y) for x, y, _ in caminho], fechado=fechado),
                CAR_MIN_RADIUS_M * SCALE, len(caminho))
    return caminho


def _destravar(vehicle, trajeto):
    """Recoloca o ego no traçado, parado, depois de encravar.

    Sem isto um unico encrave mata o resto da coleta: o veiculo nao e recriado
    entre episodios. Zera a velocidade de proposito -- reposicionar mantendo o
    vetor antigo e justamente o que encrava.
    """
    loc = vehicle.get_location()
    i = min(range(len(trajeto)),
            key=lambda k: (trajeto[k][0] - loc.x) ** 2 + (trajeto[k][1] - loc.y) ** 2)
    px, py, pyaw = trajeto[i]
    vehicle.set_transform(carla.Transform(
        carla.Location(px, py, loc.z + 0.2), carla.Rotation(yaw=math.degrees(pyaw))))
    try:
        vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    except AttributeError:
        pass
    return True


def pista_e_episodios(item, padrao):
    """``"oval_tcc*8"`` -> ``("oval_tcc", 8)``; sem ``*`` vale o padrao.

    Para dar a uma pista mais episodios que as outras na MESMA coleta -- o oval do
    MVP nao pode ficar sub-representado no meio de 90 pistas curtas.
    """
    nome, _, n = str(item).partition("*")
    nome = nome.strip()
    if not nome:
        raise ValueError("pista vazia em %r" % (item,))
    if not n:
        return nome, int(padrao)
    if int(n) < 1:
        raise ValueError("numero de episodios tem de ser >= 1 em %r" % (item,))
    return nome, int(n)


def _em_que_bateu(eventos):
    """Resumo dos atores tocados num episodio -- '' se nenhum.

    Sem isto o log so dizia "1070 toques", e o primeiro palpite (parede) estava
    errado: era a borda da laje da primeira peca.
    """
    if not eventos:
        return ""
    cont = {}
    for e in eventos:
        try:
            nome = e.other_actor.type_id
        except Exception:
            nome = "?"
        cont[nome] = cont.get(nome, 0) + 1
    top = sorted(cont.items(), key=lambda kv: -kv[1])[:3]
    return "  [tocou: %s]" % ", ".join("%s x%d" % kv for kv in top)


def _volta_ao_inicio(world, vehicle, largada, z):
    """Recoloca o ego parado na largada de uma pista com fim, para o proximo episodio."""
    x0, y0, yaw0 = largada
    vehicle.set_transform(carla.Transform(carla.Location(x0, y0, z),
                                          carla.Rotation(yaw=math.degrees(yaw0))))
    try:
        vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    except AttributeError:
        pass
    vehicle.apply_control(carla.VehicleControl(hand_brake=True))
    for _ in range(20):          # assenta antes de dirigir (ver eval_closedloop)
        world.tick()
    vehicle.apply_control(carla.VehicleControl(hand_brake=False))


def _empurrao(vehicle, centerline, half_width, veh_width, rng, max_yaw, max_lat):
    """Desloca o ego lateralmente A PARTIR DE ONDE ELE ESTA, dentro do corredor.

    Duas versoes anteriores erraram o mesmo ponto por lados opostos, e as duas
    travaram o carro:

      - a partir do EIXO: o carro dirige o traçado, a 1,93 m dali, entao ele era
        arrancado 1,93 m de lado antes de somar o deslocamento;
      - a partir do TRACADO: no inicio do episodio o carro ainda esta no eixo,
        entao um "empurrao de 0,6" virou um salto medido de 1,4 m que deixou a
        carroceria 25 cm DENTRO da parede.

    Deslocar a partir da posicao atual torna o empurrao exatamente o que diz
    ser. O rumo e preservado (so uma guinada pequena por cima), entao a
    velocidade continua alinhada com o carro e nao ha derrapagem instantanea.
    """
    tf = vehicle.get_transform()
    loc = tf.location
    atual, normal = lateral_offset(loc.x, loc.y, centerline)
    lo, hi = nudge_bounds(atual, half_width, veh_width)
    # 20 cm de margem das paredes -- o limite exato deixa a carroceria encostada --
    # e um teto de magnitude: o corredor sozinho permitiria metros, e um carro de
    # 4,7 m atirado longe numa curva fechada encosta de qualquer jeito.
    lo = max(lo + 0.20, -float(max_lat))
    hi = min(hi - 0.20, float(max_lat))
    if hi <= lo:
        return False
    lat = rng.uniform(lo, hi)
    yaw = rng.choice((-1.0, 1.0)) * rng.uniform(2.0, max_yaw) if max_yaw > 2.0 else 0.0
    vehicle.set_transform(carla.Transform(
        carla.Location(x=loc.x + float(normal[0]) * lat,
                       y=loc.y + float(normal[1]) * lat, z=loc.z),
        carla.Rotation(pitch=tf.rotation.pitch, yaw=tf.rotation.yaw + yaw,
                       roll=tf.rotation.roll)))
    return True


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
                  recovery_lat=0.7, recovery_yaw=6.0, recovery_min_speed=1.0, seed=0,
                  tracado="estadio", margem=0.0, recovery_amp=0.2,
                  empurrar=True, fim_m=4.5, pistas_teste=None, dagger=None,
                  dagger_beta=0.0, lookahead=None):
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
        # Pistas SEPARADAS para avaliar generalizacao: o modelo nunca as ve.
        "pistas_teste": list(pistas_teste or []),
        "fim_m": fim_m,
        "lookahead_m": float(lookahead if lookahead else prof.get("lookahead", 4.0)),
        # DAgger (2026-10-10): o MODELO dirige e o expert rotula. None = coleta comum.
        "dagger": (None if not dagger else {"modelo": os.path.basename(str(dagger)),
                                            "beta": dagger_beta}),
        "escala": track_cfg0.get("escala"),
        "camera": actor_cfg.get("camera", {}),
        "lidar_sectors": {"n_sectors": LIDAR_N_SECTORS, "max_range_m": LIDAR_MAX_RANGE_M},
        "tracado": tracado, "margem_parede_m": margem,
        "recovery_modo": ("ruido+empurrao" if empurrar else "ruido"),
        "recovery_amp": recovery_amp,
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
            # RUIDO em vez de TELEPORTE para gerar recuperacao.
            #
            # Medido em 2026-10-04: o teleporte (lateral ate 1,2 m + ate 18 graus
            # de guinada) TRAVA o carro nesta pista. O ep0 ficou 54% parado e
            # terminou com v=0,01; os seis episodios seguintes foram 100% parados.
            # set_transform mantem a velocidade antiga enquanto troca a pose, e um
            # Tesla de 4,7 m atravessado numa faixa de 6,36 encrava.
            #
            # O injetor perturba o esterco APLICADO em surtos e grava como rotulo o
            # comando LIMPO do expert. O carro deriva por fisica, nunca em pose
            # impossivel, e aprende a correcao para os estados em que derivou.
            # Encrave: o veiculo nao e recriado entre episodios, entao um carro
            # preso contamina TODO o resto da coleta em silencio. Medido duas
            # vezes: 11 episodios de 1200 quadros imoveis com "dropped 0".
            travado = StuckDetector(v_min=0.3, steps=int(2.0 / fixed))
            # Janela depois do toque (ver JanelaDeColisao): reiniciada a cada
            # episodio -- solta no laco, ela descartava episodios inteiros.
            janela = JanelaDeColisao(int(1.0 / fixed))
            ruido = SteeringNoiseInjector(
                dt=fixed, active_fraction=0.3, amplitude=recovery_amp,
                seed=seed) if recovery else None
            sched = RecoveryScheduler(interval_steps=max(1, int(recovery_every / fixed)),
                                      window_steps=int(1.5 / fixed), enabled=recovery)
            rng = random.Random(seed)
            # DAgger: o v5 imita o expert a 99-100% nas corridas DELE, mas bate quando
            # dirige sozinho -- um erro pequeno leva a um estado que o expert nunca
            # visitou, e num "U" a 93% do esterco nao ha reserva para voltar. Aqui o
            # modelo dirige e o expert diz, em cada estado que o MODELO visitou, o que
            # deveria ter feito. `dagger_beta` = chance de o expert dirigir um passo.
            politica = None
            if dagger:
                from ai.model_policy import DrivingPolicy
                politica = DrivingPolicy(dagger)
                logger.info("DAgger: %s dirige (beta=%.2f), o expert rotula",
                            os.path.basename(str(dagger)), dagger_beta)
            rng_dagger = random.Random(seed + 7)
            fator = 12.0 if str(track_cfg0.get("escala", "meio")).lower() == "real" else 1.0
            z_spawn = float(track_cfg0.get("z", 0.05)) * fator + 0.3

            global_ep = 0
            for item in pistas:
                pista, n_eps = pista_e_episodios(item, episodes_por_pista)
                track_cfg = dict(track_cfg0)
                track_cfg["preset"] = pista
                # obstaculos fora na coleta (Pure Pursuit nao desvia deles)
                track_cfg["obstacles"] = {"count": 0}
                pista_actors = []
                logger.info("=== PISTA %s ===", pista)
                build_track(world, track_cfg, pista_actors)
                centerline = track_centerline(track_cfg)
                lim_fora = track_width(track_cfg)   # desvio > largura util = fora da pista
                # Pista com FIM (2026-10-09): cada episodio vai do comeco ate
                # `fim_m` antes da parede do fim, e o ego volta ao comeco.
                fechado = pista_fechada(pista, fator)
                # O expert segue o TRACADO, nao o eixo. O eixo da curva pede raio
                # 3,18 m e o carro faz 7,50 (0,625 medidos x12): seguir o eixo
                # sempre foi impossivel, e e disso que vinha o subesterco. O eixo
                # continua servindo para medir desvio -- ele e o meio da faixa.
                trajeto = _trajeto_do_expert(tracado, centerline, track_cfg, margem,
                                             fechado=fechado)

                # Nasce no EIXO, nao no traçado: o traçado encosta no limite do
                # corredor e o CARLA recusa o spawn por colisao com a parede. O
                # Pure Pursuit converge para o traçado nos primeiros metros.
                x0, y0, yaw0 = (centerline[0] if fechado
                                else ponto_de_largada(centerline, RECUO_LARGADA_M))
                spawn_tf = carla.Transform(carla.Location(x0, y0, z_spawn),
                                           carla.Rotation(yaw=math.degrees(yaw0)))
                # UMA fonte para o esterco: o mesmo numero limita a roda no
                # CARLA e normaliza o Pure Pursuit logo abaixo. Separa-los faz
                # o expert comandar 1.0 enquanto o carro gira muito mais.
                max_steer = float(prof.get("max_steer_deg", 70.0))
                # A roda do CARLA vira so ~82% do configurado (ver steer_scale):
                # o limite FISICO sobe, e o Pure Pursuit normaliza pelo EFETIVO.
                max_steer_fisico = sim_physical_max_steer_deg(max_steer)
                ego, sensors = spawn_actor_vehicle(
                    world, pista_actors, actor_cfg, spawn_transform=spawn_tf,
                    max_steer_deg=max_steer_fisico)
                for _ in range(10):
                    world.tick()
                # Sensor de colisao: o empurrao pode por o carro encostado na
                # parede, e MEDIDO no dataset_oval_v2, 656 quadros (4,7%) saiam
                # com a carroceria sobreposta a parede e 138 deles com o carro a
                # menos de 1 m/s e esterco -1,000. Isso ensina "nariz na parede
                # -> trava total", que nao e recuperacao, e moagem.
                colisoes = _attach_collision_sensor(world, ego, pista_actors)

                pp = PurePursuit(
                    trajeto, wheelbase=_wheelbase(ego),
                    lookahead=float(lookahead if lookahead else prof.get("lookahead", 4.0)),
                    target_speed=float(prof.get("target_speed", 3.0)),
                    k_throttle=float(prof.get("k_throttle", 0.5)),
                    max_steer_deg=max_steer, fechado=fechado)
                # O carro nasce no eixo, que pode estar longe do ponto 0 do
                # traçado. O `_nearest_ahead` so olha 80 pontos a frente do
                # ultimo indice, entao sem isto ele mira do outro lado da pista
                # e o expert se debate no lugar -- 0,26 m/s em vez de 1,7.
                pp.idx = min(range(len(trajeto)),
                             key=lambda i: (trajeto[i][0] - x0) ** 2
                             + (trajeto[i][1] - y0) ** 2)

                for _ep in range(n_eps):
                    if not fechado and _ep > 0:
                        _volta_ao_inicio(world, ego, (x0, y0, yaw0), z_spawn)
                        pp.idx = min(range(len(trajeto)),
                                     key=lambda i: (trajeto[i][0] - x0) ** 2
                                     + (trajeto[i][1] - y0) ** 2)
                        travado = StuckDetector(v_min=0.3, steps=int(2.0 / fixed))
                    writer = EpisodeWriter(_episode_dir(out_dir, global_ep))
                    global_ep += 1
                    kept = dropped = recovered = 0
                    chegou = False
                    col_ini = len(colisoes)
                    janela.reinicia(len(colisoes))
                    n_destravadas = 0
                    sched.reset()   # o `step` recomeca em 0: o relogio tem que recomecar junto
                    obs_ant = None
                    passos_modelo = 0
                    try:
                        for step in range(steps_per_ep):
                            janela.atualiza(len(colisoes), step)
                            v_agora = _speed_ms(ego)
                            if travado.update(v_agora):
                                _destravar(ego, trajeto)
                                n_destravadas += 1
                            moving = v_agora >= recovery_min_speed
                            # Perto das pontas de uma pista com fim as normais do
                            # eixo nao valem (falta vizinho de um lado), e um
                            # empurrao perto do fim joga o carro na parede.
                            longe_das_pontas = fechado or (pp.idx > 8
                                                           and pp.restante() > 3 * fim_m)
                            if (empurrar and longe_das_pontas
                                    and sched.should_teleport(step, moving)):
                                _empurrao(ego, centerline,
                                          track_width(track_cfg) / 2.0,
                                          CAR_WIDTH_M * SCALE, rng, recovery_yaw,
                                          recovery_lat)
                                sched.mark(step)

                            tf = ego.get_transform()
                            speed = _speed_ms(ego)
                            steer, throttle, brake = pp.control(
                                tf.location.x, tf.location.y, math.radians(tf.rotation.yaw), speed)
                            # O rotulo e sempre o esterco LIMPO; o ruido so vai para
                            # o atuador. E isso que transforma a deriva em exemplo
                            # de recuperacao em vez de em ruido no alvo.
                            # Quem dirige. Encostado na parede, o expert assume:
                            # esses quadros saem do dataset de qualquer jeito, e o
                            # modelo raspando so encravaria o carro.
                            modelo_dirige = (politica is not None and obs_ant is not None
                                             and not janela.batendo(step)
                                             and rng_dagger.random() >= dagger_beta)
                            if modelo_dirige:
                                ruidando = True
                                aplicado = max(-1.0, min(1.0, float(politica(obs_ant)[0])))
                                passos_modelo += 1
                            elif ruido is not None:
                                extra, ruidando = ruido.step()
                                aplicado = max(-1.0, min(1.0, steer + extra))
                            else:
                                ruidando, aplicado = False, steer
                            ego.apply_control(carla.VehicleControl(
                                steer=aplicado, throttle=throttle, brake=brake))
                            world.tick()

                            obs = read_observation(ego, sensors)
                            obs_ant = obs
                            dev, _ = deviation_from_centerline(centerline, tf.location.x, tf.location.y)
                            recovering = ruidando or (empurrar and sched.is_recovering(step))
                            # Quadro com o carro parado nao e exemplo de nada: a
                            # imagem nao muda e o rotulo e trava total, porque o
                            # expert continua tentando. Fora do dataset.
                            batendo = janela.batendo(step)
                            if (obs["image"] is not None and dev <= lim_fora
                                    and speed >= 0.3 and not batendo):
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
                            if not fechado and pp.restante() <= fim_m:
                                chegou = True
                                break
                    finally:
                        writer.close()
                    logger.info("  %s ep%d: kept %d (%d recovery), dropped %d%s%s%s",
                                pista, global_ep - 1, kept, recovered, dropped,
                                "" if politica is None else
                                "  [modelo dirigiu %d passos]" % passos_modelo,
                                "" if fechado else ("  [chegou ao fim]" if chegou
                                                    else "  [NAO chegou ao fim]"),
                                "".join(
                                    ([] if not n_destravadas
                                     else ["  [%d destravadas]" % n_destravadas])
                                    + ([] if not janela.toques
                                       else ["  [%d toques]" % janela.toques])
                                    + [_em_que_bateu(colisoes[col_ini:])]))
                    if fechado and kept < 0.5 * steps_per_ep:
                        logger.warning(
                            "  ep%d aproveitou so %d de %d quadros -- o carro passou "
                            "a maior parte do tempo parado. Conferir antes de treinar.",
                            global_ep - 1, kept, steps_per_ep)

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
                   help="Presets separados por virgula; 'oval_tcc*8' da 8 episodios "
                        "so a essa pista. Pistas da grade: "
                        "grade:SDSES (codigo), ou os grupos grade:treino, "
                        "grade:teste, grade:fechadas, grade:todas "
                        "(ver ai/pistas_grade.py)")
    p.add_argument("--frac-teste", type=float, default=0.15,
                   help="fracao das pistas abertas da grade separadas para teste")
    p.add_argument("--semente-split", type=int, default=0)
    p.add_argument("--curva-final", action="store_true",
                   help="inclui as pistas que terminam numa curva contra a parede")
    p.add_argument("--fim-m", type=float, default=4.5, metavar="M",
                   help="pista com fim: o episodio acaba a M metros (sim) do fim do "
                        "traçado; 4,5 m = ~37 cm no carro, onde a regra do beco "
                        "ja parou o carro real")
    p.add_argument("--episodes-por-pista", type=int, default=4)
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--no-launch", action="store_true")
    p.add_argument("--quality", default="Low")
    p.add_argument("--tracado", choices=list(MODOS), default="estadio",
                   help="Caminho do expert. 'estadio' e a linha que o carro "
                        "consegue executar; 'eixo' e a linha de centro, que "
                        "este carro NAO consegue seguir em curva nenhuma.")
    p.add_argument("--margem", type=float, default=0.0, metavar="M",
                   help="folga extra do traçado ate a parede, em metros de "
                        "simulador. 0 usa o corredor inteiro e o expert raspa: "
                        "o Pure Pursuit erra 0,37 m em media.")
    p.add_argument("--recovery", action="store_true")
    p.add_argument("--recovery-amp", type=float, default=0.2, metavar="A",
                   help="amplitude do ruido de esterco (so com --recovery)")
    p.add_argument("--sem-empurrao", action="store_true",
                   help="so ruido de esterco, sem reposicionar o ego. Medido: so "
                        "o ruido cobre +/-1,4 cm na escala do carro, insuficiente "
                        "para ensinar recuperacao numa faixa de 53 cm.")
    p.add_argument("--recovery-every", type=float, default=5.0)
    p.add_argument("--recovery-lat", type=float, default=0.7,
                   help="teto do empurrao lateral, em metros de simulador")
    p.add_argument("--recovery-yaw", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lookahead", type=float, default=None, metavar="M",
                   help="mira do Pure Pursuit, em metros de sim (padrao: o do settings, "
                        "4.0). Com 4 m o expert se afasta ate 0,6-0,8 m da propria linha")
    p.add_argument("--dagger", default=None, metavar="CKPT",
                   help="DAgger: este modelo dirige e o expert rotula cada quadro")
    p.add_argument("--dagger-beta", type=float, default=0.0, metavar="B",
                   help="chance de o expert dirigir cada passo (0 = so o modelo)")
    a = p.parse_args()

    if a.report:
        print_report(dataset_report(a.report))
        return

    pistas, teste = expande_pistas(a.pistas, frac_teste=a.frac_teste,
                                   semente=a.semente_split, curva_final=a.curva_final)
    logger.info("%d pistas na coleta (%d abertas da grade ficaram para teste)",
                len(pistas), len(teste))
    collect_track(
        settings_path=a.settings, out_dir=a.out, pistas=pistas,
        episodes_por_pista=a.episodes_por_pista, seconds=a.seconds,
        launch=not a.no_launch, quality=a.quality, recovery=a.recovery,
        recovery_every=a.recovery_every, recovery_lat=a.recovery_lat,
        recovery_yaw=a.recovery_yaw, seed=a.seed, tracado=a.tracado, margem=a.margem,
        recovery_amp=a.recovery_amp, empurrar=not a.sem_empurrao,
        fim_m=a.fim_m, pistas_teste=teste, dagger=a.dagger, dagger_beta=a.dagger_beta,
        lookahead=a.lookahead)


if __name__ == "__main__":
    main()
