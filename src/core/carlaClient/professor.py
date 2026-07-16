# Anotacoes preguicosas (type hints como texto) -> o modulo importa sem o CARLA,
# permitindo testar o controlador (matematica pura) em Python puro.
from __future__ import annotations

# imports
import os
import csv
import math
import logging
import datetime

try:
    import carla
except ImportError:          # ambiente sem CARLA (ex.: testar so o controlador)
    carla = None

# reusa a geometria da pista: os waypoints saem da MESMA descricao que monta a pista
from core.carlaClient.track_builder import gerar_waypoints

logger = logging.getLogger(__name__)


# =============================================================================
# 1. CONTROLADOR — PURE PURSUIT (esterco) + P de velocidade (acelerador/freio)
# =============================================================================
# Pure pursuit e' o classico "mira num ponto a frente e vira pra ele". Dado onde o
# carro esta (x, y, yaw) e a linha de centro (waypoints), ele:
#   1) acha o waypoint mais proximo A FRENTE (nao atras);
#   2) anda por essa linha ate um ponto de LOOKAHEAD a distancia Ld;
#   3) calcula o ANGULO alpha entre o "nariz" do carro e esse ponto;
#   4) o esterco vem da geometria da bicicleta: delta = atan2(2*L*sin(alpha), Ld),
#      onde L = distancia entre eixos. Ld maior => curva mais suave (e vice-versa).
# So depende de matematica -> da' pra testar sem o CARLA (ver test no fim do arquivo
# do repo de scripts). Frame do CARLA e' canhoto: forward=(cos yaw, sin yaw),
# right=(-sin yaw, cos yaw); esterco +1 = direita.

class PurePursuit:
    def __init__(self, waypoints, wheelbase=2.8, lookahead=4.0,
                 target_speed=5.0, k_throttle=0.5, max_steer_deg=70.0):
        """Args:
            waypoints    : lista de (x, y, yaw_rad) no frame do CARLA (linha de centro).
            wheelbase    : distancia entre eixos do carro (m).
            lookahead    : distancia Ld do ponto de mira a frente (m).
            target_speed : velocidade alvo (m/s).
            k_throttle   : ganho do P de velocidade.
            max_steer_deg: esterco fisico maximo do carro (p/ normalizar delta em [-1,1])."""
        self.wps = waypoints
        self.n = len(waypoints)
        self.L = wheelbase
        self.Ld = lookahead
        self.target_speed = target_speed
        self.k = k_throttle
        self.max_steer = math.radians(max_steer_deg)
        self.idx = 0            # ultimo waypoint mais proximo (memoria de progresso)
        self.laps = 0           # voltas completas (detecta o "wrap" do indice)

    def _nearest_ahead(self, x, y, janela=80):
        """Acha o waypoint mais proximo, buscando SO PARA FRENTE a partir do ultimo
        (janela limitada). Isso evita 'voltar' para um trecho ja passado quando a
        pista se aproxima de si mesma, e e' O(janela) em vez de O(n)."""
        melhor, melhor_d = self.idx, 1e18
        for off in range(janela):
            i = (self.idx + off) % self.n
            dx = self.wps[i][0] - x
            dy = self.wps[i][1] - y
            d = dx * dx + dy * dy
            if d < melhor_d:
                melhor_d, melhor = d, i
        # conta volta: se o indice 'deu a volta' (de perto do fim para perto do inicio)
        if melhor < self.idx and (self.idx - melhor) > self.n // 2:
            self.laps += 1
        self.idx = melhor
        return melhor

    def _ponto_lookahead(self, x, y):
        """Do waypoint mais proximo, anda pela linha acumulando distancia ate passar
        de Ld. Retorna esse ponto (a 'mira')."""
        i = self._nearest_ahead(x, y)
        j = i
        dist = 0.0
        while dist < self.Ld:
            nj = (j + 1) % self.n
            dist += math.hypot(self.wps[nj][0] - self.wps[j][0],
                               self.wps[nj][1] - self.wps[j][1])
            j = nj
            if j == i:          # deu a volta inteira (pista curtissima) -> para
                break
        return self.wps[j]

    def control(self, x, y, yaw, speed):
        """Calcula (steer, throttle, brake) em [-1,1]/[0,1]/[0,1] para o estado atual.
            x, y   : posicao do carro (m, frame CARLA).
            yaw    : orientacao (RAD, frame CARLA).
            speed  : velocidade escalar atual (m/s)."""
        tx, ty, _ = self._ponto_lookahead(x, y)
        dx, dy = tx - x, ty - y

        # Leva o vetor ate a mira para o FRAME DO CARRO (forward / right do CARLA).
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        ex = cos_y * dx + sin_y * dy       # componente para frente (forward)
        ey = -sin_y * dx + cos_y * dy      # componente lateral (right, + = direita)

        Ld_real = math.hypot(ex, ey)
        if Ld_real < 1e-3:
            steer = 0.0
        else:
            alpha = math.atan2(ey, ex)                       # angulo ate a mira
            delta = math.atan2(2.0 * self.L * math.sin(alpha), Ld_real)
            steer = max(-1.0, min(1.0, delta / self.max_steer))

        # P de velocidade: acelera se abaixo do alvo, freia se bem acima.
        err = self.target_speed - speed
        throttle, brake = 0.0, 0.0
        if err >= 0:
            throttle = min(1.0, self.k * err)
        elif err < -0.5:
            brake = min(1.0, -self.k * err)
        return steer, throttle, brake


# =============================================================================
# 2. WAYPOINTS NO FRAME DO CARLA (mesma calibracao de eixo da pista)
# =============================================================================
def _waypoints_carla(track_cfg):
    """Gera a linha de centro no MESMO frame em que a pista foi montada.

    build_track usa preset+escala (fator) e aplica flip_y/flip_yaw ao posicionar as
    pecas. Aqui reproduzimos isso: geramos os waypoints 'matematicos' e aplicamos o
    mesmo flip, devolvendo (x, y, yaw_rad) prontos para o controlador."""
    preset = track_cfg.get("preset", "oval")
    escala = str(track_cfg.get("escala", "meio")).lower()
    fator = 12.0 if escala == "real" else 1.0
    flip_y = float(track_cfg.get("flip_y", -1.0))
    flip_yaw = float(track_cfg.get("flip_yaw", -1.0))
    # espacamento em METROS no frame ja escalado (real): 0.5 => waypoints a cada 0,5 m
    # (denso, ~154 numa pista de ~77 m -> rotulos de esterco suaves).
    espac = float(track_cfg.get("professor", {}).get("espacamento", 0.5))

    wps = gerar_waypoints(preset, fator, espac)
    return [(w["x"], flip_y * w["y"], flip_yaw * w["heading"]) for w in wps]


# =============================================================================
# 3. SENSOR "CRU" — guarda o objeto do CARLA para usar o save_to_disk nativo
# =============================================================================
# A venv nao tem PIL/cv2, entao gravamos com o save_to_disk do proprio CARLA
# (PNG p/ camera, PLY p/ LiDAR, em C++). Para isso precisamos do objeto de sensor
# CRU (nao do numpy dos wrappers) -> este mini-wrapper so guarda o ultimo dado.
class _SensorCru:
    def __init__(self, sensor):
        self.sensor = sensor
        self.ultimo = None
        sensor.listen(self._cb)

    def _cb(self, data):
        self.ultimo = data


def _montar_sensores_professor(world, ego, actor_list, cam_cfg, lidar_cfg):
    """Anexa camera RGB + LiDAR ao ego (guardando os dados crus p/ save_to_disk)."""
    bl = world.get_blueprint_library()

    cam_bp = bl.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(cam_cfg.get("width", 800)))
    cam_bp.set_attribute("image_size_y", str(cam_cfg.get("height", 600)))
    cam_bp.set_attribute("fov", str(cam_cfg.get("fov", 90)))
    cam_tf = carla.Transform(
        carla.Location(x=cam_cfg.get("x", 1.5), y=0.0, z=cam_cfg.get("z", 1.4)),
        carla.Rotation(pitch=cam_cfg.get("pitch", -15.0)),   # -15 = decisao de treino
    )
    cam = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)
    actor_list.append(cam)

    lid_bp = bl.find("sensor.lidar.ray_cast")
    for k, v in {"channels": 32, "points_per_second": 100000, "rotation_frequency": 20,
                 "range": 100, "upper_fov": 15, "lower_fov": -25}.items():
        lid_bp.set_attribute(k, str(lidar_cfg.get(k, v)))
    lid_tf = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.0))
    lid = world.spawn_actor(lid_bp, lid_tf, attach_to=ego)
    actor_list.append(lid)

    return _SensorCru(cam), _SensorCru(lid)


# =============================================================================
# 4. DATASET (behavior cloning): imagens + lidar + control.csv
# =============================================================================
class DataLogger:
    def __init__(self, raiz="dataset"):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(raiz, "prof_" + stamp)
        self.img_dir = os.path.join(self.dir, "images")
        self.lid_dir = os.path.join(self.dir, "lidar")
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.lid_dir, exist_ok=True)
        self.csv_path = os.path.join(self.dir, "control.csv")
        self.f = open(self.csv_path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["frame", "t", "img", "lidar",
                         "steer", "throttle", "brake", "speed", "x", "y", "yaw_deg"])
        self.n = 0
        logger.info("Professor: dataset em %s", self.dir)

    def log(self, cam_cru, lid_cru, control, speed, tf):
        """Grava um exemplo: imagem + nuvem + a acao tomada (o que a rede vai clonar)."""
        fid = self.n
        img_name = "%06d.png" % fid
        lid_name = "%06d.ply" % fid
        if cam_cru.ultimo is not None:
            cam_cru.ultimo.save_to_disk(os.path.join(self.img_dir, img_name))
        if lid_cru.ultimo is not None:
            lid_cru.ultimo.save_to_disk(os.path.join(self.lid_dir, lid_name))
        loc, rot = tf.location, tf.rotation
        self.w.writerow([fid, "%.3f" % (fid * 0.05), img_name, lid_name,
                         "%.4f" % control[0], "%.4f" % control[1], "%.4f" % control[2],
                         "%.3f" % speed, "%.3f" % loc.x, "%.3f" % loc.y, "%.2f" % rot.yaw])
        self.n += 1

    def close(self):
        self.f.close()
        logger.info("Professor: %d exemplos gravados em %s", self.n, self.dir)


# =============================================================================
# 5. ORQUESTRACAO — spawna o ego, dirige seguindo os waypoints e grava o dataset
# =============================================================================
def _wheelbase(vehicle):
    """Le a distancia entre eixos do carro pela fisica (posicoes das rodas, em cm)."""
    try:
        w = vehicle.get_physics_control().wheels
        xs = [wh.position.x for wh in w]     # cm, frame do mundo no momento
        return max(0.5, (max(xs) - min(xs)) / 100.0)
    except Exception:
        return 2.8


def run_professor(world, track_cfg, actor_list):
    """Roda o 'professor': ego + sensores dirigindo pela pista, gravando o dataset.

    Pre-requisito: a pista JA foi montada (build_track) no mesmo `world`/`track_cfg`.
    O ego e' um carro NATIVO do CARLA (escala real) — por isso combina com escala 'real'."""
    prof = track_cfg.get("professor", {})
    waypoints = _waypoints_carla(track_cfg)
    if len(waypoints) < 3:
        logger.error("Professor: waypoints insuficientes (%d)", len(waypoints))
        return

    escala = str(track_cfg.get("escala", "meio")).lower()
    if escala != "real":
        logger.warning("Professor: escala '%s' — o ego e' um carro nativo (grande); "
                       "o esperado e' escala 'real'.", escala)

    # --- spawn do ego no 1o waypoint, virado no rumo da pista ---
    bl = world.get_blueprint_library()
    veh_bp = bl.filter(prof.get("vehicle_filter", "vehicle.tesla.model3"))[0]
    x0, y0, yaw0 = waypoints[0]
    z0 = float(track_cfg.get("z", 0.05)) * (12.0 if escala == "real" else 1.0) + 0.3
    spawn_tf = carla.Transform(carla.Location(x0, y0, z0),
                               carla.Rotation(yaw=math.degrees(yaw0)))
    ego = world.try_spawn_actor(veh_bp, spawn_tf)
    if ego is None:
        logger.error("Professor: falha ao spawnar o ego em (%.1f, %.1f)", x0, y0)
        return
    actor_list.append(ego)
    logger.info("Professor: ego %s spawnado; %d waypoints", ego.type_id, len(waypoints))

    cam, lid = _montar_sensores_professor(
        world, ego, actor_list, prof.get("camera", {}), prof.get("lidar", {}))

    pp = PurePursuit(
        waypoints,
        wheelbase=_wheelbase(ego),
        lookahead=float(prof.get("lookahead", 4.0)),
        target_speed=float(prof.get("target_speed", 5.0)),
        k_throttle=float(prof.get("k_throttle", 0.5)),
        max_steer_deg=float(prof.get("max_steer_deg", 70.0)),
    )
    logger.info("Professor: L=%.2f m, Ld=%.1f m, alvo=%.1f m/s",
                pp.L, pp.Ld, pp.target_speed)

    logger_ds = DataLogger(prof.get("dataset_dir", "dataset")) if prof.get("log", True) else None
    # laps/max_steps: 0 (ou negativo) = SEM limite -> roda INDEFINIDAMENTE ate Ctrl+C.
    # Se ambos forem 0, o professor da voltas p/ sempre e nada e' destruido ate voce parar.
    max_laps = int(prof.get("laps", 2))
    max_steps = int(prof.get("max_steps", 6000))
    if max_laps <= 0 and max_steps <= 0:
        logger.info("Professor: rodando INDEFINIDAMENTE (laps=0, max_steps=0) — Ctrl+C p/ parar.")
    log_every = max(1, int(prof.get("log_every", 1)))
    spectator = world.get_spectator()

    try:
        step = 0
        while True:
            world.tick()
            tf = ego.get_transform()
            vel = ego.get_velocity()
            speed = math.hypot(vel.x, vel.y)
            steer, throttle, brake = pp.control(
                tf.location.x, tf.location.y, math.radians(tf.rotation.yaw), speed)
            ego.apply_control(carla.VehicleControl(
                throttle=throttle, steer=steer, brake=brake))

            if logger_ds and (step % log_every == 0):
                logger_ds.log(cam, lid, (steer, throttle, brake), speed, tf)

            # camera de perseguicao (spectator atras do ego)
            fwd = tf.get_forward_vector()
            spectator.set_transform(carla.Transform(
                carla.Location(tf.location.x - fwd.x * 8, tf.location.y - fwd.y * 8,
                               tf.location.z + 4),
                carla.Rotation(pitch=-15, yaw=tf.rotation.yaw)))

            step += 1
            # limites ATIVOS so quando > 0 (0 = infinito)
            if max_laps > 0 and pp.laps >= max_laps:
                logger.info("Professor: %d voltas completas — encerrando.", pp.laps)
                break
            if max_steps > 0 and step >= max_steps:
                logger.info("Professor: max_steps (%d) atingido — encerrando.", max_steps)
                break
    except KeyboardInterrupt:
        logger.info("Professor: interrompido pelo usuario (Ctrl+C).")
    finally:
        ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        if logger_ds:
            logger_ds.close()
