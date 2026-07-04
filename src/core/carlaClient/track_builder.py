# imports
import math
import random
import logging

import carla

# constants
logger = logging.getLogger(__name__)

# Medidas CONGELADAS das pecas (batem com o generate_pieces.py do repo do Blender).
COMPRIMENTO_RETA = 0.50
RAIO_CURVA = 0.65
LARGURA_PISTA = 0.65     # largura util (chao preto entre as bordas)

# Conectores de cada peca, em coordenadas LOCAIS (identicos ao montar_pista.py):
#   p_in/p_out = posicao (x, y) da entrada/saida; h_in/h_out = direcao de marcha (rad).
CONECTORES = {
    "tcc_reta": {
        "p_in": (-COMPRIMENTO_RETA / 2.0, 0.0), "h_in": 0.0,
        "p_out": (+COMPRIMENTO_RETA / 2.0, 0.0), "h_out": 0.0,
    },
    "tcc_curva90": {
        "p_in": (RAIO_CURVA, 0.0), "h_in": math.pi / 2.0,
        "p_out": (0.0, RAIO_CURVA), "h_out": math.pi,
    },
    "tcc_curva45": {
        "p_in": (RAIO_CURVA, 0.0), "h_in": math.pi / 2.0,
        "p_out": (RAIO_CURVA * math.cos(math.radians(45)),
                  RAIO_CURVA * math.sin(math.radians(45))),
        "h_out": math.pi / 2.0 + math.radians(45),
    },
}


# functions
def _rot2d(ang: float, x: float, y: float) -> tuple[float, float]:
    """Rotaciona o ponto (x, y) por `ang` radianos (rotacao 2D padrao)."""
    ca, sa = math.cos(ang), math.sin(ang)
    return (ca * x - sa * y, sa * x + ca * y)


def _montar(sequencia: list[str]) -> tuple[list[tuple], tuple[float, float, float]]:
    """Logica de "tartaruga": encaixa a ENTRADA de cada peca na SAIDA da anterior.

    Args:
        sequencia: lista de nomes de peca (ex.: ["tcc_reta", "tcc_curva90", ...]).

    Returns:
        (poses, pose_final) onde poses e uma lista de (nome, x, y, alpha_rad) para
        cada peca, e pose_final = (x, y, heading_rad) apos a ultima peca (para
        checar se o circuito fecha).
    """
    px, py, h = 0.0, 0.0, 0.0
    poses = []
    waypoints = []                                # pontos SOBRE a pista (p/ obstaculos)
    for nome in sequencia:
        waypoints.append((px, py, h))             # ponto atual = conector de entrada da peca
        c = CONECTORES[nome]
        pin_x, pin_y = c["p_in"]
        pout_x, pout_y = c["p_out"]
        alpha = h - c["h_in"]                     # gira o heading local ate o global
        rx, ry = _rot2d(alpha, pin_x, pin_y)
        poses.append((nome, px - rx, py - ry, alpha))   # ancora a entrada no ponto atual
        h = h + (c["h_out"] - c["h_in"])          # novo heading = saida girada
        dox, doy = _rot2d(alpha, pout_x - pin_x, pout_y - pin_y)
        px, py = px + dox, py + doy               # avanca para a saida
    return poses, (px, py, h), waypoints


# Presets FECHADOS (giram sempre para a esquerda; total = 360 -> fecham o loop).
def _preset(nome: str) -> list[str]:
    if nome == "oval":       # estadio: reta*3 + 180(2x90) + reta*3 + 180
        return (["tcc_reta"] * 3 + ["tcc_curva90"] * 2) * 2
    if nome == "quadrado":   # 4 cantos de 90
        return (["tcc_reta"] * 2 + ["tcc_curva90"]) * 4
    if nome == "octogono":   # 8 cantos de 45
        return (["tcc_reta"] * 1 + ["tcc_curva45"]) * 8
    if nome == "calibrar":   # trecho curto p/ calibrar eixo
        return ["tcc_reta", "tcc_reta", "tcc_curva90", "tcc_reta"]
    raise ValueError(f"preset de pista desconhecido: '{nome}'")


def build_track(world: carla.World, track_config: dict, actor_list: list) -> list:
    """Monta uma pista com as pecas do TCC (props tcc_*) no mundo dado.

    Args:
        world: mundo CARLA (ja com o mapa carregado).
        track_config: secao "track" do settings.json.
        actor_list: lista onde registrar os atores criados (para limpeza).

    Returns:
        Lista dos atores (pecas) spawnados.
    """
    preset = track_config.get("preset", "oval")
    flip_y = float(track_config.get("flip_y", -1.0))       # calibracao de eixo Blender->CARLA
    flip_yaw = float(track_config.get("flip_yaw", -1.0))
    z = float(track_config.get("z", 0.05))

    # Palco quase vazio: descarrega as camadas do mapa (so ceu + luz).
    if track_config.get("unload_layers", True):
        world.unload_map_layer(carla.MapLayer.All)
        logger.info("Track: camadas do mapa descarregadas (palco vazio)")

    sequencia = _preset(preset)
    poses, (fx, fy, fh), waypoints = _montar(sequencia)

    gap = math.hypot(fx, fy)
    ang = math.degrees(fh) % 360.0
    ang = min(ang, 360.0 - ang)
    logger.info(f"Track '{preset}': {len(sequencia)} pecas | fechamento gap={gap:.3f} m, ang={ang:.1f} deg")

    bl = world.get_blueprint_library()
    props = []
    xs, ys = [], []
    for i, (nome, tx, ty, alpha) in enumerate(poses):
        bp = bl.find("static.prop." + nome)
        # spawn num canto vazio e teleporta (set_transform nao checa colisao de emenda)
        ator = world.try_spawn_actor(bp, carla.Transform(carla.Location(i * 3.0, -100.0, 50.0)))
        if ator is None:
            logger.warning(f"Track: falhou spawn de {nome}")
            continue
        loc = carla.Location(x=tx, y=flip_y * ty, z=z)
        ator.set_transform(carla.Transform(loc, carla.Rotation(yaw=flip_yaw * math.degrees(alpha))))
        actor_list.append(ator)
        props.append(ator)
        xs.append(loc.x)
        ys.append(loc.y)

    logger.info(f"Track: {len(props)}/{len(poses)} pecas colocadas")

    # --- Obstaculos em posicoes ALEATORIAS sobre a pista ---
    # Sorteia um ponto da linha da pista (waypoint) + um deslocamento lateral
    # dentro da largura util (deixando folga p/ o carro passar do outro lado).
    obs_cfg = track_config.get("obstacles", {})
    n_obs = int(obs_cfg.get("count", 0))
    tipos = obs_cfg.get("types", ["tcc_cone", "tcc_mureta", "tcc_pessoa2d"])
    margem = float(obs_cfg.get("margem_lateral", 0.12))     # folga p/ o carro desviar
    lim = max(LARGURA_PISTA / 2.0 - margem, 0.0)
    stage_k = len(poses)                                    # staging distinto das pecas
    n_obs_ok = 0
    for _ in range(n_obs):
        if not waypoints:
            break
        wpx, wpy, wph = random.choice(waypoints)
        nome = random.choice(tipos)
        lat = random.uniform(-lim, lim)
        ox = wpx + lat * (-math.sin(wph))     # desloca perpendicular ao rumo da pista
        oy = wpy + lat * (math.cos(wph))
        bp = bl.find("static.prop." + nome)
        ator = world.try_spawn_actor(bp, carla.Transform(carla.Location(stage_k * 3.0, -200.0, 50.0)))
        stage_k += 1
        if ator is None:
            logger.warning(f"Track: falhou spawn de obstaculo {nome}")
            continue
        loc = carla.Location(x=ox, y=flip_y * oy, z=z)
        ator.set_transform(carla.Transform(loc, carla.Rotation(yaw=flip_yaw * math.degrees(wph))))
        actor_list.append(ator)
        n_obs_ok += 1
    if n_obs:
        logger.info(f"Track: {n_obs_ok}/{n_obs} obstaculos aleatorios colocados")

    # Camera de topo enquadrando a pista.
    if xs:
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        span = max(max(xs) - min(xs), max(ys) - min(ys), 3.0)
        spectator = world.get_spectator()
        spectator.set_transform(carla.Transform(
            # carla.Location(cx, cy, span * 1.6 + 3.0),
            # carla.Rotation(pitch=-89.0)))
            carla.Location(cx - span, cy, span * 0.7),
            carla.Rotation(pitch=-35.0, yaw=0.0)
        ))
        logger.info("Track: camera posicionada sobre a pista")

    return props
