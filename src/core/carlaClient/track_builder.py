# Anotacoes "preguicosas": faz o Python tratar as type hints como TEXTO (nao as
# avalia na definicao). Sem isto, `def build_track(world: carla.World, ...)`
# tentaria avaliar `carla.World` ao importar o modulo e quebraria quando o CARLA
# nao estiver instalado (ver o import protegido abaixo).
from __future__ import annotations

# imports
import math
import random
import logging

# `carla` so e' preciso para SPAWNAR (build_track / medir_assets). A parte de
# GEOMETRIA (montagem da pista e geracao de waypoints) e' matematica pura e roda
# sem o CARLA instalado -> permite testar os waypoints em Python puro. Por isso o
# import e' protegido: se o CARLA nao existir, `carla` fica None e so as funcoes
# que o usam e' que vao falhar (as de geometria continuam funcionando).
try:
    import carla
except ImportError:  # ambiente sem CARLA (ex.: rodar so o gerador de waypoints)
    carla = None

# constants
logger = logging.getLogger(__name__)

# Medidas das pecas (batem com o generate_pieces.py do repo do Blender).
# Revisao jul/2026 (pistas do estande): a pista e' feita de 3 pecas, com medidas de
# FAIXA UTIL (onde o carro passa, SEM a parede). Duas retas (branca 31,5 / cinza 21,5)
# + curva 90 = 1/4 de DISCO 53x53 (raio externo 53, interno 0 -> eixo central 26,5 cm).
COMPRIMENTO_RETA_BRANCA = 0.315   # reta longa
COMPRIMENTO_RETA_CINZA = 0.215    # reta curta
LARGURA_PISTA = 0.53              # faixa util (onde o carro passa)
RAIO_CURVA = LARGURA_PISTA / 2.0  # 0,265 m — eixo central da curva (1/4 de disco)

# Conectores de cada peca, em coordenadas LOCAIS.
#   p_in/p_out = posicao (x, y) da entrada/saida; h_in/h_out = direcao de marcha (rad).
# `fator` escala as POSICOES (nao os angulos) p/ casar com o tamanho das pecas:
#   1.0 = conjunto 1/12 (tcc_*);  12.0 = conjunto real (tcc_*_real).
#
# Curva a ESQUERDA x DIREITA com a MESMA peca (1/4 de disco): a peca ocupa o
# quadrante +X+Y, com bordas radiais em +X (angulo 0) e +Y (angulo 90). Entrar pela
# borda +X e sair pela +Y vira a ESQUERDA (+90). Percorrer o MESMO disco ao contrario
# (entrar pela +Y, sair pela +X) vira a DIREITA (-90) — sem espelhar a malha. Por isso
# ha dois conectores ("tcc_curva90" e "tcc_curva90_r") que apontam para o MESMO prop.
def _conectores(fator=1.0):
    Lb = COMPRIMENTO_RETA_BRANCA * fator
    Lc = COMPRIMENTO_RETA_CINZA * fator
    R = RAIO_CURVA * fator
    return {
        "tcc_reta_branca": {
            "p_in": (-Lb / 2.0, 0.0), "h_in": 0.0,
            "p_out": (+Lb / 2.0, 0.0), "h_out": 0.0,
        },
        "tcc_reta_cinza": {
            "p_in": (-Lc / 2.0, 0.0), "h_in": 0.0,
            "p_out": (+Lc / 2.0, 0.0), "h_out": 0.0,
        },
        # Curva 90 a ESQUERDA (+90): entra pela borda +X (heading +Y), sai pela +Y (heading -X).
        "tcc_curva90": {
            "p_in": (R, 0.0), "h_in": math.pi / 2.0,
            "p_out": (0.0, R), "h_out": math.pi,
        },
        # Curva 90 a DIREITA (-90): mesmo disco percorrido ao contrario — entra pela
        # borda +Y (heading +X), sai pela +X (heading -Y).
        "tcc_curva90_r": {
            "p_in": (0.0, R), "h_in": 0.0,
            "p_out": (R, 0.0), "h_out": -math.pi / 2.0,
        },
        "tcc_curva45": {
            "p_in": (R, 0.0), "h_in": math.pi / 2.0,
            "p_out": (R * math.cos(math.radians(45)), R * math.sin(math.radians(45))),
            "h_out": math.pi / 2.0 + math.radians(45),
        },
    }


# Nome logico da peca -> nome do PROP no CARLA (sem o sufixo de escala).
# A curva a direita usa a MESMA malha da esquerda (so muda o conector/pose).
def _prop_base(nome: str) -> str:
    if nome == "tcc_curva90_r":
        return "tcc_curva90"
    return nome


# functions
def _rot2d(ang: float, x: float, y: float) -> tuple[float, float]:
    """Rotaciona o ponto (x, y) por `ang` radianos (rotacao 2D padrao)."""
    ca, sa = math.cos(ang), math.sin(ang)
    return (ca * x - sa * y, sa * x + ca * y)


def _montar(sequencia, conectores, fechar_gap=False, fator=1.0):
    """Logica de "tartaruga": encaixa a ENTRADA de cada peca na SAIDA da anterior.

    Args:
        sequencia: lista de nomes de peca (ex.: ["tcc_reta_branca", "tcc_curva90", ...]).
        fechar_gap: se True e a pista e' um laco QUASE fechado, distribui o residuo
            de fechamento por todas as emendas (poucos mm em cada uma).
        fator: escala da geometria (1.0 = 1/12, 12.0 = real). O limiar de "residuo
            pequeno" escala junto (0.15 * fator) — senao, em escala real um residuo
            proporcionalmente identico ao de 1/12 (ex.: pista1 ~0,1 m -> 1,2 m)
            passaria do limiar fixo e NAO seria distribuido, deixando um salto grande
            numa unica emenda. Por que existe: os layouts do grupo sao desenhos em grade
            esquematica — com os comprimentos reais (31,5/21,5/26,5) alguns lacos
            nao fecham EXATO por aritmetica. Na pista fisica isso e' absorvido por
            folguinhas nas emendas; aqui fazemos o mesmo, em vez de deixar uma
            unica emenda com o erro inteiro.

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
        c = conectores[nome]
        pin_x, pin_y = c["p_in"]
        pout_x, pout_y = c["p_out"]
        alpha = h - c["h_in"]                     # gira o heading local ate o global
        rx, ry = _rot2d(alpha, pin_x, pin_y)
        poses.append((nome, px - rx, py - ry, alpha))   # ancora a entrada no ponto atual
        h = h + (c["h_out"] - c["h_in"])          # novo heading = saida girada
        dox, doy = _rot2d(alpha, pout_x - pin_x, pout_y - pin_y)
        px, py = px + dox, py + doy               # avanca para a saida
    # Distribui o residuo de fechamento (se pedido e se for pequeno): a peca i
    # desliza -erro*(i/N); cada emenda fica com erro/N (~mm). Pecas e waypoints
    # sao corrigidos juntos para continuarem casando.
    if fechar_gap and 1e-9 < math.hypot(px, py) < 0.15 * fator:
        ex, ey, n = px, py, len(sequencia)
        poses = [(nome, x - ex * i / n, y - ey * i / n, a)
                 for i, (nome, x, y, a) in enumerate(poses)]
        waypoints = [(x - ex * i / n, y - ey * i / n, hh)
                     for i, (x, y, hh) in enumerate(waypoints)]
        px, py = ex / n, ey / n                   # residuo que sobra na ultima emenda
    return poses, (px, py, h), waypoints


# Presets FECHADOS. Convencao: "tcc_curva90" vira +90 (esquerda), "tcc_curva90_r"
# vira -90 (direita). Para um laco simples fechar, o giro liquido = +/-360.
def _preset(nome: str) -> list[str]:
    if nome == "oval":       # estadio: reta*3 + 180(2x90) + reta*3 + 180
        return (["tcc_reta_branca"] * 3 + ["tcc_curva90"] * 2) * 2
    if nome == "quadrado":   # 4 cantos de 90 a esquerda
        return (["tcc_reta_branca"] * 2 + ["tcc_curva90"]) * 4
    if nome == "quadrado_dir":  # 4 cantos de 90 a DIREITA (testa a curva espelhada)
        return (["tcc_reta_branca"] * 2 + ["tcc_curva90_r"]) * 4
    if nome == "octogono":   # 8 cantos de 45
        return (["tcc_reta_branca"] * 1 + ["tcc_curva45"]) * 8
    if nome == "calibrar":   # trecho curto p/ calibrar eixo
        return ["tcc_reta_branca", "tcc_reta_branca", "tcc_curva90", "tcc_reta_branca"]
    # --- Pistas do estande (SchemaPista1/2/3) — transcricao peca a peca ---
    # Preenchidas apos conferir a sequencia fechada de cada uma com o grupo.
    if nome in ("pista1", "pista2", "pista3"):
        seq = _PISTAS_ESTANDE.get(nome)
        if not seq:
            raise NotImplementedError(
                f"'{nome}': sequencia ainda nao transcrita (falta a leitura fechada do esquema)")
        return seq
    raise ValueError(f"preset de pista desconhecido: '{nome}'")


# Sequencias das pistas do estande (SchemaPista*.png). Vao sendo preenchidas conforme
# a transcricao peca a peca for confirmada e o laco fechar (gap ~ 0). Pecas:
#   tcc_reta_branca (31,5) | tcc_reta_cinza (21,5) | tcc_curva90 (esq) | tcc_curva90_r (dir)
_B, _C = "tcc_reta_branca", "tcc_reta_cinza"
_L, _R = "tcc_curva90", "tcc_curva90_r"
# Um "mergulho" (lomba p/ baixo do fundo da pista 3): desce (L), nivela (R),
# sobe (R), nivela (L) — 4 curvas de 90, sem retas. Os picos entre mergulhos
# ENCOSTAM na faixa do topo (topo do disco = borda inferior da faixa, folga 0).
_MERGULHO_P3 = [_L, _R, _R, _L]

_PISTAS_ESTANDE = {
    # Pista 1 (SchemaPista1): anel com topo de 7B+1C, laterais curtas (curva+B+curva;
    # a direita tem uma B extra antes do mergulho) e MERGULHO central no fundo
    # (desce 2 curvas, 3B no fundo, sobe 2 curvas). O laco nao fecha exato por
    # aritmetica dos comprimentos (residuo ~10 cm) -> `fechar_gap` distribui
    # ~4,5 mm por emenda, como a montagem fisica faz.
    "pista1": ([_B] * 7 + [_C]              # topo (esq -> dir)
               + [_R, _B, _R]               # lateral direita desce
               + [_B]                       # branca extra (lado direito)
               + [_L, _R]                   # desce ao mergulho
               + [_B, _B, _B]               # fundo do mergulho
               + [_R, _L]                   # sobe do mergulho
               + [_R, _B, _R]),             # lateral esquerda sobe e fecha
    # Pista 2 (SchemaPista2): anel com CHICANE central em "∩" (sobe, atravessa,
    # desce; o topo da chicane ENCOSTA na faixa de cima). Fecha EXATA (gap 0).
    "pista2": ([_C, _C] + [_B] * 6          # topo (cinzas a esquerda)
               + [_R, _B, _B, _C, _R]       # lateral direita desce (cinza embaixo)
               + [_B, _B]                   # fundo, a direita da chicane
               + [_R, _B, _L, _B, _L, _B, _R]   # chicane (sobe, topo, desce)
               + [_B]                       # fundo, a esquerda da chicane
               + [_R, _C, _B, _B, _R]),     # lateral esquerda sobe e fecha
    # Pista 3 (SchemaPista3, CONFIRMADA 2026-07-22): topo 4 brancas + 4 cinzas,
    # caps de 180 nas pontas (2 curvas), fundo com 2 mergulhos cujos picos
    # ENCOSTAM na faixa do topo. Fecha exata: gap=0.
    "pista3": ([_B] * 4 + [_C] * 4 + [_R, _R]
               + _MERGULHO_P3 + _MERGULHO_P3 + [_R, _R]),
}


# =============================================================================
# WAYPOINTS (linha de centro da pista) — para o "professor" do behavior cloning
# =============================================================================
# A pista sao PROPS num palco vazio (sem OpenDRIVE), entao o CARLA NAO tem
# waypoints nativos dela. Nos geramos os nossos, densos, ao longo do EIXO CENTRAL.
# Como a pista e' uma sequencia de modulos e o `_montar` ja da a pose global de
# cada peca, basta amostrar o centro de cada modulo e transformar para o global.

def _norm_ang(a: float) -> float:
    """Normaliza um angulo para o intervalo (-pi, pi]. Evita 'saltos' de 2*pi ao
    comparar/derivar headings (ex.: pi/2 - (-3*pi/2) deveria ser 2*pi, i.e. 0)."""
    return math.atan2(math.sin(a), math.cos(a))


def _amostrar_peca_local(c: dict, espacamento: float) -> list:
    """Amostra o EIXO CENTRAL de UMA peca em coordenadas LOCAIS.

    Recebe os conectores da peca (p_in/p_out/h_in/h_out) e devolve uma lista de
    (lx, ly, heading_local) ao longo do centro, com passo ~= `espacamento`.

    Como decide a forma SO pelos conectores (nada chumbado):
      - RETA  : h_in == h_out  -> segmento de reta de p_in ate p_out.
      - CURVA : h_in != h_out  -> arco de circulo. O angulo varrido pela marcha
        (dh = h_out - h_in) e' o MESMO angulo central do arco. Com a corda
        (distancia p_in->p_out) e dh, o raio sai de R = corda / (2*sin(dh/2)).
        O centro do arco fica a R, perpendicular ao heading de entrada (para a
        esquerda se dh>0, direita se dh<0). Isso funciona para curva a esquerda
        OU a direita (peca espelhada), sem hipotese sobre onde e' o centro.

    heading_local = direcao de MARCHA naquele ponto (tangente ao caminho)."""
    pin_x, pin_y = c["p_in"]
    pout_x, pout_y = c["p_out"]
    h_in, h_out = c["h_in"], c["h_out"]
    dh = _norm_ang(h_out - h_in)          # angulo total girado pela peca

    pts = []
    if abs(dh) < 1e-9:
        # ---- RETA: interpola em linha reta de p_in a p_out ----
        dist = math.hypot(pout_x - pin_x, pout_y - pin_y)
        n = max(1, round(dist / espacamento))
        for i in range(n + 1):
            t = i / n
            pts.append((pin_x + t * (pout_x - pin_x),
                        pin_y + t * (pout_y - pin_y),
                        h_in))
        return pts

    # ---- CURVA: arco de circulo ----
    corda = math.hypot(pout_x - pin_x, pout_y - pin_y)
    raio = corda / (2.0 * math.sin(abs(dh) / 2.0))
    giro = 1.0 if dh > 0 else -1.0        # +1 = esquerda (anti-horario), -1 = direita
    # Centro: a partir de p_in, ande `raio` na perpendicular ao heading de entrada.
    # normal a esquerda de um heading h e' (-sin h, cos h); a direita e' o oposto.
    cx = pin_x + giro * (-math.sin(h_in)) * raio
    cy = pin_y + giro * (math.cos(h_in)) * raio
    ang0 = math.atan2(pin_y - cy, pin_x - cx)   # angulo do ponto de entrada no circulo
    arco = raio * abs(dh)                        # comprimento do arco
    n = max(1, round(arco / espacamento))
    for i in range(n + 1):
        ang = ang0 + dh * (i / n)                # varre de ang0 por dh (com sinal)
        lx = cx + raio * math.cos(ang)
        ly = cy + raio * math.sin(ang)
        # tangente: perpendicular ao raio, no sentido da marcha (esq: +90, dir: -90)
        heading = ang + giro * (math.pi / 2.0)
        pts.append((lx, ly, _norm_ang(heading)))
    return pts


def gerar_waypoints(sequencia, fator: float = 1.0, espacamento: float = 0.10) -> list:
    """Gera a LINHA DE CENTRO (waypoints densos) de uma pista.

    Args:
        sequencia: lista de nomes de peca (ex.: ["tcc_reta", "tcc_curva90", ...])
                   OU o nome de um preset ("oval", "quadrado", "octogono", ...).
        fator:     escala da geometria — 1.0 (1/12, pecas tcc_*) ou 12.0 (real).
                   Precisa bater com a `escala` usada para montar a pista.
        espacamento: passo aproximado entre waypoints, na MESMA unidade da geometria
                   (metros ja escalados por `fator`).

    Returns:
        Lista de dicts {"x", "y", "heading" (rad), "s" (dist acumulada)}, no MESMO
        frame "matematico" em que as pecas sao montadas (ANTES de aplicar o
        flip_y/flip_yaw do CARLA — use `waypoints_para_carla` para converter).

    Os waypoints seguem a ordem da sequencia; para pistas fechadas o ultimo ponto
    coincide (aprox.) com o primeiro."""
    conectores = _conectores(fator)
    if isinstance(sequencia, str):
        sequencia = _preset(sequencia)
    poses, _final, _wps_entrada = _montar(sequencia, conectores, fechar_gap=True, fator=fator)

    pontos = []   # (x, y, heading) globais, ainda sem 's'
    for (nome, tx, ty, alpha) in poses:
        ca, sa = math.cos(alpha), math.sin(alpha)
        locais = _amostrar_peca_local(conectores[nome], espacamento)
        for j, (lx, ly, h_local) in enumerate(locais):
            # Emenda: o 1o ponto de cada peca (menos a 1a) repete o ultimo da
            # anterior -> pula, para nao duplicar waypoints coincidentes.
            if pontos and j == 0:
                continue
            gx = tx + ca * lx - sa * ly          # rotaciona por alpha e translada
            gy = ty + sa * lx + ca * ly
            pontos.append((gx, gy, _norm_ang(h_local + alpha)))

    # Acumula a distancia percorrida (s) ao longo da linha.
    waypoints = []
    s = 0.0
    for i, (x, y, h) in enumerate(pontos):
        if i > 0:
            s += math.hypot(x - pontos[i - 1][0], y - pontos[i - 1][1])
        waypoints.append({"x": x, "y": y, "heading": h, "s": s})
    return waypoints


def waypoints_para_carla(waypoints: list, flip_y: float = -1.0,
                         flip_yaw: float = -1.0, z: float = 0.05) -> list:
    """Converte os waypoints do frame 'matematico' para o frame do CARLA.

    Aplica EXATAMENTE a mesma calibracao de eixo que o `build_track` usa ao
    posicionar as pecas (flip_y no Y, flip_yaw no yaw) — assim os waypoints caem
    sobre a pista de verdade. Devolve `carla.Transform` prontos para o professor
    (pure-pursuit) mirar. Requer o CARLA instalado."""
    if carla is None:
        raise RuntimeError("carla nao disponivel: waypoints_para_carla precisa do CARLA")
    transforms = []
    for w in waypoints:
        loc = carla.Location(x=w["x"], y=flip_y * w["y"], z=z)
        rot = carla.Rotation(yaw=flip_yaw * math.degrees(w["heading"]))
        transforms.append(carla.Transform(loc, rot))
    return transforms


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

    # Escala: "meio" (1/12, pecas tcc_*) ou "real" (x12, pecas tcc_*_real).
    # O fator escala a GEOMETRIA (conectores + z + margem) e o sufixo escolhe as pecas.
    escala = str(track_config.get("escala", "meio")).lower()
    fator = 12.0 if escala == "real" else 1.0
    sufixo = "_real" if escala == "real" else ""
    z = z * fator
    logger.info("Track: escala '%s' (fator %g, sufixo '%s')" % (escala, fator, sufixo))

    # Palco quase vazio: descarrega as camadas do mapa (so ceu + luz).
    if track_config.get("unload_layers", True):
        world.unload_map_layer(carla.MapLayer.All)
        logger.info("Track: camadas do mapa descarregadas (palco vazio)")

    conectores = _conectores(fator)
    sequencia = _preset(preset)
    poses, (fx, fy, fh), waypoints = _montar(sequencia, conectores, fechar_gap=True, fator=fator)

    gap = math.hypot(fx, fy)
    ang = math.degrees(fh) % 360.0
    ang = min(ang, 360.0 - ang)
    logger.info(f"Track '{preset}': {len(sequencia)} pecas | fechamento gap={gap:.3f} m, ang={ang:.1f} deg")

    bl = world.get_blueprint_library()
    props = []
    xs, ys = [], []
    for i, (nome, tx, ty, alpha) in enumerate(poses):
        bp = bl.find("static.prop." + _prop_base(nome) + sufixo)
        # spawn num ponto de staging alto e teleporta (set_transform nao checa colisao de
        # emenda); como cada peca e movida ANTES da proxima, o ponto fica livre a cada vez.
        ator = world.try_spawn_actor(bp, carla.Transform(carla.Location(0.0, 0.0, 100.0)))
        if ator is None:
            logger.warning("Track: falhou spawn de %s" % (nome + sufixo))
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
    margem = float(obs_cfg.get("margem_lateral", 0.12)) * fator   # folga p/ o carro desviar
    lim = max(LARGURA_PISTA * fator / 2.0 - margem, 0.0)
    n_obs_ok = 0
    for _ in range(n_obs):
        if not waypoints:
            break
        wpx, wpy, wph = random.choice(waypoints)
        nome = random.choice(tipos)
        lat = random.uniform(-lim, lim)
        ox = wpx + lat * (-math.sin(wph))     # desloca perpendicular ao rumo da pista
        oy = wpy + lat * (math.cos(wph))
        bp = bl.find("static.prop." + _prop_base(nome) + sufixo)
        ator = world.try_spawn_actor(bp, carla.Transform(carla.Location(0.0, 0.0, 100.0)))
        if ator is None:
            logger.warning("Track: falhou spawn de obstaculo %s" % (nome + sufixo))
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

    medir_assets(world)   # inspeciona o que foi spawnado e imprime as medidas
    return props


def medir_assets(world, filtro="static.prop.tcc_*"):
    """Inspeciona os props JA presentes no mundo (nao spawna nada) e imprime as
    medidas de cada TIPO no terminal.

    Le o `bounding_box` de cada ator que casa com `filtro`, agrupa por `type_id`
    (todas as copias de um tipo tem a mesma medida) e loga as dimensoes TOTAIS
    (L x P x A, em metros = 2 x extent) + a contagem.

    Args:
        world: mundo CARLA.
        filtro: wildcard de blueprint (padrao: as pecas do TCC).

    Returns:
        dict {type_id: {"dim": (x, y, z), "n": quantidade}}.
    """
    atores = list(world.get_actors().filter(filtro))
    por_tipo = {}
    for a in atores:
        e = a.bounding_box.extent                      # meia-extensao (m), no frame local
        dim = (round(2.0 * e.x, 2), round(2.0 * e.y, 2), round(2.0 * e.z, 2))
        d = por_tipo.setdefault(a.type_id, {"dim": dim, "n": 0})
        d["n"] += 1

    logger.info("=== Medidas dos assets no mapa: %d atores, %d tipos (L x P x A, m) ===",
                len(atores), len(por_tipo))
    for tid in sorted(por_tipo):
        d = por_tipo[tid]
        logger.info("  %-34s x%-2d  %6.2f x %6.2f x %6.2f m",
                    tid, d["n"], d["dim"][0], d["dim"][1], d["dim"][2])
    return por_tipo
