"""Referencia da pista custom: linha de centro + desvio lateral (puro, sem CARLA).

A pista custom sao props num palco vazio — o CARLA NAO tem OpenDRIVE dela, entao
`map.get_waypoint()` nao funciona. Aqui derivamos a linha de centro da MESMA
geometria que monta a pista (`track_builder.gerar_waypoints`), aplicamos o mesmo
flip de eixo que o `build_track` usa ao posicionar as pecas, e medimos o desvio
lateral perpendicular a ela. Usado na coleta (recuperacao) e no loop fechado.

Tudo aqui e' matematica pura (nao importa `carla`), entao e' testavel offline.
"""
import math

from core.carlaClient.track_builder import gerar_waypoints, LARGURA_PISTA


def _fator(track_cfg):
    return 12.0 if str(track_cfg.get("escala", "meio")).lower() == "real" else 1.0


def track_centerline(track_cfg):
    """Linha de centro como lista de ``(x, y, yaw_rad)`` no frame do CARLA.

    Reusa `gerar_waypoints` (geometria pura) e aplica o mesmo `flip_y`/`flip_yaw`
    que o `build_track` aplica as pecas — assim os pontos caem sobre a pista real.
    """
    preset = track_cfg.get("preset", "oval")
    flip_y = float(track_cfg.get("flip_y", -1.0))
    flip_yaw = float(track_cfg.get("flip_yaw", -1.0))
    espac = float(track_cfg.get("professor", {}).get("espacamento", 0.5))
    wps = gerar_waypoints(preset, _fator(track_cfg), espac)
    return [(w["x"], flip_y * w["y"], flip_yaw * w["heading"]) for w in wps]


def track_width(track_cfg):
    """Largura util da pista (m), na mesma escala da geometria."""
    return LARGURA_PISTA * _fator(track_cfg)


def _nearest_index(centerline, x, y, start_idx=0, window=0):
    """Indice do vertice mais proximo de (x, y).

    Com ``window > 0`` busca so ``window`` pontos a frente de ``start_idx`` (O(window),
    como o `PurePursuit._nearest_ahead` — evita 'voltar' quando a pista se aproxima de
    si mesma); senao faz a busca completa O(n)."""
    n = len(centerline)
    if window and window > 0:
        candidatos = ((start_idx + o) % n for o in range(window))
    else:
        candidatos = range(n)
    melhor_i, melhor_d2 = start_idx % n, float("inf")
    for i in candidatos:
        dx = centerline[i][0] - x
        dy = centerline[i][1] - y
        d2 = dx * dx + dy * dy
        if d2 < melhor_d2:
            melhor_d2, melhor_i = d2, i
    return melhor_i


def deviation_from_centerline(centerline, x, y, start_idx=0, window=0):
    """Desvio lateral perpendicular a linha de centro + indice do vertice mais proximo.

    Retorna ``(dev_m, idx)``. O desvio e' a componente PERPENDICULAR ao heading do
    vertice mais proximo (remove a parte 'ao longo da pista'), mais preciso que a
    distancia ao vertice cru. `departed = dev > track_width/2`.
    """
    idx = _nearest_index(centerline, x, y, start_idx, window)
    wx, wy, wyaw = centerline[idx]
    dx, dy = x - wx, y - wy
    lateral = -math.sin(wyaw) * dx + math.cos(wyaw) * dy   # componente 'a direita' do heading
    return abs(lateral), idx
