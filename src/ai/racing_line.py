"""Traçado viavel dentro do corredor da pista (Fase 6c).

O expert sempre seguiu o EIXO da pista, e o carro nunca conseguiu segui-lo: o
raio minimo MEDIDO do WLtoys e 0,625 m e o maior arco concentrico que cabe numa
faixa de 0,53 m tem 0,426 m. O modelo aprendeu com fidelidade uma trajetoria
inexecutavel -- e dai vieram o subesterco, o ganho de esterco, o tremor.

Aqui geramos a linha que um piloto faz: por fora nas retas, cortando o apice na
curva. Ela nao e refinamento de traçado; e a unica linha que existe para este
carro.

**Por que a linha do oval e um ESTADIO.** No `oval_tcc` as duas curvas de cada
ponta estao separadas por uma reta curta, mas o carro nao tem como endireitar
entre elas: o arco que toca o eixo das duas retas precisaria de 0,69 m de reta e
so ha 0,50 m. Entao as duas curvas de 90 graus viram UMA manobra de 180, e o
traçado inteiro e duas semicircunferencias ligadas por duas retas.

**Por que isto e analitico e nao otimizado.** Tentei antes um otimizador geral
(descida de gradiente projetada sobre o deslocamento lateral, multi-escala,
penalizando curvatura em excesso). Ele empaca em "colar na parede externa" --
raio 5,35 m onde o alvo e 7,50 -- porque sair dali exige mover um bloco inteiro
de pontos de forma coerente: para fora na reta e para DENTRO no apice. Nenhum
passo de gradiente local faz isso. Traçado otimo e problema de pesquisa; para um
oval a resposta tem forma fechada, e forma fechada e exata.

Matematica pura: nao importa ``carla``, roda nos testes sem simulador.
"""
import math

import numpy as np


def _xy(pontos):
    return np.array([(float(p[0]), float(p[1])) for p in pontos], dtype=float)


def _sem_ponto_repetido(pts):
    """Remove o ponto final que fecha o laco em cima do inicial.

    ``track_centerline`` devolve o laco com o primeiro ponto repetido no fim.
    Num calculo circular de curvatura isso vira um triangulo degenerado e a
    curvatura dos vizinhos explode -- o gradiente chegou a 1.5e5 por causa disso,
    e levou um bom tempo para aparecer.
    """
    if len(pts) > 1 and np.allclose(pts[0], pts[-1], atol=1e-9):
        return pts[:-1]
    return pts


def menger_curvature(p0, p1, p2):
    """Curvatura do circulo que passa pelos tres pontos: ``4A / (a*b*c)``.

    Independente de como os pontos estao espacados -- e e por isso que nao
    usamos a segunda diferenca simples, que mistura curvatura com espacamento e,
    num corredor circular, premiaria o lado de DENTRO (o caminho mais curto, e o
    de MAIOR curvatura). Tres pontos colineares ou coincidentes dao zero.
    """
    a = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    b = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    c = math.hypot(p2[0] - p0[0], p2[1] - p0[1])
    if a < 1e-12 or b < 1e-12 or c < 1e-12:
        return 0.0
    cruz = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
    return 2.0 * abs(cruz) / (a * b * c)


def path_curvatures(pts, fechado=True):
    """Curvatura em cada ponto, em 1/m (vetorizado)."""
    p = _xy(pts)
    if fechado:
        p = _sem_ponto_repetido(p)
    p0, p2 = np.roll(p, 1, axis=0), np.roll(p, -1, axis=0)
    a = np.hypot(*(p - p0).T)
    b = np.hypot(*(p2 - p).T)
    c = np.hypot(*(p2 - p0).T)
    cruz = ((p[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
            - (p[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0]))
    den = a * b * c
    den[den < 1e-12] = np.inf
    return 2.0 * np.abs(cruz) / den


def min_radius_m(pts, fechado=True):
    """Menor raio de curva do caminho, em metros (``inf`` se for reto)."""
    k = path_curvatures(pts, fechado=fechado).max()
    return float("inf") if k <= 1e-12 else 1.0 / float(k)


def oval_dims(centerline):
    """``(a, b, cx, cy)`` do eixo de um oval: semi-eixos maior e menor, e o centro.

    Vale para o ``oval_tcc``, cujo eixo e um retangulo arredondado alinhado aos
    eixos. Levanta se a pista nao for alongada o bastante para ter um eixo maior
    definido -- nesse caso o estadio nao e a figura certa.
    """
    p = _sem_ponto_repetido(_xy(centerline))
    cx, cy = (p[:, 0].max() + p[:, 0].min()) / 2, (p[:, 1].max() + p[:, 1].min()) / 2
    sx, sy = (p[:, 0].max() - p[:, 0].min()) / 2, (p[:, 1].max() - p[:, 1].min()) / 2
    a, b = max(sx, sy), min(sx, sy)
    if a - b < 1e-6:
        raise ValueError(
            "o eixo e quadrado (%.2f x %.2f m): nao da para dizer qual lado e o "
            "longo, e o estadio nao se aplica" % (2 * sx, 2 * sy))
    return a, b, cx, cy


def stadium_path(centerline, half_width, vehicle_width, n_points=240):
    """Traçado em estadio: duas semicircunferencias ligadas por duas retas.

    O maior estadio que cabe, que e tambem o de menor curvatura. Com ``a`` e
    ``b`` os semi-eixos do eixo da pista e ``c`` a folga util
    (``half_width - vehicle_width/2``):

        raio   R  = b + c        -- as retas encostam no limite do corredor
        centro xc = a - b        -- o ponto mais distante fica em ``a + c``

    O limite de ``xc`` sai de ``xc + R <= a + c``, que e exatamente ``xc <= a-b``.

    Args:
        centerline: eixo da pista, ``(x, y)`` ou ``(x, y, yaw)``.
        half_width: meia-largura da faixa util, em metros.
        vehicle_width: largura do veiculo. O caminho e o CENTRO dele, entao o
            corredor encolhe meia largura de cada lado. Use a largura do carro
            REAL escalada, nao a do veiculo do simulador: o traçado tem de ser
            executavel por quem vai dirigi-lo de verdade.

    Returns:
        ``np.ndarray`` ``(n_points, 3)`` com ``(x, y, yaw)``, fechado e no mesmo
        sentido de percurso do eixo recebido.

    Raises:
        ValueError: se o corredor util for vazio.
    """
    a, b, cx, cy = oval_dims(centerline)
    folga = float(half_width) - float(vehicle_width) / 2.0
    if folga <= 0.0:
        raise ValueError(
            "corredor util vazio: faixa de %.2f m nao comporta um veiculo de %.2f m"
            % (2 * float(half_width), float(vehicle_width)))

    R = b + folga
    xc = a - b

    p = _sem_ponto_repetido(_xy(centerline))
    eixo_x = (p[:, 0].max() - p[:, 0].min()) >= (p[:, 1].max() - p[:, 1].min())

    # Perimetro do estadio: duas retas de 2*xc + duas semicircunferencias.
    reta, arco = 2.0 * xc, math.pi * R
    total = 2.0 * reta + 2.0 * arco
    pontos = []
    for i in range(int(n_points)):
        s = total * i / float(n_points)
        if s < reta:                                   # reta de cima, +u
            u, v, h = -xc + s, R, 0.0
        elif s < reta + arco:                          # 180 na ponta +u
            t = (s - reta) / R
            u, v = xc + R * math.sin(t), R * math.cos(t)
            h = -t
        elif s < 2 * reta + arco:                      # reta de baixo, -u
            u, v, h = xc - (s - reta - arco), -R, math.pi
        else:                                          # 180 na ponta -u
            t = (s - 2 * reta - arco) / R
            u, v = -xc - R * math.sin(t), -R * math.cos(t)
            h = math.pi - t
        pontos.append((u, v, h))

    saida = []
    for u, v, h in pontos:
        if eixo_x:
            saida.append((cx + u, cy + v, h))
        else:                                          # oval deitado no eixo Y
            saida.append((cx - v, cy + u, h + math.pi / 2.0))
    arr = np.array(saida, dtype=float)

    # Mesmo sentido de percurso do eixo recebido: o Pure Pursuit anda para a
    # frente na lista, e um traçado invertido faria o carro dar re na pista.
    if _sentido(p) != _sentido(arr[:, :2]):
        arr = arr[::-1].copy()
        arr[:, 2] = (arr[:, 2] + math.pi) % (2 * math.pi)
    return arr


def _sentido(pts):
    """+1 se o laco e percorrido no sentido anti-horario, -1 se horario."""
    p = np.asarray(pts, dtype=float)
    x, y = p[:, 0], p[:, 1]
    area = float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    return 1 if area > 0 else -1


def distance_to_centerline(path, centerline):
    """Distancia de cada ponto do caminho a POLIGONAL do eixo, em metros.

    Ate o segmento, nao ate o vertice amostrado mais proximo. Com o eixo
    amostrado a cada 0,5 m, medir por vertice superestima em ~1 cm -- o
    bastante para reprovar um traçado que encosta de proposito no limite do
    corredor, que e o caso do estadio maximo.
    """
    c = _sem_ponto_repetido(_xy(centerline))
    p = _xy(path)
    a = c
    b = np.roll(c, -1, axis=0)
    ab = b - a
    comp2 = np.sum(ab * ab, axis=1)
    comp2[comp2 < 1e-18] = 1e-18
    out = np.empty(len(p))
    for i, q in enumerate(p):
        t = np.clip(np.sum((q - a) * ab, axis=1) / comp2, 0.0, 1.0)
        proj = a + t[:, None] * ab
        out[i] = float(np.min(np.hypot(proj[:, 0] - q[0], proj[:, 1] - q[1])))
    return out


def fits(path, centerline, half_width, vehicle_width, tol=1e-6):
    """A carroceria inteira fica dentro da faixa, em todo o traçado?

    Conferencia independente de como o caminho foi gerado -- um traçado que sai
    da pista e pior que nenhum, porque o expert gravaria o carro raspando.
    """
    lim = float(half_width) - float(vehicle_width) / 2.0
    return bool(distance_to_centerline(path, centerline).max() <= lim + tol)
