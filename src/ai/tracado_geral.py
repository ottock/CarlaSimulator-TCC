"""Traçado viavel para QUALQUER pista da grade, nao so para o oval (2026-10-09).

O estadio (``racing_line.stadium_path``) e a resposta exata para um oval: duas
semicircunferencias do maior raio que cabe. Uma pista com "S", com fim, ou com o
quadrado 3 x 3 nao tem forma fechada assim. Aqui o mesmo criterio vira problema
de otimizacao: **o caminho de MENOR curvatura maxima dentro do corredor**. No
oval ele reencontra o estadio (raio, desvio do eixo e folga iguais -- e o teste
que ancora este modulo).

Como. O caminho e o eixo deslocado de ``a_i`` ao longo da normal de cada ponto.
A curvatura de tres pontos depende so de tres deslocamentos vizinhos, entao a
jacobiana e tridiagonal. Cada passo lineariza a curvatura e resolve um PROGRAMA
LINEAR (HiGHS, via scipy) com regiao de confianca:

    1. minimax:   min t   s.a.  |k + J d| <= t,  |a + d| <= limite
    2. suaviza:   com |k| <= t* + 3%, min sum|k| ds + sum|dk|

O passo 2 existe porque o minimax so enxerga o pior ponto: no resto da pista
qualquer caminho abaixo do pico empata, e o LP devolvia retas onduladas. Somar
|k| e |dk| favorece retas e arcos de curvatura constante -- a forma do estadio.

Por que programacao linear, e nao gradiente. Tentei antes, em 2026-10-09:
descida com Adam, L-BFGS com pontos de controle (B-spline), norma-p alta,
epigrafe com penalidade. Todos empacaram entre 89% e 134% do esterco no oval,
onde o estadio usa 74%: afastar um ponto da parede cria um bico, e o "maximo" e
nao-suave. O LP trata o maximo exatamente e chega a 73,5% em menos de 1 s.

A folga ate a parede e CONSTANTE, a varredura da carroceria no raio do proprio
traçado mais a margem -- exatamente como o estadio do v4, que tambem a aplica
nas retas. Conferido com o retangulo do carro contra as paredes nas 457
combinacoes da grade: folga minima igual a margem pedida.

Matematica pura: numpy + scipy, sem ``carla``.
"""
import math

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


def _curvaturas(p, fechado):
    """Curvatura com sinal (Menger) em cada ponto; pontas de caminho aberto = 0."""
    if fechado:
        a, b, c = np.roll(p, 1, 0), p, np.roll(p, -1, 0)
    else:
        a, b, c = p[:-2], p[1:-1], p[2:]
    la = np.hypot(*(b - a).T)
    lb = np.hypot(*(c - b).T)
    lc = np.hypot(*(c - a).T)
    cruz = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    k = 2.0 * cruz / (la * lb * lc + 1e-15)
    if not fechado:
        k = np.r_[0.0, k, 0.0]
    return k


def sobra(k, comprimento, largura):
    """Quanto a quina externa passa do centro, numa curva de curvatura ``k``.

    E o ``racing_line.swept_overhang`` escrito em curvatura, para nao dividir por
    zero na reta (onde vale ``largura / 2``).
    """
    k = np.abs(np.asarray(k, dtype=float))
    L, W = float(comprimento), float(largura)
    s = np.sqrt((1.0 + k * W / 2.0) ** 2 + (k * L / 2.0) ** 2)
    return (W + k * W * W / 4.0 + k * L * L / 4.0) / (s + 1.0)


def _reamostra(eixo, passo, fechado):
    """Eixo reamostrado a cada ``passo`` metros (interpolacao linear ao longo dele).

    A coleta entrega o eixo a cada 0,5 m (``professor.espacamento``). O traçado
    sai com a resolucao do eixo, e o Pure Pursuit segue esses pontos: medido em
    2026-10-09, o oval com 118 pontos aproveitou ~760 quadros por episodio na
    coleta, contra 1123 do estadio com 240 -- mesma geometria, mesma semente.
    """
    e = np.asarray(eixo, dtype=float)
    xy = e[:, :2]
    if fechado:
        if np.allclose(xy[0], xy[-1], atol=1e-6):
            xy = xy[:-1]
        xy = np.vstack([xy, xy[:1]])
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    n = max(int(round(s[-1] / float(passo))), 5)
    alvo = np.linspace(0.0, s[-1], n + 1)
    if fechado:
        alvo = alvo[:-1]
    x = np.interp(alvo, s, xy[:, 0])
    y = np.interp(alvo, s, xy[:, 1])
    if fechado:
        tg = np.stack([np.roll(x, -1) - np.roll(x, 1), np.roll(y, -1) - np.roll(y, 1)], 1)
    else:
        tg = np.stack([np.gradient(x), np.gradient(y)], 1)
    return np.c_[x, y, np.arctan2(tg[:, 1], tg[:, 0])]


class _Corredor:
    """Eixo + normais; o caminho e ``eixo + a * normal``."""

    def __init__(self, eixo, fechado):
        e = np.asarray(eixo, dtype=float)
        c, h = e[:, :2], e[:, 2]
        if fechado and len(c) > 1 and np.allclose(c[0], c[-1], atol=1e-6):
            c, h = c[:-1], h[:-1]
        self.c, self.n, self.fechado = c, len(c), bool(fechado)
        self.nrm = np.stack([-np.sin(h), np.cos(h)], 1)

    def pontos(self, a):
        return self.c + a[:, None] * self.nrm

    def kappa(self, a):
        return _curvaturas(self.pontos(a), self.fechado)

    def jacobiana(self, a, eps=1e-6):
        """d kappa / d a, tridiagonal (ciclica no laco), por diferencas centrais.

        ``kappa_j`` depende so de ``a_{j-1}, a_j, a_{j+1}``: perturbando de uma vez
        todos os ``a_i`` com o mesmo resto mod 3, cada ``kappa_j`` sente um so deles.
        No laco com ``n`` nao multiplo de 3 a emenda quebraria isso, entao os
        ultimos ``n % 3`` pontos vao sozinhos.
        """
        n = self.n
        corte = n if (not self.fechado or n % 3 == 0) else n - n % 3
        grupos = [np.arange(r, corte, 3) for r in range(3)]
        grupos += [np.array([i]) for i in range(corte, n)]
        lin, col, val = [], [], []
        for g in grupos:
            if not len(g):
                continue
            d = np.zeros(n)
            d[g] = eps
            dk = (self.kappa(a + d) - self.kappa(a - d)) / (2.0 * eps)
            for i in g:
                for j in (i - 1, i, i + 1):
                    if self.fechado:
                        j %= n
                    elif j < 0 or j >= n:
                        continue
                    lin.append(j)
                    col.append(i)
                    val.append(dk[j])
        return sparse.csr_matrix((val, (lin, col)), shape=(n, n))


def _limites_do_passo(cor, a, lim, delta, largada_no_eixo):
    hi = np.minimum(lim - a, delta)
    lo = np.maximum(-lim - a, -delta)
    lo = np.minimum(lo, hi)
    if largada_no_eixo:
        # O carro larga parado no meio da faixa, apontado ao longo dela.
        lo[:2] = -a[:2]
        hi[:2] = -a[:2]
    return lo, hi


def _minimax(cor, folga_fixa, largura_util, comp, larg, kref, delta, largada_no_eixo, iters=80):
    n = cor.n
    a = np.zeros(n)

    def limite(t):
        return max(largura_util - folga_fixa - float(sobra(min(t, 1.5) * kref, comp, larg)),
                   0.01 * largura_util)

    def merito(a):
        k = cor.kappa(a)
        t = float(np.abs(k).max()) / kref
        viol = max(0.0, float(np.max(np.abs(a))) - limite(t))
        return t + 100.0 * viol / largura_util, k

    m_atual, k = merito(a)
    delta_max = 0.6 * largura_util
    for _ in range(iters):
        t = float(np.abs(k).max()) / kref
        J = cor.jacobiana(a)
        lo, hi = _limites_do_passo(cor, a, limite(t), delta, largada_no_eixo)
        col_t = sparse.csr_matrix(np.full((n, 1), -kref))
        A = sparse.vstack([sparse.hstack([J, col_t]), sparse.hstack([-J, col_t])]).tocsr()
        res = linprog(np.r_[np.zeros(n), 1.0], A_ub=A, b_ub=np.r_[-k, k],
                      bounds=list(zip(lo, hi)) + [(0, None)], method="highs")
        if res.status == 0:
            m_novo, k_novo = merito(a + res.x[:n])
            if m_novo < m_atual - 1e-9:
                a, k, m_atual = a + res.x[:n], k_novo, m_novo
                delta = min(delta * 1.5, delta_max)
            else:
                delta *= 0.5
        else:
            delta *= 0.5
        if delta < 1e-5 * largura_util:
            break
    return a, k


def _suaviza(cor, a, teto, lim, kref, delta, largada_no_eixo, lam_tv=1.0, iters=60):
    n = cor.n
    p = cor.pontos(a)
    ds = np.hypot(*(np.roll(p, -1, 0) - p).T)
    I_n = sparse.eye(n, format="csr")
    if cor.fechado:
        D = sparse.eye(n, k=1, format="csr") + sparse.csr_matrix(
            ([1.0], ([n - 1], [0])), shape=(n, n)) - I_n
    else:
        D = (sparse.eye(n, k=1, format="csr") - I_n)[:-1]
    m = D.shape[0]

    def custo(a):
        k = cor.kappa(a)
        viol = max(0.0, float(np.max(np.abs(a))) - lim)
        exc = max(0.0, float(np.abs(k).max()) / kref - teto)
        return ((np.sum(np.abs(k) * ds) + lam_tv * np.sum(np.abs(D @ k))) / kref
                + 1e3 * (viol / lim + exc)), k

    c_atual, k = custo(a)
    I_m = sparse.eye(m, format="csr")
    Znm, Zmn = sparse.csr_matrix((n, m)), sparse.csr_matrix((m, n))
    delta_max = 2.0 * delta
    for _ in range(iters):
        J = cor.jacobiana(a)
        lo, hi = _limites_do_passo(cor, a, lim, delta, largada_no_eixo)
        DJ, Dk = (D @ J).tocsr(), D @ k
        A = sparse.vstack([sparse.hstack([J, -I_n, Znm]), sparse.hstack([-J, -I_n, Znm]),
                           sparse.hstack([DJ, Zmn, -I_m]), sparse.hstack([-DJ, Zmn, -I_m])]).tocsr()
        res = linprog(np.r_[np.zeros(n), ds / kref, np.full(m, lam_tv / kref)], A_ub=A,
                      b_ub=np.r_[-k, k, -Dk, Dk],
                      bounds=list(zip(lo, hi)) + [(0, teto * kref)] * n + [(0, None)] * m,
                      method="highs")
        if res.status == 0:
            c_novo, k_novo = custo(a + res.x[:n])
            if c_novo < c_atual - 1e-9:
                a, k, c_atual = a + res.x[:n], k_novo, c_novo
                delta = min(delta * 1.5, delta_max)
            else:
                delta *= 0.5
        else:
            delta *= 0.5
        if delta < 1e-5 * lim:
            break
    return a, k


def tracado_geral(eixo, fechado, half_width, vehicle_width, vehicle_length, margin,
                  raio_ref, folga_curvatura=0.03, largada_no_eixo=None, passo=None):
    """Caminho de menor curvatura maxima dentro do corredor.

    Args:
        eixo: linha de centro, ``(x, y, yaw)`` -- de ``track_ref.track_centerline``.
        fechado: laco (True) ou pista com comeco e fim (False).
        half_width: meia-largura da faixa util.
        vehicle_width, vehicle_length: o carro REAL escalado, como no estadio.
        margin: folga extra ate a parede (a mesma ``--margem`` do estadio).
        raio_ref: raio minimo do carro, na escala do eixo. So normaliza a
            curvatura (o resultado fica em "fracao do esterco"); quem RECUSA um
            caminho inexecutavel continua sendo ``racing_line.expert_path``.
        folga_curvatura: quanto a suavizacao pode subir o pico, em fracao do
            esterco (0.03 = 3 pontos percentuais).
        largada_no_eixo: prende os dois primeiros pontos no eixo. Padrao: so em
            pista aberta, onde o carro larga parado no meio da faixa.
        passo: espacamento do traçado, em metros. Padrao ``half_width / 12``
            (0,265 m no gemeo, a resolucao do estadio). O eixo e reamostrado.

    Returns:
        ``np.ndarray (n, 3)`` com ``(x, y, yaw)``, no sentido do eixo recebido,
        com ``n`` dado pelo ``passo`` (nao pelo eixo recebido).
    """
    if largada_no_eixo is None:
        largada_no_eixo = not fechado
    hw, comp, larg = float(half_width), float(vehicle_length), float(vehicle_width)
    if hw - margin - larg / 2.0 <= 0.0:
        raise ValueError("corredor util vazio: faixa de %.2f m, carro de %.2f m, margem %.2f m"
                         % (2 * hw, larg, margin))
    kref = 1.0 / float(raio_ref)
    eixo = _reamostra(eixo, float(passo) if passo else hw / 12.0, fechado)
    cor = _Corredor(eixo, fechado)
    if cor.n < 5:
        raise ValueError("eixo com %d pontos: curto demais para otimizar" % cor.n)
    delta0 = 0.15 * hw
    a, k = _minimax(cor, float(margin), hw, comp, larg, kref, delta0, largada_no_eixo)
    teto = float(np.abs(k).max()) / kref + float(folga_curvatura)
    lim = hw - float(margin) - float(sobra(teto * kref, comp, larg))
    a, k = _suaviza(cor, a, teto, lim, kref, delta0, largada_no_eixo)
    p = cor.pontos(a)
    if fechado:
        tg = np.roll(p, -1, 0) - np.roll(p, 1, 0)
    else:
        tg = np.gradient(p, axis=0)
    yaw = np.arctan2(tg[:, 1], tg[:, 0])
    return np.c_[p, yaw]


def folga_exata(caminho, eixo, fechado, half_width, vehicle_width, vehicle_length,
                passo=None, ignorar_fim=0.0):
    """Folga ate a parede de cada ponto do caminho, com o retangulo do carro.

    Conferencia independente do modelo de folga usado na otimizacao: o perimetro
    do carro (centrado no ponto, alinhado a tangente) contra o corredor, medindo
    o deslocamento lateral de cada ponto do perimetro em relacao ao eixo.

    ``ignorar_fim`` (metros de caminho) pula o trecho final de uma pista aberta:
    o carro para antes da parede do fim, e perto dela o eixo do COMECO pode estar
    ao lado -- a medida lateral ali compara com a pista errada.

    Returns:
        ``np.ndarray`` com a folga de cada ponto (``inf`` nos ignorados).
    """
    L, W, hw = float(vehicle_length), float(vehicle_width), float(half_width)
    passo = passo or W / 8.0
    xs = np.arange(-L / 2, L / 2 + 1e-9, passo)
    ys = np.arange(-W / 2, W / 2 + 1e-9, passo)
    perim = np.vstack([np.c_[xs, np.full_like(xs, W / 2)], np.c_[xs, np.full_like(xs, -W / 2)],
                       np.c_[np.full_like(ys, -L / 2), ys], np.c_[np.full_like(ys, L / 2), ys]])
    e = np.asarray(eixo, dtype=float)[:, :2]
    if fechado and np.allclose(e[0], e[-1], atol=1e-6):
        e = e[:-1]
    a, b = (e, np.roll(e, -1, 0)) if fechado else (e[:-1], e[1:])
    ab = b - a
    l2 = np.maximum((ab ** 2).sum(1), 1e-18)
    P = np.asarray(caminho, dtype=float)
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P[:, :2], axis=0).T))]
    out = np.full(len(P), np.inf)
    for i, (x, y, h) in enumerate(P):
        if not fechado and s[-1] - s[i] < ignorar_fim:
            continue
        c, sn = math.cos(h), math.sin(h)
        q = np.array([x, y]) + perim @ np.array([[c, sn], [-sn, c]])
        t = ((q[:, None, :] - a[None]) * ab[None]).sum(2) / l2[None]
        proj = a[None] + np.clip(t, 0, 1)[..., None] * ab[None]
        d2 = ((q[:, None, :] - proj) ** 2).sum(2)
        j = d2.argmin(1)
        lat = np.sqrt(d2[np.arange(len(q)), j])
        if not fechado:
            # Antes do comeco ou depois do fim nao ha parede lateral para medir: a
            # distancia ate a ponta do eixo seria longitudinal, e acusaria um
            # carro que so esta largando.
            tj = t[np.arange(len(q)), j]
            fora = ((j == 0) & (tj < 0)) | ((j == len(a) - 1) & (tj > 1))
            lat = lat[~fora]
            if not len(lat):
                continue
        out[i] = hw - float(lat.max())
    return out
