"""Seguidor de linha para o expert: antecipa a curvatura e corrige o erro (2026-10-10).

Por que existe. O Pure Pursuit mira um ponto a frente e, por construcao, corta as
curvas: medido no gemeo calibrado, ele se afastou 4,5-7 cm (na escala do carro)
da propria linha, e a linha tem 3 cm de folga ate a parede. O modelo aprende o
que o expert FAZ, nao o que a linha diz -- e herdava uma folga de ~1 cm nas
pistas com "U" logo na largada.

Como. A linha e conhecida, entao o esterco que a gera tambem e:

    esterco = previsto(curvatura logo a frente)            # o que a linha pede
            + atan(k_e * erro_lateral / (v + v0))           # volta para a linha
            + k_psi * erro_de_rumo                          # alinha com ela

O ponto de referencia e o CENTRO do carro (o do CARLA, e o que o traçado usa),
nao o eixo traseiro da bicicleta: numa curva de raio R no centro, o eixo
traseiro anda em ``sqrt(R^2 - b^2)``, e o centro anda torto (deriva ``beta``).

Frame do CARLA: ``yaw`` cresce virando a DIREITA, e esterco +1 = direita. A
curvatura com sinal (Menger) nesse frame e positiva numa curva a direita, entao
os sinais casam sem trocar nada.

Mesma interface do ``professor.PurePursuit`` (``control``, ``idx``, ``restante``,
``fechado``): a coleta troca um pelo outro. Python puro + numpy, sem ``carla``.
"""
import math

import numpy as np


def _curvatura_com_sinal(xy, fechado):
    a = np.roll(xy, 1, 0)
    c = np.roll(xy, -1, 0)
    la = np.hypot(*(xy - a).T)
    lb = np.hypot(*(c - xy).T)
    lc = np.hypot(*(c - a).T)
    cruz = (xy[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (xy[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    k = 2.0 * cruz / (la * lb * lc + 1e-12)
    if not fechado:
        k[0], k[-1] = k[1], k[-2]
    return k


def _ang(a):
    return math.atan2(math.sin(a), math.cos(a))


class SeguidorDeLinha:
    def __init__(self, waypoints, wheelbase=2.875, max_steer_deg=27.0, target_speed=2.0,
                 k_throttle=0.5, fechado=True, k_e=1.5, k_psi=1.0, v0=0.5,
                 previa_s=0.25, centro_ao_eixo=None, janela=120):
        """Args:
            waypoints: a linha, ``(x, y, yaw)`` no frame do CARLA (a do traçado).
            wheelbase: entre-eixos (m).
            max_steer_deg: angulo EFETIVO que ``steer = 1`` produz (normalizacao).
            k_e, k_psi, v0: ganhos da correcao lateral e de rumo.
            previa_s: quanto tempo a frente a curvatura e lida, para compensar o
                atraso entre comandar e a roda chegar la.
            centro_ao_eixo: distancia do centro do carro ao eixo traseiro;
                padrao metade do entre-eixos.
        """
        p = np.asarray([(float(w[0]), float(w[1])) for w in waypoints], dtype=float)
        if fechado and len(p) > 2 and np.allclose(p[0], p[-1], atol=1e-6):
            p = p[:-1]
        self.p = p
        self.n = len(p)
        self.fechado = bool(fechado)
        self.L = float(wheelbase)
        self.b = float(centro_ao_eixo) if centro_ao_eixo is not None else self.L / 2.0
        self.max_steer = math.radians(float(max_steer_deg))
        self.target_speed = float(target_speed)
        self.k = float(k_throttle)
        self.k_e, self.k_psi, self.v0 = float(k_e), float(k_psi), float(v0)
        self.previa_s = float(previa_s)
        self.janela = int(janela)
        self.kappa = _curvatura_com_sinal(p, self.fechado)
        seg = np.roll(p, -1, 0) - p
        if not self.fechado:
            seg[-1] = seg[-2]
        self.seg = seg
        self.seg_len = np.maximum(np.hypot(*seg.T), 1e-9)
        self.rumo = np.arctan2(seg[:, 1], seg[:, 0])
        self.idx = 0
        self.laps = 0
        # compatibilidade com o PurePursuit (a coleta le .wps no _destravar)
        self.wps = [(float(x), float(y), float(h)) for (x, y), h in zip(p, self.rumo)]

    # -- geometria --------------------------------------------------------
    def _projeta(self, x, y):
        """(indice do segmento, fracao, erro lateral com sinal: + = carro a DIREITA)."""
        melhor = (self.idx, 0.0, 0.0, 1e18)
        q = np.array([x, y])
        for off in range(self.janela):
            i = self.idx + off
            if self.fechado:
                i %= self.n
            elif i >= self.n - 1:
                break
            a, d = self.p[i], self.seg[i]
            t = float(np.clip(np.dot(q - a, d) / (self.seg_len[i] ** 2), 0.0, 1.0))
            r = q - (a + t * d)
            d2 = float(r @ r)
            if d2 < melhor[3]:
                # lado: + quando o carro esta a direita do rumo da linha (frame CARLA)
                lado = (d[0] * r[1] - d[1] * r[0]) / self.seg_len[i]
                melhor = (i, t, lado, d2)
        i, t, lado, _ = melhor
        if self.fechado and i < self.idx and (self.idx - i) > self.n // 2:
            self.laps += 1
        self.idx = i
        return i, t, lado

    def _curvatura_a_frente(self, i, t, dist):
        s, j = (1.0 - t) * self.seg_len[i], i
        while s < dist:
            j = j + 1
            if self.fechado:
                j %= self.n
            elif j >= self.n - 1:
                j = self.n - 1
                break
            s += self.seg_len[j]
        return float(self.kappa[j])

    def restante(self):
        if self.fechado:
            return float("inf")
        return float(self.seg_len[self.idx:self.n - 1].sum())

    # -- controle ---------------------------------------------------------
    def esterco_previsto(self, kappa_centro):
        """Angulo de roda que poe o CENTRO do carro numa curva de curvatura dada."""
        if abs(kappa_centro) < 1e-6:
            return 0.0
        R = 1.0 / abs(kappa_centro)
        r_eixo = math.sqrt(max(R * R - self.b * self.b, 1e-6))
        return math.copysign(math.atan(self.L / r_eixo), kappa_centro)

    def control(self, x, y, yaw, speed):
        i, t, lado = self._projeta(x, y)
        kappa = self._curvatura_a_frente(i, t, max(0.5, speed * self.previa_s))
        d_ff = self.esterco_previsto(kappa)
        # O centro anda torto em curva: a direcao do MOVIMENTO e yaw + beta.
        beta = math.atan(self.b * math.tan(d_ff) / self.L)
        e_rumo = _ang(float(self.rumo[i]) - (yaw + beta))
        delta = (d_ff + math.atan2(-self.k_e * lado, speed + self.v0) + self.k_psi * e_rumo)
        steer = max(-1.0, min(1.0, delta / self.max_steer))
        err = self.target_speed - speed
        throttle, brake = 0.0, 0.0
        if err >= 0:
            throttle = min(1.0, self.k * err)
        elif err < -0.5:
            brake = min(1.0, -self.k * err)
        return steer, throttle, brake
