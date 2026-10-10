"""Seguidor de linha do expert (2026-10-10): sinais no frame do CARLA e precisao."""
import math

import numpy as np
import pytest

from ai.seguidor import SeguidorDeLinha

WB, B = 2.875, 1.44


def _reta(n=200, passo=0.25):
    return [(i * passo, 0.0, 0.0) for i in range(n)]


def _circulo_direita(R=8.0, n=400):
    # Frame do CARLA: yaw cresce virando a DIREITA, de x para y.
    out = []
    for i in range(n):
        t = 2 * math.pi * i / n
        out.append((R * math.sin(t), R - R * math.cos(t), t))
    return out


def test_on_a_straight_a_car_to_the_right_steers_left():
    s = SeguidorDeLinha(_reta(), fechado=False)
    steer, _, _ = s.control(5.0, 0.5, 0.0, 1.7)       # y + = direita no CARLA
    assert steer < 0


def test_on_a_straight_a_car_pointing_left_steers_right():
    s = SeguidorDeLinha(_reta(), fechado=False)
    steer, _, _ = s.control(5.0, 0.0, -0.2, 1.7)
    assert steer > 0


def test_on_the_line_of_a_right_curve_it_steers_what_the_curve_needs():
    R = 8.0
    s = SeguidorDeLinha(_circulo_direita(R), fechado=True, previa_s=0.0)
    x, y, yaw = _circulo_direita(R)[10]
    beta = math.atan(B / math.sqrt(R * R - B * B))
    steer, _, _ = s.control(x, y, yaw - beta, 1.7)     # o centro anda torto (beta)
    esperado = math.atan(WB / math.sqrt(R * R - B * B)) / math.radians(27.0)
    assert steer == pytest.approx(esperado, abs=0.03)


def test_it_tracks_a_curve_far_better_than_its_margin():
    # Bicicleta pelo eixo traseiro, saida pelo centro, esterco com atraso.
    R = 8.0
    linha = _circulo_direita(R)
    s = SeguidorDeLinha(linha, fechado=True)
    x0, y0, yaw = linha[0]
    xr, yr, d = x0 - B, y0, 0.0
    erros = []
    for k in range(800):
        xc, yc = xr + B * math.cos(yaw), yr + B * math.sin(yaw)
        steer, _, _ = s.control(xc, yc, yaw, 1.7)
        d += (steer * math.radians(27.0) - d) * 0.5
        xr += 1.7 * math.cos(yaw) * 0.05
        yr += 1.7 * math.sin(yaw) * 0.05
        yaw += 1.7 * math.tan(d) / WB * 0.05
        if k > 100:
            erros.append(abs(math.hypot(xc, yc - R) - R))
    assert max(erros) < 0.1                          # < 1 cm no carro


def test_remaining_distance_and_the_open_end():
    s = SeguidorDeLinha(_reta(n=41, passo=0.25), fechado=False)
    assert s.restante() == pytest.approx(10.0)
    s.control(7.1, 0.0, 0.0, 1.7)
    assert s.restante() == pytest.approx(3.0, abs=0.26)
    assert SeguidorDeLinha(_circulo_direita(), fechado=True).restante() == math.inf
