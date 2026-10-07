"""Radar visto de cima: o que o LiDAR enxerga em volta do carro, para um humano.

Feito para ``hardware/testa_ponto_cego.py``, mas puro (so numpy + cv2), entao
testavel no PC sem sensor.

A separacao carroceria x parede e o motivo deste modulo existir. A primeira
versao do teste de ponto cego tomava o minimo sobre TODOS os pontos, e nas
rodadas gravadas 3.853 das 3.976 leituras abaixo de 0,215 m vinham do angulo
30-45 graus do sensor -- a propria carroceria, ja descrita em
``ai.car.lidar_frame``. O teste teria lido o carro a 0,202 m e anunciado
"ponto cego confirmado" nao importa o que o operador fizesse. Aqui o minimo e
SEMPRE so da parede, e a carroceria e desenhada a parte, para se ver.

Convencao de angulos (a mesma do simulador, ``ai.sim_lidar``): 0 = frente do
carro, POSITIVO = DIREITA. No desenho a frente fica para cima.
"""
import math

import numpy as np

from ai.car.lidar_frame import _in_arc, rotate_angles

# Cores BGR.
_FUNDO = (24, 24, 24)
_ANEL = (70, 70, 70)
_TEXTO = (230, 230, 230)
_PAREDE = (200, 200, 200)
_CORPO = (200, 90, 40)          # azul: e o proprio carro, nao conta
_ZONA_CORPO = (60, 30, 15)
_REF = (40, 40, 220)            # vermelho: a linha de 0,20 m sob suspeita
_PERTO = (60, 220, 60)          # verde: o ponto mais proximo da PAREDE

LARGURA = 640
ALTURA_RADAR = 640
ALTURA_GRAFICO = 200


def separa(pontos, arcos):
    """Divide ``(angulo_sensor, dist)`` em (parede, carroceria).

    Os arcos estao no frame do SENSOR, igual a ``drop_self_occlusion``: a
    carroceria nao se mexe em relacao ao sensor.
    """
    parede, corpo = [], []
    for a, d in pontos:
        if any(_in_arc(a, s, e) for s, e in arcos):
            corpo.append((a, d))
        else:
            parede.append((a, d))
    return parede, corpo


def para_carro(pontos, offset_deg, invert):
    """``(angulo_sensor, dist)`` -> ``(angulo_carro, dist)``, 0 = frente."""
    if not pontos:
        return []
    angs = rotate_angles([p[0] for p in pontos], offset_deg, invert)
    return [(a, p[1]) for a, p in zip(angs, pontos)]


def mais_proximo(pontos):
    """O ponto de menor distancia, ou ``None`` se nao ha pontos."""
    if not pontos:
        return None
    return min(pontos, key=lambda p: p[1])


def lado(angulo_carro):
    """Rotulo humano de um angulo no frame do carro (positivo = direita)."""
    a = ((angulo_carro + 180.0) % 360.0) - 180.0        # -180..180
    if -30.0 <= a <= 30.0:
        return "frente"
    if 30.0 < a < 150.0:
        return "direita"
    if -150.0 < a < -30.0:
        return "esquerda"
    return "tras"


def assinado(angulo_carro):
    """Angulo em -180..180, mais legivel que 0..360."""
    return ((angulo_carro + 180.0) % 360.0) - 180.0


def _xy(angulo_carro, dist, cx, cy, escala):
    """Frente para cima, direita para a direita."""
    t = math.radians(angulo_carro)
    return (int(round(cx + dist * math.sin(t) * escala)),
            int(round(cy - dist * math.cos(t) * escala)))


def desenha(parede, corpo, perto, menor_teste, historico, restante_s,
            arcos_sensor=(), offset_deg=0.0, invert=False,
            raio_m=0.6, ref_m=0.20, janela_s=15.0):
    """Monta o quadro: radar em cima, grafico do minimo no tempo embaixo.

    Args:
        parede / corpo: ``(angulo_carro, dist)``, ja separados e girados.
        perto: ``(angulo_carro, dist)`` mais proximo da parede, ou ``None``.
        menor_teste: menor distancia de parede no teste todo, ou ``None``.
        historico: lista de ``(t, dist ou None)``; ``None`` = nenhuma parede.
        arcos_sensor: arcos da carroceria, para sombrear a zona que nao conta.
    """
    import cv2

    img = np.zeros((ALTURA_RADAR + ALTURA_GRAFICO, LARGURA, 3), dtype=np.uint8)
    img[:] = _FUNDO
    cx, cy = LARGURA // 2, ALTURA_RADAR // 2
    escala = (min(LARGURA, ALTURA_RADAR) / 2.0 - 30.0) / float(raio_m)

    # Zona da carroceria: amostra o arco no frame do sensor e gira ponto a
    # ponto, em vez de girar so as pontas -- com invert=True o arco troca de
    # sentido e girar as pontas desenharia o complemento.
    for s, e in arcos_sensor:
        largura = (e - s) % 360.0
        angs = [s + largura * k / 60.0 for k in range(61)]
        angs_carro = rotate_angles(angs, offset_deg, invert)
        poly = [(cx, cy)] + [_xy(a, raio_m, cx, cy, escala) for a in angs_carro]
        cv2.fillPoly(img, [np.array(poly, dtype=np.int32)], _ZONA_CORPO)

    # Aneis de distancia.
    for r in (0.1, 0.2, 0.3, 0.4, 0.5):
        if r > raio_m:
            continue
        cv2.circle(img, (cx, cy), int(r * escala), _ANEL, 1, cv2.LINE_AA)
        cv2.putText(img, "%d cm" % int(round(r * 100)),
                    (cx + 4, cy - int(r * escala) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, _ANEL, 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), int(ref_m * escala), _REF, 1, cv2.LINE_AA)

    # Pontos.
    for a, d in corpo:
        if d <= raio_m:
            cv2.circle(img, _xy(a, d, cx, cy, escala), 2, _CORPO, -1)
    for a, d in parede:
        if d <= raio_m:
            cv2.circle(img, _xy(a, d, cx, cy, escala), 2, _PAREDE, -1)

    # O sensor e a frente.
    cv2.drawMarker(img, (cx, cy), _TEXTO, cv2.MARKER_TRIANGLE_UP, 14, 2)
    cv2.putText(img, "FRENTE", (cx - 30, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, _TEXTO, 1, cv2.LINE_AA)
    cv2.putText(img, "ESQ", (8, cy + 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, _TEXTO, 1, cv2.LINE_AA)
    cv2.putText(img, "DIR", (LARGURA - 40, cy + 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, _TEXTO, 1, cv2.LINE_AA)

    # O ponto mais proximo da parede.
    if perto is not None and perto[1] <= raio_m:
        p = _xy(perto[0], perto[1], cx, cy, escala)
        cv2.line(img, (cx, cy), p, _PERTO, 1, cv2.LINE_AA)
        cv2.circle(img, p, 7, _PERTO, 2, cv2.LINE_AA)

    # Texto do topo.
    agora = ("agora: %.3f m  %+.0f graus (%s)"
             % (perto[1], assinado(perto[0]), lado(perto[0]))
             if perto is not None else "agora: nenhuma parede")
    teste = ("menor do teste: %.3f m" % menor_teste
             if menor_teste is not None else "menor do teste: -")
    cv2.putText(img, agora, (10, ALTURA_RADAR - 44), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, _PERTO, 1, cv2.LINE_AA)
    cv2.putText(img, teste, (10, ALTURA_RADAR - 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, _TEXTO, 1, cv2.LINE_AA)
    cv2.putText(img, "faltam %.0f s" % max(0.0, restante_s), (LARGURA - 120, ALTURA_RADAR - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, _TEXTO, 1, cv2.LINE_AA)
    cv2.putText(img, "azul = carroceria (ignorada)", (LARGURA - 260, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _CORPO, 1, cv2.LINE_AA)
    cv2.putText(img, "vermelho = 20 cm", (LARGURA - 260, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _REF, 1, cv2.LINE_AA)

    _grafico(img, historico, ref_m, raio_m, janela_s)
    return img


def _grafico(img, historico, ref_m, raio_m, janela_s):
    """Minimo da parede ao longo do tempo. E aqui que um ponto cego aparece:
    aproximando a parede, a linha desce ate um valor e para -- ou SALTA para
    cima, quando a parede some e o proximo ponto visivel esta mais longe."""
    import cv2

    y0 = ALTURA_RADAR
    h = ALTURA_GRAFICO
    margem = 30
    cv2.line(img, (0, y0), (LARGURA, y0), _ANEL, 1)

    def y_de(d):
        d = min(max(d, 0.0), raio_m)
        return int(y0 + h - 12 - (h - 24) * d / raio_m)

    for r in (0.1, 0.2, 0.3, 0.4, 0.5):
        if r > raio_m:
            continue
        cor = _REF if abs(r - ref_m) < 1e-6 else _ANEL
        cv2.line(img, (margem, y_de(r)), (LARGURA - 5, y_de(r)), cor, 1)
        cv2.putText(img, "%d" % int(round(r * 100)), (4, y_de(r) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, cor, 1, cv2.LINE_AA)

    if not historico:
        return
    t_fim = historico[-1][0]
    anterior = None
    for t, d in historico:
        if t_fim - t > janela_s:
            anterior = None
            continue
        x = int(margem + (LARGURA - 5 - margem) * (1.0 - (t_fim - t) / janela_s))
        if d is None:
            anterior = None
            continue
        p = (x, y_de(d))
        if anterior is not None:
            cv2.line(img, anterior, p, _PERTO, 1, cv2.LINE_AA)
        anterior = p
    cv2.putText(img, "minimo da parede, ultimos %.0f s (cm)" % janela_s,
                (margem + 4, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _TEXTO, 1,
                cv2.LINE_AA)
