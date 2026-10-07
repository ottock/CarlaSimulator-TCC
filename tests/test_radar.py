"""O radar desenha o mundo do lado certo?

Um radar espelhado e pior que nenhum: o operador encosta a parede pela direita,
ve o ponto aparecer na esquerda e conclui que o sensor esta errado. A convencao
e a do simulador (`ai.sim_lidar`): 0 = frente, POSITIVO = DIREITA.
"""
import numpy as np
import pytest

from ai.car import radar

cv2 = pytest.importorskip("cv2")


def _cor_perto(img, x, y, raio=3):
    """Ha algum pixel nao-fundo perto de (x, y)?"""
    janela = img[max(0, y - raio):y + raio + 1, max(0, x - raio):x + raio + 1]
    return bool((janela.astype(int).sum(axis=2) > 300).any())


def test_body_and_wall_are_split_by_the_sensor_arcs():
    pts = [(10.0, 0.2), (130.0, 0.5), (355.0, 0.3)]
    parede, corpo = radar.separa(pts, [(350.0, 120.0)])       # arco cruza o zero
    assert corpo == [(10.0, 0.2), (355.0, 0.3)]
    assert parede == [(130.0, 0.5)]


def test_the_calibrated_front_of_the_sensor_is_zero_in_the_car_frame():
    # Com o offset e a inversao medidos no carro, o sensor a 233.2 aponta a frente.
    [(a, d)] = radar.para_carro([(233.2, 0.3)], 126.8, True)
    assert radar.assinado(a) == pytest.approx(0.0, abs=1e-6)
    assert d == 0.3


@pytest.mark.parametrize("ang,esperado", [
    (0.0, "frente"), (25.0, "frente"), (90.0, "direita"), (-90.0, "esquerda"),
    (270.0, "esquerda"), (180.0, "tras"), (-170.0, "tras"),
])
def test_side_labels(ang, esperado):
    assert radar.lado(ang) == esperado


def test_nearest_point_or_none():
    assert radar.mais_proximo([]) is None
    assert radar.mais_proximo([(10, 0.5), (20, 0.2), (30, 0.3)]) == (20, 0.2)


def test_front_is_drawn_up_and_right_is_drawn_right():
    raio = 0.6
    escala = (min(radar.LARGURA, radar.ALTURA_RADAR) / 2.0 - 30.0) / raio
    cx, cy = radar.LARGURA // 2, radar.ALTURA_RADAR // 2

    img = radar.desenha([(0.0, 0.4)], [], None, None, [], 10.0, raio_m=raio)
    assert _cor_perto(img, cx, int(cy - 0.4 * escala)), "frente nao ficou para cima"

    img = radar.desenha([(90.0, 0.4)], [], None, None, [], 10.0, raio_m=raio)
    assert _cor_perto(img, int(cx + 0.4 * escala), cy), "direita nao ficou a direita"
    assert not _cor_perto(img, int(cx - 0.4 * escala), cy), "desenhou espelhado"


def test_empty_scan_does_not_crash():
    img = radar.desenha([], [], None, None, [], 0.0)
    assert img.shape == (radar.ALTURA_RADAR + radar.ALTURA_GRAFICO, radar.LARGURA, 3)
    assert img.dtype == np.uint8


def test_history_with_gaps_does_not_crash():
    hist = [(0.0, 0.3), (0.1, None), (0.2, 0.25), (0.3, 0.9)]
    img = radar.desenha([], [], None, 0.25, hist, 5.0)
    assert img.shape[0] == radar.ALTURA_RADAR + radar.ALTURA_GRAFICO
