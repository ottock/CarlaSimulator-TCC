"""Frame do LiDAR: auto-oclusao e alinhamento com a frente do carro (Fase 6b).

Duas correcoes que o log do carro exigiu, ambas matematica pura:

1. A carroceria oclui um arco fixo do sensor. Medido no `runs/` de 2026-09-12:
   ~0 a 120 graus lendo 0.20-0.32 m com spread <= 0.04 (perto E constante), mais
   um ponto encostado em ~212 graus. Sem mascarar, o modelo ve uma parede
   permanente a 25 cm em um terco do circulo -- entrada que ele nunca viu.
2. O zero do sensor nao aponta necessariamente para a frente do carro, e o
   sentido de rotacao pode ser o oposto do simulador.
"""
import pytest

from ai.car.lidar_frame import drop_self_occlusion, rotate_angles


def test_drop_removes_returns_inside_the_arc():
    ang = [10.0, 50.0, 200.0, 300.0]
    dist = [0.2, 0.25, 5.0, 3.0]
    a, d = drop_self_occlusion(ang, dist, [(0.0, 120.0)])
    assert a == [200.0, 300.0]
    assert d == [5.0, 3.0]


def test_drop_keeps_everything_when_no_arc_is_given():
    ang, dist = [10.0, 200.0], [0.2, 5.0]
    assert drop_self_occlusion(ang, dist, []) == (ang, dist)


def test_drop_handles_an_arc_that_crosses_zero():
    # Um arco 350->10 passa pelo zero; tratar como intervalo comum apagaria
    # o circulo inteiro em vez de 20 graus.
    ang = [355.0, 5.0, 180.0]
    dist = [0.2, 0.2, 4.0]
    a, d = drop_self_occlusion(ang, dist, [(350.0, 10.0)])
    assert a == [180.0] and d == [4.0]


def test_drop_accepts_several_arcs():
    ang = [10.0, 180.0, 212.0, 300.0]
    dist = [0.2, 4.0, 0.002, 3.0]
    a, _ = drop_self_occlusion(ang, dist, [(0.0, 120.0), (210.0, 215.0)])
    assert a == [180.0, 300.0]


def test_rotate_shifts_into_the_car_frame():
    # offset = angulo do sensor que aponta para a FRENTE do carro; depois da
    # rotacao esse angulo vira 0.
    assert rotate_angles([240.0], offset_deg=240.0) == [0.0]
    assert rotate_angles([250.0], offset_deg=240.0) == [10.0]


def test_rotate_wraps_around_instead_of_going_negative():
    assert rotate_angles([230.0], offset_deg=240.0) == [350.0]


def test_rotate_with_no_offset_is_identity():
    assert rotate_angles([0.0, 90.0, 359.0], offset_deg=0.0) == [0.0, 90.0, 359.0]


def test_invert_mirrors_the_direction_of_rotation():
    # Se o sensor gira ao contrario do simulador, esquerda e direita trocam --
    # e o carro esterca com conviccao para o lado errado.
    assert rotate_angles([90.0], offset_deg=0.0, invert=True) == [270.0]
    assert rotate_angles([270.0], offset_deg=0.0, invert=True) == [90.0]


def test_invert_is_applied_before_the_offset():
    # Ordem fixada de proposito: espelha no frame do SENSOR, depois gira para o
    # frame do carro. A ordem oposta daria outro resultado.
    assert rotate_angles([90.0], offset_deg=30.0, invert=True) == pytest.approx([240.0])


def test_parse_arcs_reads_the_command_line_format():
    from ai.car.lidar_frame import parse_arcs
    assert parse_arcs("0:120,210:215") == [(0.0, 120.0), (210.0, 215.0)]


def test_parse_arcs_accepts_empty_as_no_masking():
    from ai.car.lidar_frame import parse_arcs
    assert parse_arcs("") == []
    assert parse_arcs(None) == []


def test_parse_arcs_refuses_garbage_instead_of_silently_masking_nothing():
    # Um formato errado aceito em silencio desligaria a mascara sem avisar, e a
    # carroceria voltaria a entrar no vetor sem ninguem perceber.
    from ai.car.lidar_frame import parse_arcs
    with pytest.raises(ValueError):
        parse_arcs("0-120")
    with pytest.raises(ValueError):
        parse_arcs("frente:120")
