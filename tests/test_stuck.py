"""Deteccao de carro travado na coleta (Fase 6c).

Por que existe: duas coletas inteiras foram para o lixo em silencio. O carro
encravava num empurrao no ep1 e os onze episodios seguintes gravavam 1200
quadros cada de um carro IMOVEL -- com o expert mandando trava total, porque
ele continua tentando. O relatorio dizia "kept 1200, dropped 0" nos doze.

Quadros assim nao sao neutros, sao veneno: ensinam "esta imagem -> esterco
maximo" para uma imagem que nunca muda.
"""
import pytest

from ai.stuck import StuckDetector


def test_a_moving_car_is_never_stuck():
    d = StuckDetector(v_min=0.3, steps=10)
    for _ in range(50):
        assert d.update(1.74) is False


def test_it_fires_only_after_the_whole_window():
    d = StuckDetector(v_min=0.3, steps=5)
    for i in range(4):
        assert d.update(0.0) is False, "disparou cedo, no passo %d" % i
    assert d.update(0.0) is True


def test_one_moving_frame_resets_the_count():
    # Sem isto um carro que engasga e volta seria tratado como encravado.
    d = StuckDetector(v_min=0.3, steps=5)
    for _ in range(4):
        d.update(0.0)
    d.update(1.5)
    for i in range(4):
        assert d.update(0.0) is False, "nao zerou a contagem (passo %d)" % i
    assert d.update(0.0) is True


def test_it_does_not_fire_again_until_things_move():
    # Depois de disparar, o chamador vai tentar destravar. Disparar a cada passo
    # transformaria um encrave em uma enxurrada de reposicionamentos.
    d = StuckDetector(v_min=0.3, steps=3)
    for _ in range(3):
        d.update(0.0)
    assert d.update(0.0) is False     # ja avisou; espera o carro andar
    d.update(1.0)
    for _ in range(2):
        d.update(0.0)
    assert d.update(0.0) is True


def test_the_speed_threshold_is_respected():
    d = StuckDetector(v_min=0.5, steps=3)
    for _ in range(10):
        assert d.update(0.6) is False


def test_a_window_of_zero_is_refused():
    with pytest.raises(ValueError):
        StuckDetector(v_min=0.3, steps=0)
