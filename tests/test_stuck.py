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


# ---------------------------------------------------------------------------
# Janela depois de um toque (2026-10-09)
# ---------------------------------------------------------------------------
from ai.stuck import JanelaDeColisao  # noqa: E402


def test_frames_right_after_a_touch_are_dropped():
    j = JanelaDeColisao(20)
    j.atualiza(3, passo=100)
    assert j.batendo(100) and j.batendo(119)
    assert not j.batendo(120)
    assert j.toques == 3


def test_a_touch_late_in_one_episode_does_not_drop_the_next_one():
    # O bug: passo 1100 no episodio anterior, e o novo comeca no passo 0.
    j = JanelaDeColisao(20)
    j.atualiza(5, passo=1100)
    j.reinicia(n_eventos=5)
    assert not j.batendo(0)
    assert j.toques == 0


def test_a_new_sensor_starts_counting_from_zero_again():
    # Pista nova = sensor novo, lista de eventos vazia: a contagem velha (842)
    # impedia de registrar qualquer toque.
    j = JanelaDeColisao(20)
    j.atualiza(842, passo=50)
    j.reinicia(n_eventos=0)
    j.atualiza(1, passo=7)
    assert j.batendo(8) and j.toques == 1
