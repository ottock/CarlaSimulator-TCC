"""Mapeamento da saida do modelo para o servo do carro (Fase 6b).

O modelo devolve steer em [-1, 1]; o servo fala microssegundos. Este mapa e o
ultimo ponto antes do hardware, entao ele tem de ser seguro mesmo recebendo lixo:
um NaN ou um valor fora de faixa NAO pode virar um comando extremo no servo.
"""

from ai.car.control_map import (
    ESC_NEUTRAL_US,
    STEER_CENTER_US,
    STEER_LEFT_US,
    STEER_RIGHT_US,
    FrameWatchdog,
    steer_to_us,
)


def test_zero_steer_is_centre():
    assert steer_to_us(0.0) == 1500


def test_full_right_matches_carla_convention():
    # CARLA: steer +1 = direita. NESTE carro o servo e espelhado (STEER_SIGN),
    # entao a direita sai em 1200 us -- medido, nao deduzido.
    assert steer_to_us(1.0) == STEER_RIGHT_US == 1200


def test_full_left():
    assert steer_to_us(-1.0) == STEER_LEFT_US == 1800


def test_half_right_is_linear():
    assert steer_to_us(0.5) == 1350


def test_out_of_range_is_clamped_not_wrapped():
    # A cabeca de steer usa tanh, entao nao deveria estourar -- mas se estourar,
    # o servo nao pode receber um comando alem do batente mecanico.
    assert steer_to_us(5.0) == STEER_RIGHT_US
    assert steer_to_us(-5.0) == STEER_LEFT_US


def test_nan_is_centre():
    assert steer_to_us(float("nan")) == STEER_CENTER_US


def test_none_is_centre():
    assert steer_to_us(None) == STEER_CENTER_US


def test_positive_infinity_is_hard_right():
    # Infinity should not crash; +inf clamped to 1.0 means hard right.
    assert steer_to_us(float("inf")) == STEER_RIGHT_US


def test_negative_infinity_is_hard_left():
    # Infinity should not crash; -inf clamped to -1.0 means hard left.
    assert steer_to_us(float("-inf")) == STEER_LEFT_US


def test_esc_neutral_constant():
    assert ESC_NEUTRAL_US == 1500


def test_watchdog_does_not_fire_on_the_first_tick():
    wd = FrameWatchdog(timeout_s=0.25)
    assert wd.tick(100.0) is False


def test_watchdog_quiet_when_frames_are_fast():
    wd = FrameWatchdog(timeout_s=0.25)
    wd.tick(100.0)
    assert wd.tick(100.05) is False


def test_watchdog_fires_when_a_frame_stalls():
    wd = FrameWatchdog(timeout_s=0.25)
    wd.tick(100.0)
    assert wd.tick(100.5) is True


def test_watchdog_recovers_after_a_stall():
    wd = FrameWatchdog(timeout_s=0.25)
    wd.tick(100.0)
    wd.tick(100.5)
    assert wd.tick(100.55) is False


# ---------------------------------------------------------------------------
# Velocidade de cruzeiro constante (Fase 6b parte 2)
#
# Decisao do Rafael: velocidade CONSTANTE, sem aceleracao; e se ela nao puder ser
# constante, entao ZERO. O modelo comanda so o esterco -- as cabecas de throttle e
# brake sao ignoradas (a de freio esta inerte, o dataset tem 0% de frenagem).
# ---------------------------------------------------------------------------
import numpy as np
import pytest

from ai.car.control_map import (
    ESC_MAX_US, ESC_MIN_MOVE_US, clamp_cruise_us, front_blocked, front_min,
)


def test_cruise_passes_through_a_valid_value():
    assert clamp_cruise_us(1650) == 1650


def test_cruise_never_exceeds_the_calibrated_maximum():
    assert clamp_cruise_us(1800) == ESC_MAX_US
    assert clamp_cruise_us(9999) == ESC_MAX_US


def test_cruise_inside_the_esc_deadzone_becomes_neutral():
    # O ESC so anda a partir de 1600 us. Pedir 1550 nao moveria o carro; devolver
    # neutro mantem "comandado" e "real" coerentes em vez de fingir um comando.
    assert clamp_cruise_us(1550) == ESC_NEUTRAL_US
    assert clamp_cruise_us(ESC_MIN_MOVE_US) == ESC_MIN_MOVE_US


def test_cruise_never_goes_backwards():
    # Re nao faz parte desta fase: qualquer valor abaixo do neutro vira neutro.
    assert clamp_cruise_us(1400) == ESC_NEUTRAL_US
    assert clamp_cruise_us(0) == ESC_NEUTRAL_US


def test_cruise_with_garbage_is_neutral():
    assert clamp_cruise_us(None) == ESC_NEUTRAL_US
    assert clamp_cruise_us(float("nan")) == ESC_NEUTRAL_US
    assert clamp_cruise_us("rapido") == ESC_NEUTRAL_US
    assert clamp_cruise_us(float("inf")) == ESC_MAX_US


def test_front_min_looks_only_at_the_frontal_cone():
    v = np.ones(72, dtype=np.float32)
    v[0] = 0.2                                  # setor 0 = 2.5 graus, dentro do cone
    assert front_min(v) == pytest.approx(0.2)
    v2 = np.ones(72, dtype=np.float32)
    v2[36] = 0.05                               # 182.5 graus = atras
    assert front_min(v2) == pytest.approx(1.0)


def test_front_blocked_trips_on_something_close_ahead():
    v = np.ones(72, dtype=np.float32)
    v[0] = 0.2
    assert front_blocked(v, threshold=0.25) is True


def test_front_blocked_ignores_the_same_obstacle_behind():
    v = np.ones(72, dtype=np.float32)
    v[36] = 0.05
    assert front_blocked(v, threshold=0.25) is False


def test_front_blocked_is_false_on_a_clear_road():
    assert front_blocked(np.ones(72, dtype=np.float32), threshold=0.25) is False


# ---------------------------------------------------------------------------
# Faixa do servo ajustavel (2026-09-12)
#
# Os 1300/1700 vieram do controle por teclado e eram conservadores. Medido no
# carro (calibra_servo.py, 2026-09-12): o batente real e +/-300 us; usamos 300, com folga. Como o modelo
# aprendeu steer=+/-1 significando BATENTE TOTAL, mapear +/-1 para uma faixa
# menor que a real corta todo comando de esterco proporcionalmente.
# ---------------------------------------------------------------------------

def test_a_wider_span_turns_more_for_the_same_command():
    assert steer_to_us(-0.4) == 1620                      # padrao +/-300, espelhado
    assert steer_to_us(-0.4, span_us=200) == 1580         # o valor antigo, menor


def test_the_clamp_follows_the_span():
    assert steer_to_us(-5.0, span_us=400) == 1900
    assert steer_to_us(+5.0, span_us=400) == 1100


def test_garbage_still_centres_whatever_the_span():
    assert steer_to_us(float("nan"), span_us=400) == STEER_CENTER_US
    assert steer_to_us(None, span_us=400) == STEER_CENTER_US


# ---------------------------------------------------------------------------
# Sinal do esterco (MEDIDO no carro em 2026-09-12 com hardware/teste_esquerda.py)
#
# Comandando steer = -1 (ESQUERDA na convencao do CARLA) as rodas foram para a
# DIREITA. O servo deste carro responde espelhado. Como o espelho fica DEPOIS do
# modelo, a rede pedia uma coisa e o carro fazia a oposta -- em todas as corridas
# de pista ate aqui.
# ---------------------------------------------------------------------------

def test_negative_steer_turns_the_wheels_left_on_this_car():
    # 1800 us e o valor que fisicamente vira a esquerda NESTE carro
    assert steer_to_us(-1.0) == STEER_LEFT_US == 1800


def test_positive_steer_turns_the_wheels_right_on_this_car():
    assert steer_to_us(+1.0) == STEER_RIGHT_US == 1200


def test_the_mirror_is_a_single_documented_constant():
    """O espelho tem de ser explicito, nao um sinal trocado escondido na conta.

    Se o servo for remontado, muda-se STEER_SIGN e pronto; um sinal enterrado no
    meio da expressao seria achado meses depois, por alguem refazendo este mesmo
    teste no carro.
    """
    from ai.car.control_map import STEER_SIGN
    assert STEER_SIGN == -1


# ---------------------------------------------------------------------------
# Ganho de esterco (2026-09-26)
#
# Na pista o carro virou para o lado CERTO e faltou angulo: entrou na curva a
# esquerda, girou um pouco e bateu com a quina frontal esquerda na parede
# externa -- subesterco classico. O ganho e a alavanca para testar isso sem
# recoletar nada. E um AJUSTE DE ATUADOR explicito, nao um conserto do modelo:
# se a rede ja estiver saturando em +/-1 na curva, ele nao muda nada, e a causa
# esta na velocidade.
# ---------------------------------------------------------------------------
from ai.car.control_map import apply_steer_gain


def test_gain_of_one_changes_nothing():
    for s in (-1.0, -0.37, 0.0, 0.42, 1.0):
        assert apply_steer_gain(s, 1.0) == pytest.approx(s)


def test_gain_amplifies_a_partial_command():
    assert apply_steer_gain(0.30, 2.0) == pytest.approx(0.60)
    assert apply_steer_gain(-0.25, 1.6) == pytest.approx(-0.40)


def test_gain_cannot_push_past_full_lock():
    # O batente e fisico: amplificar alem de 1 nao existe no servo.
    assert apply_steer_gain(0.8, 2.0) == pytest.approx(1.0)
    assert apply_steer_gain(-0.8, 3.0) == pytest.approx(-1.0)


def test_gain_does_nothing_to_a_model_already_saturated():
    # O caso que decide o diagnostico: se a rede ja pede batente total, ganho
    # nenhum ajuda, e o problema nao esta aqui.
    assert apply_steer_gain(1.0, 2.5) == pytest.approx(1.0)
    assert apply_steer_gain(-1.0, 2.5) == pytest.approx(-1.0)


def test_gain_never_flips_the_side():
    # Ja tivemos um espelho de esterco neste carro; um sinal trocado aqui seria
    # a mesma falha silenciosa de novo.
    assert apply_steer_gain(0.5, 2.0) > 0
    assert apply_steer_gain(-0.5, 2.0) < 0


def test_unusable_input_or_gain_centres_the_wheel():
    assert apply_steer_gain(None, 2.0) == 0.0
    assert apply_steer_gain(float("nan"), 2.0) == 0.0
    assert apply_steer_gain("esquerda", 2.0) == 0.0
    assert apply_steer_gain(0.5, None) == pytest.approx(0.5)
    assert apply_steer_gain(0.5, float("nan")) == pytest.approx(0.5)


def test_a_negative_gain_is_refused_instead_of_mirroring_the_car():
    # Um ganho negativo espelharia o esterco inteiro -- exatamente o bug que
    # custou todas as corridas ate 2026-09-12. Recusa explicita.
    with pytest.raises(ValueError):
        apply_steer_gain(0.5, -1.0)


def test_the_gain_composes_with_the_span_to_reach_the_real_stop():
    # 0.5 com ganho 2 tem de dar o mesmo us que 1.0 sem ganho.
    assert steer_to_us(apply_steer_gain(0.5, 2.0)) == steer_to_us(1.0)


# ---------------------------------------------------------------------------
# Mediana causal do esterco (2026-10-03)
#
# Medido em runs/Diag/Diag, com os dois modelos rodando sobre as MESMAS imagens
# gravadas: o modelo com aumento fotometrico inverte de lado em 22% dos quadros
# e salta de batente a batente em quadros isolados. A mediana de 3 corta o
# |d steer| de 0.156 para 0.091 custando 10% da magnitude; uma EMA alisa parecido
# mas custa 27%, e magnitude e justamente o que falta.
#
# CAUSAL de proposito: no carro nao existe o quadro seguinte. Uma mediana
# centrada alisaria melhor nos numeros offline e seria impossivel de rodar.
# ---------------------------------------------------------------------------
from ai.car.control_map import SteerMedian


def test_the_filter_passes_the_first_frames_through():
    f = SteerMedian(3)
    assert f.push(0.4) == pytest.approx(0.4)      # sem historico, nao ha o que medianar
    assert f.push(0.6) == pytest.approx(0.5)      # mediana de dois = media dos dois


def test_an_isolated_spike_is_removed():
    # O padrao real medido: +0.4, +0.4, -0.88, +0.4 -- um quadro sozinho no
    # outro extremo. E o que fazia o servo nao ir a lugar nenhum.
    f = SteerMedian(3)
    f.push(0.4); f.push(0.4)
    assert f.push(-0.88) == pytest.approx(0.4)
    assert f.push(0.4) == pytest.approx(0.4)


def test_a_sustained_turn_survives_intact():
    # O que NAO pode acontecer: o filtro comer a curva. Depois da janela encher,
    # um comando mantido sai igual.
    f = SteerMedian(3)
    saida = [f.push(-0.9) for _ in range(6)]
    assert saida[-3:] == [pytest.approx(-0.9)] * 3


def test_a_real_step_costs_one_frame_of_delay():
    # O preco da causalidade: a mediana so acompanha o degrau no 2o quadro.
    f = SteerMedian(3)
    f.push(0.0); f.push(0.0)
    assert f.push(1.0) == pytest.approx(0.0)      # ainda nao
    assert f.push(1.0) == pytest.approx(1.0)      # agora sim


def test_window_of_one_is_a_passthrough():
    f = SteerMedian(1)
    for v in (0.3, -0.9, 0.0, 1.0):
        assert f.push(v) == pytest.approx(v)


def test_an_even_window_is_refused():
    # Mediana de janela par exige media dos dois centrais, que reintroduz o
    # pico que o filtro existe para remover. Recusa explicita.
    with pytest.raises(ValueError):
        SteerMedian(2)
    with pytest.raises(ValueError):
        SteerMedian(0)


def test_unusable_values_do_not_poison_the_window():
    f = SteerMedian(3)
    f.push(0.5); f.push(0.5)
    assert f.push(float("nan")) == pytest.approx(0.5)
    assert f.push(None) == pytest.approx(0.5)


def test_reset_clears_the_history():
    f = SteerMedian(3)
    f.push(0.9); f.push(0.9); f.push(0.9)
    f.reset()
    assert f.push(-0.2) == pytest.approx(-0.2)
