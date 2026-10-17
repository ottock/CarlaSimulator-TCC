"""Dirigir o carro com o controle 8BitDo: estado do controle -> servo e ESC.

Os numeros esperados vem do carro, nao deste modulo: 1850/1150 us sao os lados
medidos com span 350 (control_map.py), 1390 e a re que o Rafael pediu no carro
(2026-10-17), 1600 e o minimo que move o ESC e 1700 o maximo calibrado.

O que estes testes NAO cobrem: o sentido do analogico no carro (direita no
controle = rodas para a direita) e a re de verdade no ESC. Isso e o primeiro
teste no Jetson, com as rodas no ar.
"""
import pytest

from ai.car.gamepad import BACK, HOME, Estado
from ai.car.manual import Manual

NEUTRO = 1500
CENTRO = 1500


class Relogio(object):
    def __init__(self):
        self.agora = 100.0

    def __call__(self):
        return self.agora


def _e(ex=0, lt=0, rt=0, botoes=0):
    return Estado(botoes=botoes, lt=lt, rt=rt, ex=ex, ey=0, dx=0, dy=0)


def _manual(**como):
    relogio = Relogio()
    return Manual(relogio=relogio, **como), relogio


def _depois_de(estado, **como):
    m, _ = _manual(**como)
    m.recebe(estado)
    return m.comando()


# ----------------------------------------------------------------- volante

def test_before_any_report_the_car_stays_centred_and_stopped():
    m, _ = _manual()
    c = m.comando()
    assert (c.servo_us, c.esc_us) == (CENTRO, NEUTRO)


def test_stick_full_right_turns_the_wheels_full_right():
    assert _depois_de(_e(ex=32767)).servo_us == 1150


def test_stick_full_left_turns_the_wheels_full_left():
    # -32768 passa de -1 por um fio: tem de parar no batente, nao alem dele.
    assert _depois_de(_e(ex=-32768)).servo_us == 1850


def test_half_stick_is_half_a_turn():
    assert _depois_de(_e(ex=16384)).servo_us == 1325


def test_a_smaller_span_turns_less():
    assert _depois_de(_e(ex=32767), span_us=260).servo_us == 1240


# --------------------------------------------------------------- acelerador

def test_rt_released_does_not_move():
    assert _depois_de(_e(rt=0)).esc_us == NEUTRO


def test_a_finger_resting_on_rt_does_not_creep():
    # 25/255 fica abaixo de 10% do curso.
    assert _depois_de(_e(rt=25)).esc_us == NEUTRO


def test_rt_just_past_the_dead_zone_gives_the_slowest_speed_that_moves():
    assert _depois_de(_e(rt=26)).esc_us == 1600


def test_rt_halfway_is_halfway_between_1600_and_1700():
    assert _depois_de(_e(rt=140)).esc_us == 1650


def test_rt_all_the_way_gives_the_maximum():
    assert _depois_de(_e(rt=255)).esc_us == 1700


def test_a_lower_maximum_caps_the_speed():
    # "Qualquer coisa a gente diminui depois."
    assert _depois_de(_e(rt=255), max_us=1650).esc_us == 1650


def test_a_maximum_above_the_calibrated_one_is_capped():
    assert _depois_de(_e(rt=255), max_us=1800).esc_us == 1700


def test_a_maximum_below_the_moving_threshold_is_refused():
    # Com maximo abaixo de 1600 o carro nunca andaria, e ninguem entenderia por que.
    with pytest.raises(ValueError):
        Manual(max_us=1550)


# --------------------------------------------------------------------- re

def test_lt_held_goes_in_reverse():
    assert _depois_de(_e(lt=255)).esc_us == 1390


def test_a_finger_resting_on_lt_does_not_reverse():
    assert _depois_de(_e(lt=25)).esc_us == NEUTRO


def test_both_triggers_held_stops_the_car():
    assert _depois_de(_e(lt=255, rt=255)).esc_us == NEUTRO


# ------------------------------------------------------- parada (Back, Home)

def test_back_stops_the_car_even_with_rt_held():
    c = _depois_de(_e(rt=255, botoes=BACK))
    assert c.esc_us == NEUTRO and c.motivo == "parado"


def test_home_stops_the_car_too():
    # O Home e apertado antes de o controle desligar: o carro ja estara parado.
    assert _depois_de(_e(rt=255, botoes=HOME)).esc_us == NEUTRO


def test_after_back_the_car_stays_stopped_while_rt_is_still_held():
    m, _ = _manual()
    m.recebe(_e(rt=255, botoes=BACK))
    m.recebe(_e(rt=255))
    assert m.comando().esc_us == NEUTRO


def test_after_back_releasing_and_pressing_rt_again_moves():
    m, _ = _manual()
    m.recebe(_e(rt=255, botoes=BACK))
    m.recebe(_e(rt=0))
    m.recebe(_e(rt=255))
    assert m.comando().esc_us == 1700


def test_after_back_reverse_also_needs_the_triggers_released_first():
    m, _ = _manual()
    m.recebe(_e(rt=255, botoes=BACK))
    m.recebe(_e(lt=255))
    assert m.comando().esc_us == NEUTRO


def test_trava_holds_the_car_until_the_triggers_are_released():
    # Usado quando a corrida autonoma acaba: um RT que ficou apertado nao
    # pode arrancar o carro no instante em que o manual volta.
    m, _ = _manual()
    m.recebe(_e(rt=255))
    m.trava()
    assert m.comando().esc_us == NEUTRO
    m.recebe(_e(rt=0))
    m.recebe(_e(rt=255))
    assert m.comando().esc_us == 1700


# --------------------------------------------------- controle que some (silencio)

def test_a_silent_controller_with_rt_held_stops_the_car():
    # O adaptador nao avisa quando o controle desliga e o ultimo estado fica
    # valendo: sem isto o carro seguiria acelerado ate bater.
    m, relogio = _manual(silencio_s=0.5)
    m.recebe(_e(rt=255))
    relogio.agora += 0.6
    c = m.comando()
    assert c.esc_us == NEUTRO and c.motivo == "silencio"


def test_a_short_silence_keeps_the_car_moving():
    m, relogio = _manual(silencio_s=0.5)
    m.recebe(_e(rt=255))
    relogio.agora += 0.4
    assert m.comando().esc_us == 1700


def test_the_car_moves_again_when_reports_come_back():
    m, relogio = _manual(silencio_s=0.5)
    m.recebe(_e(rt=255))
    relogio.agora += 0.6
    m.recebe(_e(rt=250))
    assert m.comando().esc_us > NEUTRO


def test_silence_with_the_car_stopped_is_not_an_alarm():
    # Mao parada e controle solto e o normal: o adaptador so manda quando muda.
    m, relogio = _manual(silencio_s=0.5)
    m.recebe(_e(rt=0))
    relogio.agora += 5.0
    assert m.comando().motivo is None
