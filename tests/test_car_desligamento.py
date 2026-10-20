"""Segurar LB + RB para desligar o Jetson (src/ai/car/desligamento.py).

Na feira o carro nao tem monitor nem teclado, e tirar a bateria com o Jetson
ligado ja corrompeu o sistema de arquivos uma vez. O Rafael escolheu em
2026-10-20 desligar pelo controle: LB + RB juntos por 3 s, fora de corrida.
"""
from ai.car.desligamento import SegurarParaDesligar
from ai.car.gamepad import BACK, LB, RB, START, Estado


class Relogio(object):
    def __init__(self):
        self.agora = 100.0

    def __call__(self):
        return self.agora


def _estado(botoes):
    return Estado(botoes=botoes, lt=0, rt=0, ex=0, ey=0, dx=0, dy=0)


def _prepara():
    relogio = Relogio()
    return SegurarParaDesligar(segurar_s=3.0, relogio=relogio), relogio


def test_both_held_for_three_seconds_asks_to_shut_down():
    d, relogio = _prepara()
    assert not d.passo(_estado(LB | RB), correndo=False)
    relogio.agora += 2.9
    assert not d.passo(None, correndo=False)
    relogio.agora += 0.1
    assert d.passo(None, correndo=False)


def test_holding_still_is_silence_and_still_counts():
    # O adaptador so manda relatorio quando algo muda: segurar parado e None.
    d, relogio = _prepara()
    d.passo(_estado(LB | RB), correndo=False)
    for _ in range(24):                    # 24 x 0,125 = 3 s exatos em binario
        relogio.agora += 0.125
        disparou = d.passo(None, correndo=False)
    assert disparou


def test_one_button_alone_never_shuts_down():
    for botao in (LB, RB, LB | START, RB | BACK):
        d, relogio = _prepara()
        d.passo(_estado(botao), correndo=False)
        relogio.agora += 10.0
        assert not d.passo(None, correndo=False)


def test_other_buttons_with_both_still_count():
    d, relogio = _prepara()
    d.passo(_estado(LB | RB | BACK), correndo=False)
    relogio.agora += 3.0
    assert d.passo(None, correndo=False)


def test_letting_go_starts_the_count_again():
    d, relogio = _prepara()
    d.passo(_estado(LB | RB), correndo=False)
    relogio.agora += 2.5
    d.passo(_estado(LB), correndo=False)
    d.passo(_estado(LB | RB), correndo=False)
    relogio.agora += 2.5
    assert not d.passo(None, correndo=False)
    relogio.agora += 0.5
    assert d.passo(None, correndo=False)


def test_during_a_run_it_does_nothing():
    d, relogio = _prepara()
    d.passo(_estado(LB | RB), correndo=True)
    relogio.agora += 10.0
    assert not d.passo(None, correndo=True)


def test_a_hold_that_began_in_a_run_does_not_count_after_it():
    d, relogio = _prepara()
    d.passo(_estado(LB | RB), correndo=True)
    relogio.agora += 1.0
    d.passo(None, correndo=False)          # a corrida acabou, ainda segurando
    relogio.agora += 10.0
    assert not d.passo(None, correndo=False)
    # Soltou e apertou de novo, fora de corrida: agora vale.
    d.passo(_estado(0), correndo=False)
    d.passo(_estado(LB | RB), correndo=False)
    relogio.agora += 3.0
    assert d.passo(None, correndo=False)


def test_it_fires_once_per_hold():
    d, relogio = _prepara()
    d.passo(_estado(LB | RB), correndo=False)
    relogio.agora += 3.0
    assert d.passo(None, correndo=False)
    relogio.agora += 5.0
    assert not d.passo(None, correndo=False)
