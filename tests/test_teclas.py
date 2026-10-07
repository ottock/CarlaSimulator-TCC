"""Decodificacao de teclas para a sonda estatica (Fase 6c).

A sonda e operada as cegas: o Rafael esta agachado na pista posicionando o
carro e nao ve o terminal. Entao ela e dirigida por tecla, sem Enter, e as
setas chegam como SEQUENCIA DE ESCAPE de tres bytes -- nao como um caractere.
Errar isso faz a seta virar "ESC" e abortar a medicao no meio.
"""
import pytest

from ai.car.teclas import decodifica


def test_the_arrow_keys_arrive_as_three_byte_escape_sequences():
    assert decodifica("\x1b[C") == "direita"
    assert decodifica("\x1b[D") == "esquerda"
    assert decodifica("\x1b[A") == "cima"
    assert decodifica("\x1b[B") == "baixo"


def test_plain_letters_work_as_a_fallback():
    # Nem todo terminal por SSH manda a sequencia esperada; as letras dao saida.
    assert decodifica("d") == "direita"
    assert decodifica("a") == "esquerda"
    assert decodifica("q") == "sair"


def test_uppercase_is_accepted():
    assert decodifica("D") == "direita"
    assert decodifica("Q") == "sair"


def test_a_lone_escape_is_quit_not_an_arrow():
    # ESC sozinho e o usuario desistindo. Se fosse tratado como inicio de seta,
    # o programa ficaria esperando dois bytes que nunca vem.
    assert decodifica("\x1b") == "sair"


def test_enter_and_space_advance():
    # Quem nao achar a seta no escuro aperta Enter ou espaco.
    for t in ("\r", "\n", " "):
        assert decodifica(t) == "direita"


def test_ctrl_c_is_quit():
    assert decodifica("\x03") == "sair"


def test_an_unknown_key_is_ignored_rather_than_guessed():
    # Adivinhar aqui faria uma tecla errada disparar uma medicao que o Rafael
    # nao pediu, e ele nao veria acontecer.
    assert decodifica("z") is None
    assert decodifica("\x1b[Z") is None
    assert decodifica("") is None
