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


# ---------------------------------------------------------------------------
# A LEITURA dos bytes (2026-10-07)
#
# Na pista, qualquer seta encerrava a sonda. A tabela acima estava certa: o erro
# era ler com `sys.stdin.read(1)`, que tem buffer. A seta manda 3 bytes juntos,
# o read(1) puxa os 3 do descritor e devolve so o \x1b -- e \x1b sozinho e
# "sair". Estes testes alimentam um pipe com os bytes exatos da seta.
# ---------------------------------------------------------------------------
import os

import pytest

from ai.car.teclas import LeitorDeTeclas


def _leitor_com(*pedacos):
    """Leitor ligado a um pipe ja carregado com estes bytes."""
    r, w = os.pipe()
    for p in pedacos:
        os.write(w, p)
    os.close(w)
    return LeitorDeTeclas(fd=r), r


def test_an_arrow_arriving_as_one_chunk_is_not_cut_into_a_bare_escape():
    """O teste que pega o bug da pista: 3 bytes de uma vez devem virar
    'direita', nunca 'sair'."""
    leitor, r = _leitor_com(b"\x1b[C")
    try:
        texto = leitor.ler()
        assert texto == "\x1b[C", "a seta chegou partida: %r" % texto
        assert decodifica(texto) == "direita"
    finally:
        os.close(r)


def test_the_left_arrow_too():
    leitor, r = _leitor_com(b"\x1b[D")
    try:
        assert decodifica(leitor.ler()) == "esquerda"
    finally:
        os.close(r)


def test_two_arrows_in_a_row_are_two_separate_actions():
    """Se a primeira leitura levasse bytes da segunda, a medicao seguinte
    receberia lixo e seria ignorada em silencio."""
    leitor, r = _leitor_com(b"\x1b[C", b"\x1b[C")
    try:
        assert decodifica(leitor.ler()) == "direita"
        assert decodifica(leitor.ler()) == "direita"
    finally:
        os.close(r)


def test_a_plain_letter_still_works():
    leitor, r = _leitor_com(b"d")
    try:
        assert decodifica(leitor.ler()) == "direita"
    finally:
        os.close(r)


def test_enter_and_space_still_measure():
    for byte in (b"\r", b"\n", b" "):
        leitor, r = _leitor_com(byte)
        try:
            assert decodifica(leitor.ler()) == "direita"
        finally:
            os.close(r)


def test_q_quits():
    leitor, r = _leitor_com(b"q")
    try:
        assert decodifica(leitor.ler()) == "sair"
    finally:
        os.close(r)


def test_closed_input_quits_instead_of_spinning_forever():
    """Sem isto, um terminal que cai deixaria o laco girando com o carro na
    pista e ninguem olhando a tela."""
    r, w = os.pipe()
    os.close(w)
    leitor = LeitorDeTeclas(fd=r)
    try:
        assert decodifica(leitor.ler()) == "sair"
    finally:
        os.close(r)


def test_an_injected_fd_never_touches_the_terminal():
    """O contexto com fd injetado nao pode mexer no terminal -- se mexesse, os
    testes derrubariam o terminal de quem roda a suite."""
    r, w = os.pipe()
    os.write(w, b"d")
    os.close(w)
    try:
        with LeitorDeTeclas(fd=r) as t:
            assert t.ler() == "d"
    finally:
        os.close(r)


@pytest.mark.skipif(os.name == "nt",
                    reason="select() no Windows so funciona com socket, nao com pipe")
def test_a_lone_escape_is_still_a_quit():
    """ESC sozinho tem de continuar encerrando: e a saida de emergencia."""
    leitor, r = _leitor_com(b"\x1b")
    try:
        assert decodifica(leitor.ler()) == "sair"
    finally:
        os.close(r)
