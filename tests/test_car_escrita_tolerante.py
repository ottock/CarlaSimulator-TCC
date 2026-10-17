"""Escrever no PCA9685 aguentando um fio que solta por um instante.

Visto em 2026-10-14 no controle manual: um jumper do I2C soltou por um
instante, uma escrita falhou com Errno 121 (Remote I/O error) e o programa
caiu. A escrita seguinte, a do estado seguro, funcionou: o contato ja tinha
voltado. Cair nao deixa o carro mais seguro -- o PCA9685 segue mandando o
ultimo pulso sozinho, e a parada sai pelo mesmo fio. Continuar tentando faz o
comando atual (inclusive "parar") chegar assim que o contato voltar.
"""

import pytest

from ai.car.escrita_tolerante import EscritaTolerante, PCA9685Perdido, para_com_insistencia


class Relogio(object):
    def __init__(self):
        self.agora = 100.0

    def __call__(self):
        return self.agora


class Placa(object):
    """Faz as vezes do PCA9685: falha nas escritas marcadas, guarda as outras."""

    def __init__(self, falhas=()):
        self.recebidos = []
        self._falhas = set(falhas)
        self._n = 0

    def escreve(self, servo_us, esc_us):
        self._n += 1
        if self._n in self._falhas:
            raise OSError(121, "Remote I/O error")   # EREMOTEIO no Jetson (Linux)
        self.recebidos.append((servo_us, esc_us))


def _tolerante(placa, **como):
    relogio = Relogio()
    return EscritaTolerante(placa.escreve, relogio=relogio, **como), relogio


def test_a_single_failed_write_does_not_stop_the_program():
    placa = Placa(falhas={2})
    escrita, _ = _tolerante(placa)
    assert escrita.manda(1500, 1600) is True
    assert escrita.manda(1500, 1610) is False
    assert escrita.manda(1500, 1620) is True
    assert placa.recebidos == [(1500, 1600), (1500, 1620)]


def test_failures_that_last_too_long_mean_the_board_is_lost():
    placa = Placa(falhas={1, 2, 3})
    escrita, relogio = _tolerante(placa, limite_s=0.1)
    escrita.manda(1500, 1600)
    relogio.agora += 0.05
    escrita.manda(1500, 1600)
    relogio.agora += 0.06
    with pytest.raises(PCA9685Perdido):
        escrita.manda(1500, 1600)


def test_a_write_that_gets_through_restarts_the_count():
    # Contato intermitente que volta: cada falha isolada e so uma falha.
    placa = Placa(falhas={1, 3})
    escrita, relogio = _tolerante(placa, limite_s=0.1)
    escrita.manda(1500, 1600)
    relogio.agora += 0.05
    escrita.manda(1500, 1600)
    relogio.agora += 0.2
    assert escrita.manda(1500, 1600) is False


def test_the_failures_are_counted_for_the_operator_to_see():
    placa = Placa(falhas={1, 3})
    escrita, _ = _tolerante(placa)
    for _ in range(4):
        escrita.manda(1500, 1500)
    assert escrita.falhas == 2


def test_a_programming_error_is_not_mistaken_for_a_loose_wire():
    def quebrada(servo_us, esc_us):
        raise TypeError("bug, nao fio")

    escrita = EscritaTolerante(quebrada, relogio=Relogio())
    with pytest.raises(TypeError):
        escrita.manda(1500, 1500)


def test_stopping_insists_until_a_write_gets_through():
    placa = Placa(falhas={1, 2, 3})
    pausas = []
    assert para_com_insistencia(placa.escreve, 1500, 1500, tentativas=10,
                                dorme=pausas.append) is True
    assert placa.recebidos == [(1500, 1500)]
    assert len(pausas) == 3


def test_stopping_gives_up_after_the_last_attempt_and_says_so():
    placa = Placa(falhas=set(range(1, 100)))
    assert para_com_insistencia(placa.escreve, 1500, 1500, tentativas=5,
                                dorme=lambda s: None) is False
    assert placa.recebidos == []
