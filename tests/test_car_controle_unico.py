"""Um programa so para o controle: dirige a mao e larga a corrida autonoma.

A regra que manda em tudo aqui: o PCA9685 tem UM dono por vez. Sem corrida,
quem escreve nele e o controle manual. Com corrida, e o jetson_runtime.py, e o
programa do controle nao escreve nada -- dois processos mandando pulsos ao
mesmo servo e ESC fariam o carro obedecer a quem escreveu por ultimo.
"""
import signal

from ai.car.gamepad import BACK, START, Estado
from ai.car.controle_unico import ControleUnico
from ai.car.manual import Manual
from test_car_largada import Fabrica, Relogio

NEUTRO = 1500


def _e(ex=0, lt=0, rt=0, botoes=0):
    return Estado(botoes=botoes, lt=lt, rt=rt, ex=ex, ey=0, dx=0, dy=0)


class Mundo(object):
    """O que o ControleUnico toca fora dele, anotando a ordem dos acontecimentos."""

    def __init__(self, estaciona_ok=True, **como_a_corrida):
        self.ordem = []
        self.eventos = []
        self.fabrica = Fabrica(**como_a_corrida)
        self._estaciona_ok = estaciona_ok

    def inicia(self):
        self.ordem.append("inicia")
        return self.fabrica()

    def estaciona(self):
        viva = any(c.codigo is None for c in self.fabrica.corridas)
        self.ordem.append("estaciona com corrida viva" if viva else "estaciona")
        return self._estaciona_ok


def _unico(**como):
    mundo = Mundo(**como)
    unico = ControleUnico(mundo.inicia, Manual(max_us=1630, relogio=Relogio()),
                          mundo.estaciona, avisa=mundo.eventos.append,
                          relogio=Relogio())
    return unico, mundo


# ------------------------------------------------------------- modo manual

def test_without_a_run_the_controller_drives_the_car():
    unico, _ = _unico()
    assert unico.passo(_e(rt=255)).esc_us == 1630


def test_back_without_a_run_is_the_manual_stop():
    unico, mundo = _unico()
    cmd = unico.passo(_e(rt=255, botoes=BACK))
    assert cmd.esc_us == NEUTRO and cmd.motivo == "parado"
    assert mundo.fabrica.corridas == []


# ---------------------------------------------------------- corrida autonoma

def test_start_parks_the_car_before_launching_the_run():
    unico, mundo = _unico()
    unico.passo(_e(botoes=START))
    assert mundo.ordem == ["estaciona", "inicia"]


def test_start_while_driving_by_hand_parks_the_car_first():
    unico, mundo = _unico()
    unico.passo(_e(rt=255))
    assert unico.passo(_e(rt=255, botoes=START)) is None
    assert mundo.ordem == ["estaciona", "inicia"]


def test_while_the_run_is_on_nothing_goes_to_the_board():
    unico, _ = _unico()
    unico.passo(_e(botoes=START))
    assert unico.passo(_e(rt=255, ex=32767)) is None
    assert unico.passo(None) is None


def test_back_during_the_run_stops_the_run_and_the_board_stays_with_it():
    unico, mundo = _unico()
    unico.passo(_e(botoes=START))
    unico.passo(_e())
    assert unico.passo(_e(botoes=BACK)) is None   # ainda limpando: dono e o runtime
    assert mundo.fabrica.corridas[0].sinais == [signal.SIGINT]


def test_when_the_run_ends_the_car_waits_for_the_triggers_to_be_released():
    unico, mundo = _unico()
    unico.passo(_e(botoes=START))
    unico.passo(_e(rt=255))
    mundo.fabrica.corridas[0].codigo = 0
    assert unico.passo(_e(rt=255)).esc_us == NEUTRO
    unico.passo(_e(rt=0))
    assert unico.passo(_e(rt=255)).esc_us == 1630


def test_the_operator_hears_about_the_run():
    unico, mundo = _unico()
    unico.passo(_e(botoes=START))
    unico.passo(_e())
    unico.passo(_e(botoes=BACK))
    mundo.fabrica.corridas[0].codigo = 0
    unico.passo(None)
    assert mundo.eventos == ["largou", "parando", "acabou"]


# ------------------------------------------------------------- encerramento

def test_shutting_down_stops_the_run_before_parking_the_car():
    # Escrever no PCA9685 com o runtime vivo seria disputar o servo com ele.
    unico, mundo = _unico()
    unico.passo(_e(botoes=START))
    assert unico.encerra() is True
    assert mundo.fabrica.corridas[0].sinais == [signal.SIGINT]
    assert mundo.ordem == ["estaciona", "inicia", "estaciona"]


def test_shutting_down_without_a_run_parks_the_car():
    unico, mundo = _unico()
    assert unico.encerra() is True
    assert mundo.ordem == ["estaciona"]


def test_shutting_down_parks_the_car_even_if_the_run_will_not_stop():
    # Runtime travado: ele ja nao escreve nada, entao o neutro daqui ajuda.
    unico, mundo = _unico(obedece=False)
    unico.passo(_e(botoes=START))
    assert unico.encerra() is False
    assert mundo.ordem[-1] == "estaciona com corrida viva"


def test_shutting_down_says_so_when_the_car_could_not_be_parked():
    unico, mundo = _unico(estaciona_ok=False)
    assert unico.encerra() is False
