"""Um programa so para o controle: dirige a mao e roda a corrida autonoma.

Desde 2026-10-17 a corrida roda DENTRO do programa do controle: o modelo, a
camera e o LiDAR sao abertos uma vez, e o Start so comeca a chamar o laco de
direcao. Assim largar leva o tempo da trava de arranque, e parar e imediato.

A regra continua: quem escreve na placa e UM so. Sem corrida, o controle
manual; com corrida, o laco do modelo. Os dois nunca no mesmo giro.
"""
import pytest

from ai.car.controle_unico import ControleUnico
from ai.car.gamepad import BACK, START, Estado
from ai.car.manual import Manual

NEUTRO = 1500


def _e(ex=0, lt=0, rt=0, botoes=0):
    return Estado(botoes=botoes, lt=lt, rt=rt, ex=ex, ey=0, dx=0, dy=0)


class Relogio(object):
    def __init__(self):
        self.agora = 100.0

    def __call__(self):
        return self.agora


class Corrida(object):
    """O laco do modelo para uma corrida: cada ``passo`` e um quadro."""

    def __init__(self, mundo, falha=None):
        self._mundo = mundo
        self._falha = falha
        self.passos = 0
        self.fechada = False

    def passo(self):
        self.passos += 1
        if self._falha is not None:
            raise self._falha
        return {"esc_us": 1610, "passo": self.passos}

    def fecha(self):
        self.fechada = True
        self._mundo.ordem.append("fecha")


class Mundo(object):
    """O que o ControleUnico toca fora dele, anotando a ordem dos acontecimentos."""

    def __init__(self, estaciona_ok=True, falha=None):
        self.ordem = []
        self.eventos = []
        self.corridas = []
        self._estaciona_ok = estaciona_ok
        self._falha = falha

    def nova_corrida(self):
        self.ordem.append("inicia")
        self.corridas.append(Corrida(self, self._falha))
        return self.corridas[-1]

    def estaciona(self):
        self.ordem.append("estaciona")
        return self._estaciona_ok


def _unico(limite_s=60.0, **como):
    mundo, relogio = Mundo(**como), Relogio()
    unico = ControleUnico(mundo.nova_corrida, Manual(max_us=1630, relogio=relogio),
                          mundo.estaciona, avisa=mundo.eventos.append,
                          relogio=relogio, limite_s=limite_s)
    return unico, mundo, relogio


def _largado(**como):
    unico, mundo, relogio = _unico(**como)
    unico.passo(_e(botoes=START))
    unico.passo(_e())
    return unico, mundo, relogio


# ------------------------------------------------------------- modo manual

def test_without_a_run_the_controller_drives_the_car():
    unico, _, _ = _unico()
    assert unico.passo(_e(rt=255)).esc_us == 1630


def test_back_without_a_run_is_the_manual_stop():
    unico, mundo, _ = _unico()
    cmd = unico.passo(_e(rt=255, botoes=BACK))
    assert cmd.esc_us == NEUTRO and cmd.motivo == "parado"
    assert mundo.corridas == []


# ---------------------------------------------------------------- largar

def test_start_parks_the_car_before_starting_the_run():
    unico, mundo, _ = _unico()
    unico.passo(_e(botoes=START))
    assert mundo.ordem == ["estaciona", "inicia"]
    assert unico.correndo


def test_start_while_driving_by_hand_parks_first_and_writes_nothing():
    unico, mundo, _ = _unico()
    unico.passo(_e(rt=255))
    assert unico.passo(_e(rt=255, botoes=START)) is None
    assert mundo.ordem == ["estaciona", "inicia"]


def test_start_and_back_together_do_not_start_a_run():
    unico, mundo, _ = _unico()
    unico.passo(_e(botoes=START | BACK))
    assert mundo.corridas == []


# -------------------------------------------------------- durante a corrida

def test_the_first_step_of_the_model_runs_in_the_same_turn_as_start():
    unico, mundo, _ = _unico()
    unico.passo(_e(botoes=START))
    assert mundo.corridas[0].passos == 1


def test_during_the_run_every_turn_is_one_step_of_the_model():
    # Com ou sem relatorio do controle: o modelo nao espera o controle mudar.
    unico, mundo, _ = _largado()
    antes = mundo.corridas[0].passos
    assert unico.passo(None) is None
    assert unico.passo(_e(rt=255, ex=32767)) is None
    assert mundo.corridas[0].passos == antes + 2


def test_the_last_step_telemetry_is_kept_for_the_screen():
    unico, mundo, _ = _largado()
    assert unico.telemetria == {"esc_us": 1610, "passo": mundo.corridas[0].passos}


def test_start_again_during_the_run_does_not_start_another():
    unico, mundo, _ = _largado()
    unico.passo(_e(botoes=START))
    assert len(mundo.corridas) == 1


# ----------------------------------------------------------------- parar

def test_back_during_the_run_stops_it_at_once():
    unico, mundo, _ = _largado()
    antes = mundo.corridas[0].passos
    assert unico.passo(_e(botoes=BACK)) is None
    assert not unico.correndo
    assert mundo.ordem[-2:] == ["estaciona", "fecha"]   # para o carro, depois fecha o log
    assert mundo.corridas[0].passos == antes            # nenhum quadro no giro do Back


def test_after_the_run_the_car_waits_for_the_triggers_to_be_released():
    unico, _, _ = _largado()
    unico.passo(_e(rt=255))
    unico.passo(_e(rt=255, botoes=BACK))
    assert unico.passo(_e(rt=255)).esc_us == NEUTRO
    unico.passo(_e(rt=0))
    assert unico.passo(_e(rt=255)).esc_us == 1630


def test_a_start_still_held_after_the_run_does_not_start_another():
    # O adaptador nao zera o estado quando o controle desliga: vale a borda.
    unico, mundo, _ = _unico()
    unico.passo(_e(botoes=START))
    unico.passo(_e(botoes=START | BACK))
    unico.passo(_e(botoes=START))
    assert len(mundo.corridas) == 1


def test_the_run_stops_by_itself_at_the_time_limit():
    unico, mundo, relogio = _largado(limite_s=60.0)
    relogio.agora += 61.0
    unico.passo(None)
    assert not unico.correndo
    assert mundo.eventos[-1] == "tempo"
    assert mundo.ordem[-2:] == ["estaciona", "fecha"]


def test_an_error_in_the_model_stops_the_run_and_keeps_the_program():
    # Ex.: um fio do I2C solto no meio da corrida. Antes, com o runtime num
    # processo a parte, ele caia sozinho; aqui o programa para a corrida e volta
    # ao manual em vez de cair inteiro.
    unico, mundo, _ = _unico(falha=OSError(121, "Remote I/O error"))
    unico.passo(_e(botoes=START))
    assert not unico.correndo
    assert mundo.eventos[-1] == "erro"
    assert "Remote I/O error" in unico.erro
    assert mundo.ordem == ["estaciona", "inicia", "estaciona", "fecha"]


def test_ctrl_c_during_a_step_is_not_swallowed():
    unico, _, _ = _unico(falha=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        unico.passo(_e(botoes=START))


def test_the_operator_hears_why_the_run_ended():
    unico, mundo, _ = _largado()
    unico.passo(_e(botoes=BACK))
    assert mundo.eventos == ["largou", "parada"]


# ------------------------------------------------------------- encerramento

def test_shutting_down_stops_the_run_and_parks_the_car():
    unico, mundo, _ = _largado()
    assert unico.encerra() is True
    assert mundo.corridas[0].fechada
    assert mundo.ordem[-2:] == ["estaciona", "fecha"]


def test_shutting_down_without_a_run_parks_the_car():
    unico, mundo, _ = _unico()
    assert unico.encerra() is True
    assert mundo.ordem == ["estaciona"]


def test_shutting_down_says_so_when_the_car_could_not_be_parked():
    unico, _, _ = _unico(estaciona_ok=False)
    assert unico.encerra() is False
