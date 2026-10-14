"""Largar e parar a corrida pelo controle: Start larga, Back para.

O que estes testes NAO cobrem: o sinal de verdade chegando ao jetson_runtime.
A corrida aqui e um substituto do processo; a primeira prova real e no Jetson,
com as rodas no ar.
"""
import ast
import datetime
import io
import os
import signal
import subprocess

from ai.car.gamepad import BACK, START, ControlePerdido
from ai.car.largada import Largada, comando_da_corrida, laco, pasta_da_corrida

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RUNTIME = os.path.join(_REPO, "hardware", "jetson_runtime.py")


class Corrida(object):
    """Faz as vezes do processo do runtime (subprocess.Popen).

    Como o de verdade, nao morre no instante do sinal: fica viva (limpando) ate
    alguem esperar por ela. ``obedece=False`` e um runtime travado;
    ``acaba_sozinha`` e o Ctrl+C do terminal, que chega nela sem passar por nos.
    """

    def __init__(self, obedece=True, acaba_sozinha=False):
        self.sinais = []
        self.matou = False
        self.codigo = None
        self._obedece = obedece
        self._acaba_sozinha = acaba_sozinha

    def poll(self):
        return self.codigo

    def send_signal(self, sinal):
        self.sinais.append(sinal)

    def wait(self, timeout=None):
        if self.codigo is None and (
                self._acaba_sozinha or (self._obedece and self.sinais)):
            self.codigo = 0
        if self.codigo is None:
            raise subprocess.TimeoutExpired("runtime", timeout)
        return self.codigo

    def kill(self):
        self.matou = True
        self.codigo = -9

    terminate = kill


class Fabrica(object):
    """O ``inicia`` da Largada: cria uma Corrida e lembra de todas."""

    def __init__(self, **como):
        self.corridas = []
        self._como = como

    def __call__(self):
        self.corridas.append(Corrida(**self._como))
        return self.corridas[-1]


class Relogio(object):
    def __init__(self):
        self.agora = 100.0

    def __call__(self):
        return self.agora


def _largada(**como):
    fabrica, relogio, eventos = Fabrica(**como), Relogio(), []
    return Largada(fabrica, avisa=eventos.append, relogio=relogio), fabrica, relogio, eventos


# ---------------------------------------------------------------- Start

def test_start_launches_a_run():
    largada, fabrica, _, _ = _largada()
    largada.passo(START)
    assert len(fabrica.corridas) == 1


def test_start_is_ignored_while_a_run_is_in_progress():
    # O modelo leva 1-2 min para carregar: quem nao ve nada acontecer aperta
    # de novo. Uma segunda corrida disputaria camera, LiDAR e PCA9685.
    largada, fabrica, _, _ = _largada()
    largada.passo(START)
    largada.passo(0)
    largada.passo(START)
    assert len(fabrica.corridas) == 1


def test_start_launches_again_after_the_previous_run_ended():
    largada, fabrica, _, _ = _largada()
    largada.passo(START)
    largada.passo(0)
    fabrica.corridas[0].codigo = 0
    largada.passo(START)
    assert len(fabrica.corridas) == 2


def test_a_start_still_held_when_the_run_ends_does_not_launch_another():
    # Quando o controle desliga, o adaptador nao zera: o ultimo estado fica
    # valendo. Agir no nivel faria um Start "preso" largar corrida atras de
    # corrida. Vale a borda: so o instante em que apertou.
    largada, fabrica, _, _ = _largada()
    largada.passo(START)
    fabrica.corridas[0].codigo = 0
    largada.passo(START)
    assert len(fabrica.corridas) == 1


def test_start_and_back_pressed_together_do_not_launch():
    largada, fabrica, _, _ = _largada()
    largada.passo(START | BACK)
    assert fabrica.corridas == []


# ----------------------------------------------------------------- Back

def test_back_interrupts_the_run_the_way_ctrl_c_does():
    # O runtime ja centraliza o servo e poe o ESC em neutro ao receber Ctrl+C.
    largada, fabrica, _, _ = _largada()
    largada.passo(START)
    largada.passo(0)
    largada.passo(BACK)
    assert fabrica.corridas[0].sinais == [signal.SIGINT]


def test_back_with_no_run_does_nothing():
    largada, fabrica, _, eventos = _largada()
    largada.passo(BACK)
    assert fabrica.corridas == [] and eventos == []


def test_a_second_back_right_after_the_first_does_not_signal_again():
    # Um segundo Ctrl+C pode cair DENTRO da limpeza do runtime e interrompe-la
    # antes de o ESC ir a neutro. Quem aperta duas vezes por nervosismo nao
    # pode piorar a parada.
    largada, fabrica, relogio, _ = _largada()
    largada.passo(START)
    largada.passo(0)
    largada.passo(BACK)
    largada.passo(0)
    relogio.agora += 0.5
    largada.passo(BACK)
    assert fabrica.corridas[0].sinais == [signal.SIGINT]


def test_back_signals_again_when_the_run_is_still_alive_long_after():
    largada, fabrica, relogio, _ = _largada()
    largada.passo(START)
    largada.passo(0)
    largada.passo(BACK)
    largada.passo(0)
    relogio.agora += 10.0
    largada.passo(BACK)
    assert fabrica.corridas[0].sinais == [signal.SIGINT, signal.SIGINT]


# ------------------------------------------------------- avisos ao operador

def test_the_operator_hears_about_the_launch_the_stop_and_the_end():
    largada, fabrica, _, eventos = _largada()
    largada.passo(START)
    largada.passo(0)
    largada.passo(BACK)
    fabrica.corridas[0].codigo = 0
    largada.passo(None)          # nenhum relatorio novo: so o tempo passando
    assert eventos == ["largou", "parando", "acabou"]


# ------------------------------------------------------------- encerramento

def test_shutting_down_interrupts_a_run_in_progress():
    largada, fabrica, _, _ = _largada()
    largada.passo(START)
    assert largada.encerra() is True
    assert fabrica.corridas[0].sinais == [signal.SIGINT]


def test_shutting_down_with_no_run_has_nothing_to_stop():
    largada, fabrica, _, _ = _largada()
    assert largada.encerra() is True


def test_shutting_down_after_ctrl_c_waits_without_signalling_again():
    # O Ctrl+C do terminal chega ao runtime tambem (mesmo grupo de processos).
    largada, fabrica, _, _ = _largada(acaba_sozinha=True)
    largada.passo(START)
    assert largada.encerra(sinaliza=False) is True
    assert fabrica.corridas[0].sinais == []


def test_shutting_down_signals_when_the_ctrl_c_did_not_reach_the_run():
    largada, fabrica, _, _ = _largada()
    largada.passo(START)
    assert largada.encerra(sinaliza=False) is True
    assert fabrica.corridas[0].sinais == [signal.SIGINT]


def test_shutting_down_right_after_back_does_not_signal_again():
    largada, fabrica, _, _ = _largada()
    largada.passo(START)
    largada.passo(0)
    largada.passo(BACK)
    assert largada.encerra() is True
    assert fabrica.corridas[0].sinais == [signal.SIGINT]


def test_a_run_that_ignores_the_stop_is_never_killed():
    # Matar o runtime pula a limpeza dele, e o PCA9685 continua mandando o
    # ultimo pulso: o carro seguiria andando. Melhor avisar e deixar a chave
    # do carro resolver.
    largada, fabrica, _, eventos = _largada(obedece=False)
    largada.passo(START)
    assert largada.encerra() is False
    corrida = fabrica.corridas[0]
    assert not corrida.matou
    assert set(corrida.sinais) == {signal.SIGINT}
    assert eventos[-1] == "nao_parou"


# ------------------------------------------------------- comando da corrida

def _flags_do_runtime():
    with io.open(_RUNTIME, encoding="utf-8") as fh:
        arvore = ast.parse(fh.read(), filename=_RUNTIME)
    flags = set()
    for no in ast.walk(arvore):
        if (isinstance(no, ast.Call) and isinstance(no.func, ast.Attribute)
                and no.func.attr == "add_argument"):
            for arg in no.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    flags.add(arg.value)
    return flags


def test_the_run_command_is_the_one_typed_by_hand_today():
    # docs/ESTADO_IA.md, secao 2.
    assert comando_da_corrida(
        "python3", "hardware/jetson_runtime.py",
        engine="models/driving_grade_v5d1.engine",
        config="models/driving_grade_v5d1.json",
        out="runs/x", cruise_us=1610, beco_dist=0.5) == [
            "python3", "hardware/jetson_runtime.py",
            "--engine", "models/driving_grade_v5d1.engine",
            "--config", "models/driving_grade_v5d1.json",
            "--out", "runs/x", "--cruise-us", "1610", "--beco-dist", "0.5"]


def test_extra_flags_go_to_the_runtime_untouched():
    comando = comando_da_corrida("python3", "rt.py", engine="e", config="c",
                                 out="o", cruise_us=1610, beco_dist=0.5,
                                 extras=["--seconds", "120"])
    assert comando[-2:] == ["--seconds", "120"]


def test_every_flag_in_the_run_command_exists_in_the_runtime():
    # Um nome de flag errado so apareceria no carro, na hora de largar.
    comando = comando_da_corrida("python3", _RUNTIME, engine="e", config="c",
                                 out="o", cruise_us=1610, beco_dist=0.5)
    flags = {parte for parte in comando if parte.startswith("--")}
    assert flags and flags <= _flags_do_runtime()


def test_each_run_logs_to_a_folder_named_after_the_model_and_the_time():
    quando = datetime.datetime(2026, 10, 13, 14, 25, 1)
    assert pasta_da_corrida("models/driving_grade_v5d1.engine", quando) == \
        "runs/driving_grade_v5d1_2026-10-13_142501"


# -------------------------------------------------------------------- laco

_REPOUSO = bytes.fromhex("0014000000000000000000000000101c30640010")
_COM_START = b"\x00\x14\x10\x00" + _REPOUSO[4:]
_COM_BACK = b"\x00\x14\x20\x00" + _REPOUSO[4:]


class ControleRoteirizado(object):
    """Entrega relatorios (ou ``None``, ou um erro) na ordem do roteiro."""

    def __init__(self, roteiro):
        self._roteiro = list(roteiro)

    def le(self, timeout_ms):
        item = self._roteiro.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def test_start_then_back_on_the_controller_runs_and_stops_one_run():
    largada, fabrica, _, _ = _largada()
    controle = ControleRoteirizado(
        [_COM_START, _REPOUSO, None, _COM_BACK, ControlePerdido("fim do roteiro")])
    laco(controle, largada)
    assert len(fabrica.corridas) == 1
    assert fabrica.corridas[0].sinais == [signal.SIGINT]
    assert fabrica.corridas[0].codigo == 0


def test_losing_the_adapter_stops_the_run():
    # Sem adaptador nao ha mais botao de parar: a corrida nao continua sozinha.
    largada, fabrica, _, _ = _largada()
    controle = ControleRoteirizado([_COM_START, ControlePerdido("No such device")])
    assert laco(controle, largada) == "controle perdido"
    assert fabrica.corridas[0].sinais == [signal.SIGINT]


def test_ctrl_c_in_the_terminal_does_not_signal_the_run_a_second_time():
    largada, fabrica, _, _ = _largada(acaba_sozinha=True)
    controle = ControleRoteirizado([_COM_START, KeyboardInterrupt()])
    assert laco(controle, largada) == "interrompido"
    assert fabrica.corridas[0].sinais == []
