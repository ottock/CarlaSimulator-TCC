"""O programa do controle inteiro (hardware/controle_8bitdo.py), com placa,
controle e runtime de mentira.

Por que existe: o programa junta o controle manual e a largada da corrida
(2026-10-17), e os dois mexem no mesmo PCA9685. Estes testes rodam o ``main()``
de verdade e conferem o que chega na placa -- inclusive o que NAO pode chegar
enquanto a corrida e do runtime. Tambem cobrem o que ja aconteceu no carro: um
jumper do I2C que soltou por um instante (Errno 121) e a corrida que, com
sudo, nao achou as bibliotecas do CUDA.
"""
import importlib.util
import os
import signal
import struct
import subprocess
import sys
import time
import types
from array import array

import pytest

from ai.car.gamepad import ControlePerdido

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_REPO, "hardware", "controle_8bitdo.py")
_DORME_DE_VERDADE = time.sleep
# Qualquer arquivo que exista serve: o programa so confere que a engine existe.
_ARGS = ["--engine", "models/driving_grade_v5d1.onnx"]

# Passos de 12 bits a 50 Hz (us / 4,8828125), calculados a mao.
CENTRO = NEUTRO = 307     # 1500 us
ACELERADO = 333           # 1630 us: RT no fundo com o teto padrao
SERVO, ESC = 15, 12


def _relatorio(rt=0, ex=0, botoes=0):
    return (bytes([0x00, 0x14, botoes & 0xFF, botoes >> 8, 0, rt])
            + struct.pack("<4h", ex, 0, 0, 0) + bytes.fromhex("101c30640010"))


START, BACK = 0x10, 0x20


class Placa(object):
    """O PCA9685: guarda cada escrita; ``falha(n)`` decide se a n-esima falha."""

    def __init__(self, falha=lambda n: False):
        self.escritas = []
        self._falha = falha
        self._n = 0

    def __call__(self, address, busnum):     # Adafruit_PCA9685.PCA9685(...)
        return self

    def set_pwm_freq(self, frequencia):
        pass

    def set_pwm(self, canal, on, off):
        self._n += 1
        if self._falha(self._n):
            raise OSError(121, "Remote I/O error")
        self.escritas.append((canal, off))

    def pares(self):
        """As escritas agrupadas em (servo, esc), na ordem."""
        return [(self.escritas[i][1], self.escritas[i + 1][1])
                for i in range(0, len(self.escritas) - 1, 2)]


class Adaptador(object):
    """O controle: entrega o roteiro; depois, silencio ate ``fim_s``, e some."""

    def __init__(self, roteiro, fim_s=0.3):
        self._roteiro = list(roteiro)
        self._fim = None
        self._fim_s = fim_s

    def is_kernel_driver_active(self, interface):
        return False

    def ctrl_transfer(self, *a):
        pass

    def write(self, *a):
        pass

    def read(self, endpoint, tamanho, timeout=None):
        if self._roteiro:
            item = self._roteiro.pop(0)
            if isinstance(item, BaseException):
                raise item
            return array("B", item)
        if self._fim is None:
            self._fim = time.monotonic() + self._fim_s
        _DORME_DE_VERDADE(0.005)
        if time.monotonic() > self._fim:
            raise OSError(19, "No such device")
        raise OSError(110, "Operation timed out")


class Runtime(object):
    """O jetson_runtime.py lancado pelo Popen: vivo ate receber o Ctrl+C e ser esperado."""

    lancados = []
    acaba_sozinho = False     # o Ctrl+C do terminal chegou nele direto

    def __init__(self, comando, **como):
        self.comando = comando
        self.como = como
        self.sinais = []
        self.codigo = None
        Runtime.lancados.append(self)

    def poll(self):
        return self.codigo

    def send_signal(self, sinal):
        self.sinais.append(sinal)

    def wait(self, timeout=None):
        if self.sinais or Runtime.acaba_sozinho:
            self.codigo = 0
            return 0
        raise subprocess.TimeoutExpired("runtime", timeout)


def _carrega(monkeypatch, placa, adaptador, euid=1000):
    monkeypatch.setitem(sys.modules, "Adafruit_PCA9685",
                        types.SimpleNamespace(PCA9685=placa))
    spec = importlib.util.spec_from_file_location("controle_8bitdo", _SCRIPT)
    cm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cm)
    monkeypatch.setattr(cm, "acha_adaptador", lambda: adaptador)
    monkeypatch.setattr(cm.time, "sleep", lambda s: None)   # armar o ESC, sair
    monkeypatch.setattr(cm.os, "geteuid", lambda: euid, raising=False)
    Runtime.lancados = []
    Runtime.acaba_sozinho = False
    monkeypatch.setattr(cm.subprocess, "Popen", Runtime)
    return cm


def _roda(monkeypatch, placa, adaptador):
    return _carrega(monkeypatch, placa, adaptador).main(list(_ARGS))


# ------------------------------------------------------- manual + corrida

def test_driving_by_hand_then_start_and_back_run_one_autonomous_run(monkeypatch, capsys):
    placa = Placa()
    roteiro = [_relatorio(rt=255),                     # a mao: acelera
               _relatorio(rt=255, botoes=START),       # Start com o RT apertado
               _relatorio(rt=255, ex=32767),           # durante a corrida
               _relatorio(botoes=BACK)]                # para a corrida
    codigo = _roda(monkeypatch, placa, Adaptador(roteiro))
    assert codigo == 1 and "controle perdido" in capsys.readouterr().out
    assert len(Runtime.lancados) == 1
    corrida = Runtime.lancados[0]
    assert corrida.sinais == [signal.SIGINT]
    assert corrida.comando[1:3] == [os.path.join("hardware", "jetson_runtime.py"),
                                    "--engine"]
    assert corrida.como["cwd"] == _REPO
    assert placa.pares() == [
        (CENTRO, NEUTRO),      # ao abrir
        (CENTRO, ACELERADO),   # a mao, RT no fundo
        (CENTRO, NEUTRO),      # Start: estaciona ANTES de lancar
        # ... nada durante a corrida: a placa e do runtime ...
        (CENTRO, NEUTRO),      # saida: estaciona DEPOIS de a corrida acabar
        (0, 0),                # e solta os canais
    ]


def test_ctrl_c_during_a_run_waits_for_it_without_a_second_signal(monkeypatch, capsys):
    # O Ctrl+C do terminal chega ao runtime tambem (mesmo grupo de processos);
    # um segundo sinal poderia cair no meio da limpeza dele.
    placa = Placa()
    cm = _carrega(monkeypatch, placa, Adaptador([_relatorio(botoes=START),
                                                 KeyboardInterrupt()]))
    Runtime.acaba_sozinho = True
    assert cm.main(list(_ARGS)) == 0
    assert Runtime.lancados[0].sinais == []
    assert placa.pares()[-2:] == [(CENTRO, NEUTRO), (0, 0)]


def test_running_as_root_is_refused_before_touching_the_controller(monkeypatch):
    abertos = []

    def acha():
        abertos.append(True)
        raise ControlePerdido("adaptador de teste")

    cm = _carrega(monkeypatch, Placa(), None, euid=0)
    monkeypatch.setattr(cm, "acha_adaptador", acha)
    with pytest.raises(SystemExit) as saida:
        cm.main(list(_ARGS))
    assert "sudo" in str(saida.value.code)
    assert abertos == []


# ---------------------------------------------------------------- I2C

def test_a_momentary_i2c_failure_does_not_end_the_drive(monkeypatch, capsys):
    placa = Placa(falha=lambda n: n == 6)          # uma escrita no meio da direcao
    codigo = _roda(monkeypatch, placa, Adaptador([_relatorio(rt=255)]))
    saida = capsys.readouterr().out
    assert codigo == 1 and "controle perdido" in saida   # terminou pelo fim do roteiro
    assert "DESLIGUE A CHAVE" not in saida
    assert placa.escritas[-4:-2] == [(SERVO, NEUTRO), (ESC, NEUTRO)]
    assert placa.escritas[-2:] == [(SERVO, 0), (ESC, 0)]


def test_a_board_that_stops_answering_ends_the_drive_and_says_why(monkeypatch, capsys):
    placa = Placa(falha=lambda n: n > 6)           # o fio soltou e ficou solto
    codigo = _roda(monkeypatch, placa, Adaptador([_relatorio(rt=255)], fim_s=5.0))
    saida = capsys.readouterr().out
    assert codigo == 1
    assert "I2C" in saida
    # Nenhuma escrita passou depois: o ESC pode ter ficado no ultimo pulso.
    assert "DESLIGUE A CHAVE" in saida


def test_an_unexpected_error_is_not_reported_as_an_interruption(monkeypatch, capsys):
    placa = Placa()
    with pytest.raises(RuntimeError):
        _roda(monkeypatch, placa, Adaptador([RuntimeError("bug")]))
    saida = capsys.readouterr().out
    assert "interrompido" not in saida
    assert placa.escritas[-2:] == [(SERVO, 0), (ESC, 0)]
