"""O programa do controle inteiro (hardware/controle_8bitdo.py), com placa,
controle e jetson_runtime de mentira.

Desde 2026-10-17 a corrida roda DENTRO do programa: modelo, camera e LiDAR sao
abertos uma vez, e o Start so comeca a chamar o laco de direcao. Estes testes
rodam o ``main()`` de verdade e conferem o que chega na placa, quantas vezes os
componentes sao abertos, e o que acontece nos casos que ja aconteceram no
carro: um jumper do I2C que soltou por um instante (Errno 121) e a corrida que,
com sudo, nao achou as bibliotecas do CUDA.
"""
import argparse
import importlib.util
import os
import struct
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
ACELERADO = 331           # 1620 us: RT no fundo com o teto padrao
MODELO = (296, 329)       # o laco de mentira manda 1450 us no servo e 1610 no ESC
SERVO, ESC = 15, 12
START, BACK = 0x10, 0x20
NADA = OSError(110, "Operation timed out")   # o controle nao mandou nada


def _relatorio(rt=0, ex=0, botoes=0):
    return (bytes([0x00, 0x14, botoes & 0xFF, botoes >> 8, 0, rt])
            + struct.pack("<4h", ex, 0, 0, 0) + bytes.fromhex("101c30640010"))


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
        self.prazos = []

    def is_kernel_driver_active(self, interface):
        return False

    def ctrl_transfer(self, *a):
        pass

    def write(self, *a):
        pass

    def read(self, endpoint, tamanho, timeout=None):
        self.prazos.append(timeout)
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
        raise NADA


class Runtime(object):
    """O jetson_runtime.py: so o que o programa do controle usa dele."""

    def __init__(self, placa, falha_no_laco=None):
        self.placa = placa
        self.falha_no_laco = falha_no_laco
        self.aberturas = 0
        self.lacos = []
        self.fechados = []
        self.descartes = 0

    def modulo(self):
        rt = self

        def cria_parser():
            p = argparse.ArgumentParser()
            for flag in ("--engine", "--config", "--out", "--self-occlusion"):
                p.add_argument(flag, default="")
            p.add_argument("--cruise-us", type=int)
            p.add_argument("--beco-dist", type=float)
            p.add_argument("--seconds", type=float, default=60.0)
            p.add_argument("--scale", type=float, default=12.0)
            p.add_argument("--jpeg-every", type=int, default=10)
            p.add_argument("--log-inputs", action="store_true")
            return p

        class Componente(object):
            def __init__(self, nome):
                self.nome = nome

            def descarta_fila(self):
                rt.descartes += 1

            def close(self):
                rt.fechados.append(self.nome)

        def abre_componentes(a, cfg):
            rt.aberturas += 1
            return Componente("camera"), Componente("lidar"), Componente("engine")

        class Atuador(object):
            def __init__(self):
                rt.placa(0x40, 1)

            def set_servo_us(self, us):
                rt.placa.set_pwm(SERVO, 0, int(us / 4.8828125))

            def set_esc_us(self, us):
                rt.placa.set_pwm(ESC, 0, int(us / 4.8828125))

        class Laco(object):
            def __init__(self, atuador, logger):
                self.atuador = atuador
                self.logger = logger

            def step(self):
                if rt.falha_no_laco is not None:
                    raise rt.falha_no_laco
                self.atuador.set_servo_us(1450)
                self.atuador.set_esc_us(1610)
                return {"steer": -0.1, "servo_us": 1450, "esc_us": 1610, "fim": False,
                        "blocked": False, "blind": False, "stale_lidar": False,
                        "liberado": True, "lidar_atrasado": False, "has_scan": True}

        def cria_laco(a, cfg, max_range, arcos, camera, lidar, engine, actuator, logger):
            rt.lacos.append(Laco(actuator, logger))
            return rt.lacos[-1]

        return types.SimpleNamespace(
            cria_parser=cria_parser,
            load_model_config=lambda caminho: {"fov_deg": 180, "n_sectors": 72},
            car_max_range=lambda cfg, scale: 1.0,
            parse_arcs=lambda texto: [],
            abre_componentes=abre_componentes,
            Pca9685Actuator=Atuador,
            meta_da_corrida=lambda a, cfg, max_range: {"engine": a.engine},
            cria_laco=cria_laco)


def _carrega(monkeypatch, placa, adaptador, runtime=None, euid=1000):
    runtime = runtime or Runtime(placa)
    monkeypatch.setitem(sys.modules, "Adafruit_PCA9685",
                        types.SimpleNamespace(PCA9685=placa))
    monkeypatch.setitem(sys.modules, "jetson_runtime", runtime.modulo())
    spec = importlib.util.spec_from_file_location("controle_8bitdo", _SCRIPT)
    cm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cm)
    monkeypatch.setattr(cm, "acha_adaptador", lambda: adaptador)
    monkeypatch.setattr(cm.time, "sleep", lambda s: None)   # armar o ESC, sair
    monkeypatch.setattr(cm.os, "geteuid", lambda: euid, raising=False)
    return cm, runtime


def _roda(monkeypatch, placa, adaptador, runtime=None, args=()):
    cm, runtime = _carrega(monkeypatch, placa, adaptador, runtime)
    return cm.main(list(_ARGS) + list(args)), runtime


# ------------------------------------------------------- manual + corrida

def test_driving_by_hand_then_a_run_then_back_to_the_hand(monkeypatch, capsys):
    placa = Placa()
    adaptador = Adaptador([_relatorio(rt=255),                  # a mao: acelera
                           _relatorio(rt=255, botoes=START),    # Start com RT apertado
                           NADA, NADA,                          # dois quadros do modelo
                           _relatorio(botoes=BACK)])            # para a corrida
    codigo, runtime = _roda(monkeypatch, placa, adaptador)
    assert codigo == 1 and "controle perdido" in capsys.readouterr().out
    pares = placa.pares()
    assert pares[:7] == [
        (CENTRO, NEUTRO),      # ao abrir
        (CENTRO, ACELERADO),   # a mao, RT no fundo
        (CENTRO, NEUTRO),      # Start: estaciona antes do primeiro quadro
        MODELO, MODELO, MODELO,   # o modelo dirige: quadro do Start + dois
        (CENTRO, NEUTRO),      # Back: estaciona na hora
    ]
    # Depois: manual TRAVADO (o RT ficou apertado), saida estacionada, canais soltos.
    assert set(pares[7:-1]) == {(CENTRO, NEUTRO)}
    assert pares[-1] == (0, 0)
    assert runtime.descartes == 1          # fila do LiDAR jogada fora antes da corrida


def test_model_camera_and_lidar_are_opened_once_for_many_runs(monkeypatch, capsys):
    # O ponto da mudanca de 2026-10-17: largar nao carrega nada de novo.
    placa = Placa()
    adaptador = Adaptador([_relatorio(botoes=START), _relatorio(botoes=BACK),
                           _relatorio(botoes=START), _relatorio(botoes=BACK)])
    _, runtime = _roda(monkeypatch, placa, adaptador)
    assert runtime.aberturas == 1
    assert len(runtime.lacos) == 2         # uma corrida nova (estado zerado) por Start
    assert runtime.fechados == ["camera", "lidar"]   # fechados uma vez, na saida


def test_during_a_run_the_controller_is_read_without_waiting(monkeypatch, capsys):
    # Esperar 20 ms pelo controle a cada quadro atrasaria o modelo.
    placa = Placa()
    adaptador = Adaptador([_relatorio(botoes=START), NADA, NADA, _relatorio(botoes=BACK)])
    _roda(monkeypatch, placa, adaptador)
    assert adaptador.prazos[:5] == [20, 1, 1, 1, 20]


def test_an_error_in_the_model_stops_the_run_but_not_the_program(monkeypatch, capsys):
    placa = Placa()
    runtime = Runtime(placa, falha_no_laco=OSError(121, "Remote I/O error"))
    adaptador = Adaptador([_relatorio(botoes=START)])
    codigo, _ = _roda(monkeypatch, placa, adaptador, runtime)
    saida = capsys.readouterr().out
    assert codigo == 1 and "controle perdido" in saida    # seguiu ate o controle sumir
    assert "Remote I/O error" in saida
    assert placa.pares()[2] == (CENTRO, NEUTRO)           # estacionado no lugar do quadro


def test_ctrl_c_during_a_run_parks_the_car_and_exits(monkeypatch, capsys):
    placa = Placa()
    adaptador = Adaptador([_relatorio(botoes=START), KeyboardInterrupt()])
    codigo, runtime = _roda(monkeypatch, placa, adaptador)
    assert codigo == 0
    assert placa.pares()[-2:] == [(CENTRO, NEUTRO), (0, 0)]
    assert runtime.fechados == ["camera", "lidar"]


def test_without_grava_nothing_is_recorded(monkeypatch, capsys):
    placa = Placa()
    adaptador = Adaptador([_relatorio(botoes=START), _relatorio(botoes=BACK)])
    _, runtime = _roda(monkeypatch, placa, adaptador)
    assert type(runtime.lacos[0].logger).__name__ == "LogNulo"


def test_with_grava_each_run_gets_its_log_folder(monkeypatch, capsys):
    placa = Placa()
    adaptador = Adaptador([_relatorio(botoes=START), _relatorio(botoes=BACK)])
    pastas = []

    class LogDeVerdade(object):
        def __init__(self, pasta, meta, **como):
            pastas.append(pasta)

        def close(self):
            pass

    cm, runtime = _carrega(monkeypatch, placa, adaptador)
    monkeypatch.setattr(cm, "RunLogger", LogDeVerdade)
    cm.main(list(_ARGS) + ["--grava"])
    assert len(pastas) == 1 and pastas[0].startswith("runs/driving_grade_v5d1_")


def test_running_as_root_is_refused_before_touching_anything(monkeypatch):
    abertos = []

    def acha():
        abertos.append(True)
        raise ControlePerdido("adaptador de teste")

    placa = Placa()
    cm, runtime = _carrega(monkeypatch, placa, None, euid=0)
    monkeypatch.setattr(cm, "acha_adaptador", acha)
    with pytest.raises(SystemExit) as saida:
        cm.main(list(_ARGS))
    assert "sudo" in str(saida.value.code)
    assert abertos == [] and runtime.aberturas == 0


# ---------------------------------------------------------------- I2C

def test_a_momentary_i2c_failure_does_not_end_the_drive(monkeypatch, capsys):
    placa = Placa(falha=lambda n: n == 6)          # uma escrita no meio da direcao
    codigo, _ = _roda(monkeypatch, placa, Adaptador([_relatorio(rt=255)]))
    saida = capsys.readouterr().out
    assert codigo == 1 and "controle perdido" in saida   # terminou pelo fim do roteiro
    assert "DESLIGUE A CHAVE" not in saida
    assert placa.escritas[-4:-2] == [(SERVO, NEUTRO), (ESC, NEUTRO)]
    assert placa.escritas[-2:] == [(SERVO, 0), (ESC, 0)]


def test_a_board_that_stops_answering_ends_the_drive_and_says_why(monkeypatch, capsys):
    placa = Placa(falha=lambda n: n > 6)           # o fio soltou e ficou solto
    codigo, _ = _roda(monkeypatch, placa, Adaptador([_relatorio(rt=255)], fim_s=5.0))
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
