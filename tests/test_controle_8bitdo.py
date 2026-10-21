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
import signal
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
LB_RB = 0x0300            # LB (bit 8) + RB (bit 9): desliga o Jetson
ESQUERDA, DIREITA = 245, 368   # 1200 e 1800 us: o volante a mao vai ate +-300 us
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
    """O controle: entrega o roteiro; depois, silencio ate ``fim_s``, e some.

    No roteiro, um ``float`` e silencio de verdade por esse tanto de segundos.
    ``aberturas`` conta as mensagens de abertura recebidas (``Controle.abre``).
    """

    def __init__(self, roteiro, fim_s=0.3):
        self._roteiro = list(roteiro)
        self._fim = None
        self._fim_s = fim_s
        self.prazos = []
        self.aberturas = 0

    def is_kernel_driver_active(self, interface):
        return False

    def ctrl_transfer(self, *a):
        self.aberturas += 1

    def write(self, *a):
        pass

    def read(self, endpoint, tamanho, timeout=None):
        self.prazos.append(timeout)
        if self._roteiro:
            item = self._roteiro.pop(0)
            if isinstance(item, float):
                _DORME_DE_VERDADE(item)
                raise NADA
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


def _carrega(monkeypatch, placa, adaptador, runtime=None, euid=1000, pode_desligar=True,
             desliga_ok=True, reabre_s=999.0):
    runtime = runtime or Runtime(placa)
    # O que o programa fez para desligar o Jetson, na ordem: ("sinal", acao) e
    # ("desliga", quantas escritas a placa tinha recebido ate ali).
    runtime.eventos = []
    monkeypatch.setitem(sys.modules, "Adafruit_PCA9685",
                        types.SimpleNamespace(PCA9685=placa))
    monkeypatch.setitem(sys.modules, "jetson_runtime", runtime.modulo())
    spec = importlib.util.spec_from_file_location("controle_8bitdo", _SCRIPT)
    cm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cm)
    monkeypatch.setattr(cm, "acha_adaptador", lambda: adaptador)
    monkeypatch.setattr(cm.time, "sleep", lambda s: None)   # armar o ESC, sair
    monkeypatch.setattr(cm.os, "geteuid", lambda: euid, raising=False)
    monkeypatch.setattr(cm, "DESLIGA_S", 0.05)                # segurar 3 s, encurtado
    # Reenviar a abertura ao controle mudo: desligado, salvo nos testes dela.
    monkeypatch.setattr(cm, "REABRE_S", reabre_s)
    monkeypatch.setattr(cm, "pode_desligar", lambda: pode_desligar)
    monkeypatch.setattr(cm, "desliga_o_jetson", lambda: runtime.eventos.append(
        ("desliga", len(placa.escritas))) or desliga_ok)
    monkeypatch.setattr(cm.signal, "signal",
                        lambda sinal, acao: runtime.eventos.append(("sinal", sinal, acao)))
    return cm, runtime


def _roda(monkeypatch, placa, adaptador, runtime=None, args=(), **kw):
    cm, runtime = _carrega(monkeypatch, placa, adaptador, runtime, **kw)
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
    assert pares[:10] == [
        (CENTRO, NEUTRO),      # ao abrir
        (ESQUERDA, NEUTRO), (DIREITA, NEUTRO), (CENTRO, NEUTRO),   # pronto
        (CENTRO, ACELERADO),   # a mao, RT no fundo
        (CENTRO, NEUTRO),      # Start: estaciona antes do primeiro quadro
        MODELO, MODELO, MODELO,   # o modelo dirige: quadro do Start + dois
        (CENTRO, NEUTRO),      # Back: estaciona na hora
    ]
    # Depois: manual TRAVADO (o RT ficou apertado), saida estacionada, canais soltos.
    assert set(pares[10:-1]) == {(CENTRO, NEUTRO)}
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
    # Abrir, o aviso de pronto (3), o Start e, no lugar do quadro, estacionado.
    assert placa.pares()[5] == (CENTRO, NEUTRO)


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


def test_an_adapter_held_by_the_boot_service_says_how_to_stop_it(monkeypatch):
    # Com o servico do boot rodando (hardware/servico_carro.py), uma segunda
    # copia aberta a mao acha o adaptador ocupado. A mensagem antiga mandava
    # conferir a regra de permissao, que nao tem nada a ver.
    class Ocupado(Adaptador):
        def write(self, *a):
            raise OSError(16, "Resource busy")

    placa = Placa()
    cm, runtime = _carrega(monkeypatch, placa, Ocupado([]))
    with pytest.raises(SystemExit) as saida:
        cm.main(list(_ARGS))
    assert "sudo systemctl stop carro-controle" in str(saida.value.code)
    assert "permissao" not in str(saida.value.code)
    assert placa.escritas == [] and runtime.aberturas == 0


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


# ------------------------------------------------------ aviso de pronto

def test_when_ready_the_wheels_go_one_way_the_other_and_back(monkeypatch, capsys):
    # Pedido do Rafael em 2026-10-21: sem monitor, saber quando ligar o
    # controle. Uma ida e volta = pronto; duas = desligando.
    placa = Placa()
    _roda(monkeypatch, placa, Adaptador([], fim_s=0.05))
    pares = placa.pares()
    esquerda = pares.index((ESQUERDA, NEUTRO))
    assert pares[esquerda:esquerda + 3] == [(ESQUERDA, NEUTRO), (DIREITA, NEUTRO),
                                            (CENTRO, NEUTRO)]
    assert [s for s, _ in pares].count(ESQUERDA) == 1
    # Depois de armar o ESC em neutro, antes de ouvir o controle.
    assert pares[:esquerda] and set(pares[:esquerda]) == {(CENTRO, NEUTRO)}
    assert "PRONTO" in capsys.readouterr().out


# ------------------------------------------------- desligar pelo controle

def _desligamentos(runtime):
    return [e for e in runtime.eventos if e[0] == "desliga"]


def test_holding_lb_rb_parks_waves_the_wheels_and_shuts_the_jetson_down(monkeypatch, capsys):
    placa = Placa()
    codigo, runtime = _roda(monkeypatch, placa, Adaptador(
        [_relatorio(rt=255), _relatorio(botoes=LB_RB)], fim_s=1.0))
    saida = capsys.readouterr().out
    assert codigo == 0 and "desligando o Jetson" in saida
    (_, escritas_antes), = _desligamentos(runtime)
    # Antes do comando: estacionou e virou as rodas para os dois lados, com o
    # ESC em neutro o tempo todo depois de soltar o RT.
    pares = [(placa.escritas[i][1], placa.escritas[i + 1][1])
             for i in range(0, escritas_antes - 1, 2)]
    depois_do_rt = pares[max(i for i, (_, e) in enumerate(pares) if e == ACELERADO) + 1:]
    assert {e for _, e in depois_do_rt} == {NEUTRO}
    # Duas idas e voltas: diferente do aviso de pronto, que e uma.
    assert [s for s, _ in depois_do_rt].count(ESQUERDA) == 2
    assert [s for s, _ in depois_do_rt].count(DIREITA) == 2
    assert depois_do_rt[-1] == (CENTRO, NEUTRO)
    # O desligamento do sistema manda SIGINT ao servico: ignorado, para a saida
    # segura do programa terminar inteira.
    assert runtime.eventos.index(("sinal", signal.SIGINT, signal.SIG_IGN)) <         runtime.eventos.index(_desligamentos(runtime)[0])
    assert placa.escritas[-2:] == [(SERVO, 0), (ESC, 0)]


def test_without_the_sudo_rule_the_wheels_do_not_wave_and_driving_goes_on(monkeypatch, capsys):
    placa = Placa()
    codigo, runtime = _roda(monkeypatch, placa, Adaptador([_relatorio(botoes=LB_RB)]),
                            pode_desligar=False)
    saida = capsys.readouterr().out
    assert _desligamentos(runtime) == []
    assert "INDISPONIVEL" in saida and "servico_carro.py instala" in saida
    # So o aviso de pronto (uma ida e volta), nenhum aviso de desligar.
    assert [s for s, _ in placa.pares()].count(ESQUERDA) == 1
    assert codigo == 1 and "controle perdido" in saida      # o fim do roteiro


def test_lb_rb_during_a_run_does_not_shut_down(monkeypatch, capsys):
    placa = Placa()
    codigo, runtime = _roda(monkeypatch, placa, Adaptador(
        [_relatorio(botoes=START), _relatorio(botoes=START | LB_RB)]))
    assert _desligamentos(runtime) == []
    assert codigo == 1


def test_a_failed_shutdown_keeps_the_program_and_ctrl_c_working(monkeypatch, capsys):
    placa = Placa()
    codigo, runtime = _roda(monkeypatch, placa, Adaptador([_relatorio(botoes=LB_RB)]),
                            desliga_ok=False)
    saida = capsys.readouterr().out
    assert len(_desligamentos(runtime)) == 1
    assert "DESLIGAR FALHOU" in saida
    assert runtime.eventos[-1] == ("sinal", signal.SIGINT, signal.default_int_handler)
    assert codigo == 1 and "controle perdido" in saida


# ------------------------------------------- controle ligado depois do programa

def test_a_silent_controller_gets_the_opening_again(monkeypatch, capsys):
    # Visto no carro em 2026-10-21: ligado depois do programa, o controle
    # ficava mudo ate reiniciar o programa.
    placa = Placa()
    adaptador = Adaptador([], fim_s=0.3)
    codigo, _ = _roda(monkeypatch, placa, adaptador, reabre_s=0.05)
    assert adaptador.aberturas >= 3          # a da abertura do programa e mais
    assert adaptador.aberturas <= 8          # uma a cada 0,05 s em 0,3 s, nao a cada giro
    assert "reenviei a abertura" in capsys.readouterr().out
    assert codigo == 1                        # terminou pelo fim do roteiro


def test_after_the_opening_is_resent_the_throttle_waits_for_release(monkeypatch, capsys):
    # O adaptador guarda o ultimo estado do controle; o que ele repete depois
    # de uma reabertura nao foi visto. Um RT "preso" nao pode acelerar.
    placa = Placa()
    adaptador = Adaptador([0.1, _relatorio(rt=255)], fim_s=0.2)
    _roda(monkeypatch, placa, adaptador, reabre_s=0.05)
    assert ACELERADO not in [e for _, e in placa.pares()]
    assert "PARADO" in capsys.readouterr().out


def test_a_report_after_the_opening_is_announced(monkeypatch, capsys):
    placa = Placa()
    adaptador = Adaptador([0.1, _relatorio(), _relatorio(rt=255), _relatorio(), _relatorio(rt=255)],
                          fim_s=0.05)
    _roda(monkeypatch, placa, adaptador, reabre_s=0.05)
    saida = capsys.readouterr().out
    assert "controle respondeu" in saida
    # Soltou e apertou de novo: anda.
    assert ACELERADO in [e for _, e in placa.pares()]


def test_during_a_run_the_opening_is_not_resent(monkeypatch, capsys):
    # Com o controle desligado cada tentativa pode travar o laco por ate ~2 s.
    placa = Placa()
    adaptador = Adaptador([_relatorio(botoes=START)], fim_s=0.3)
    _roda(monkeypatch, placa, adaptador, reabre_s=0.05)
    assert adaptador.aberturas == 1


def test_a_failed_reopening_does_not_end_the_program(monkeypatch, capsys):
    class NaoResponde(Adaptador):
        def ctrl_transfer(self, *a):
            Adaptador.ctrl_transfer(self, *a)
            if self.aberturas > 1:
                raise OSError(110, "Operation timed out")

    placa = Placa()
    adaptador = NaoResponde([], fim_s=0.3)
    codigo, _ = _roda(monkeypatch, placa, adaptador, reabre_s=0.05)
    saida = capsys.readouterr().out
    assert adaptador.aberturas >= 3
    assert codigo == 1 and "controle perdido" in saida   # o fim do roteiro, nao o erro


def test_a_controller_in_use_is_never_reopened(monkeypatch, capsys):
    # Reabrir trava o acelerador: com alguem dirigindo, o carro pararia.
    placa = Placa()
    roteiro = []
    for _ in range(15):                    # ~0,15 s falando, sem 0,1 s de silencio
        roteiro += [_relatorio(rt=255), 0.01]
    adaptador = Adaptador(roteiro, fim_s=0.05)
    _roda(monkeypatch, placa, adaptador, reabre_s=0.1)
    assert adaptador.aberturas == 1
    assert "PARADO" not in capsys.readouterr().out
