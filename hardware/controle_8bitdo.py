#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""O programa do controle 8BitDo: dirige o carro a mao E roda a corrida autonoma.

Por que existe. Eram dois programas: este (o controle_teste.py com o controle
no lugar do teclado) e o largada_controle.py (Start lancava o
jetson_runtime.py como outro programa, Back parava). O Rafael pediu um so em
2026-10-17, e no mesmo dia pediu largar e parar mais rapido. Por isso a corrida
roda AQUI DENTRO: o modelo, a camera e o LiDAR sao abertos uma vez, quando o
programa abre, e o Start so comeca a chamar o laco de direcao do runtime (o
mesmo DriveLoop, com as mesmas opcoes). O controle e lido direto da USB
(docs/CONTROLE_8BITDO.md).

COMANDOS
  Sem corrida, voce dirige:
  analogico esquerdo, para os lados   volante, proporcional. Mesma conversao
                                      e mesmo sentido do runtime, mas com
                                      span 300 us para comecar (o modelo usa
                                      350; --span 350 iguala).
  RT                                  acelerador, de 1600 us (o minimo que move)
                                      ate --max-us (padrao 1620; o maximo
                                      calibrado e 1700). Soltou, para.
  LT                                  re, a 1380 us fixos (o teclado usa 1400).
  Back ou Home                        parada. Para andar de novo, solte os dois
                                      gatilhos e aperte outra vez.

  Start                               estaciona o carro (centro e neutro) e
                                      comeca a corrida. O carro anda depois da
                                      trava de arranque do runtime (~1 s de
                                      sensores em dia).
  Back, com corrida                   para na hora e volta ao manual, parado
                                      ate os gatilhos serem soltos.
  LB + RB juntos por 3 s, sem corrida estaciona, vira as rodas para os dois
                                      lados (o aviso) e DESLIGA O JETSON. So
                                      tire a bateria depois que ele terminar
                                      de desligar. Precisa da regra do sudo
                                      que o hardware/servico_carro.py instala;
                                      sem ela, as rodas nao mexem e nada
                                      acontece.
  Ctrl+C no terminal                  para a corrida, se houver, estaciona o
                                      carro e sai.

  A corrida acaba sozinha no limite de tempo (--seconds, padrao 60) ou se o
  laco do modelo der erro (camera, I2C...): o carro e estacionado e volta o
  manual. No fim da pista o carro para, mas a corrida segue ate Back ou o fim
  do tempo.

PROTECOES
  - ESC_ARMADO deste arquivo vale para o modo manual: False, o ESC nunca sai do
    neutro a mao. A corrida segue o ESC_ARMADO do jetson_runtime.py.
  - Laco travado por mais de 0,25 s: ESC em neutro naquele quadro (manual).
  - Controle mudo por mais de --silencio-s (0,5 s) com o carro andando a mao:
    para. O adaptador nao avisa quando o controle desliga e o ultimo estado
    fica valendo; sem isto um RT "preso" seguiria acelerando. O preco: segurar
    o RT no fundo com o analogico parado tambem e silencio, e pode parar o
    carro numa reta. Se acontecer demais, aumente --silencio-s.
  - Adaptador sumiu da USB: para a corrida, se houver, estaciona e sai.
  - Fio do I2C que solta por um instante (visto em 2026-10-14, Errno 121), no
    modo manual: o programa nao cai, o comando vai de novo no quadro seguinte
    e a tela conta as falhas. Falhando por 0,1 s seguidos, ele sai, tenta o
    neutro 10 vezes e, se nao conseguir, manda DESLIGAR A CHAVE: o PCA9685
    continua mandando o ultimo pulso sozinho. Na corrida, uma falha encerra a
    corrida. O conserto de verdade e o fio.
  - Os botoes valem no instante em que sao apertados: um Start "preso" depois
    de o controle desligar nao larga corrida atras de corrida.
  - Saida sempre tentando o estado seguro: servo no centro, ESC em neutro.

O QUE ELE NAO TEM / CUSTA
  - Parada frontal do LiDAR no modo manual. Quem dirige pode bater na parede.
  - Abrir o programa demora o que antes demorava cada corrida (carregar CUDA,
    camera, LiDAR e modelo), uma vez. Depois a camera e o LiDAR ficam ligados
    o tempo todo, inclusive dirigindo a mao.
  - Sem --grava, a corrida nao grava nada em runs/.

COMO USAR
  cd ~/CarlaSimulator-TCC
  python3 hardware/controle_8bitdo.py
  python3 hardware/controle_8bitdo.py --grava           # cada corrida em runs/
  python3 hardware/controle_8bitdo.py --max-us 1650     # teto mais alto a mao
  python3 hardware/controle_8bitdo.py --seconds 120     # flags que este
                                                        # programa nao conhece
                                                        # vao para o runtime
  Controle ligado (chave traseira em 2.4G, luz do Home fixa). SEM sudo: precisa
  da regra de permissao do adaptador (docs/CONTROLE_8BITDO.md, secao 7). Com
  sudo o programa recusa rodar: em 2026-10-14 o runtime, como root, nao achou
  as bibliotecas do CUDA.

  Para abrir sozinho quando o Jetson liga (feira): hardware/servico_carro.py.
  Com o servico rodando, pare-o antes de abrir esta copia a mao
  (sudo systemctl stop carro-controle).

A corrida dentro do programa foi escrita em 2026-10-17 e ainda NAO rodou no
Jetson: primeira vez com as rodas no ar.
"""
import argparse
import datetime
import errno
import os
import signal
import subprocess
import sys
import time
import types

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

from ai.car.control_map import (ESC_MAX_US, ESC_NEUTRAL_US, STEER_CENTER_US,
                                STEER_SPAN_US, FrameWatchdog)
from ai.car.controle_unico import ControleUnico
from ai.car.desligamento import SEGURAR_S, SegurarParaDesligar
from ai.car.escrita_tolerante import EscritaTolerante, PCA9685Perdido, para_com_insistencia
from ai.car.gamepad import Controle, ControlePerdido, acha_adaptador, estado
from ai.car.largada import argumentos_da_corrida, pasta_da_corrida
from ai.car.manual import ESC_RE_US, Manual
from ai.car.run_log import LogNulo, RunLogger

# ============================ SEGURANCA ============================
# True  -> o RT e o LT movem o carro. RODAS NO AR no primeiro teste.
# False -> o ESC fica em neutro o tempo todo a mao; so o volante responde.
ESC_ARMADO = True
# ===================================================================

I2C_ADDRESS = 0x40
I2C_BUSNUM = 1
PWM_FREQ = 50
SERVO_CHANNEL = 15
ESC_CHANNEL = 12

# Meia-faixa do servo no controle manual. Decisao do Rafael em 2026-10-14:
# comecar em 300 para testar. O modelo usa 350 (control_map.STEER_SPAN_US), que
# e o batente medido (raio de 47 cm); com 300 a curva mais fechada pode nao sair
# a mao. Para o mesmo batente do modelo: --span 350.
SPAN_INICIAL_US = 300

# Teto do acelerador (RT no fundo). Era 1700, o maximo calibrado do ESC; o
# Rafael achou rapido demais e baixou para 1630 (2026-10-14) e depois para 1620.
# A corrida autonoma segue a 1610 fixos. Para subir: --max-us, ate 1700.
MAX_INICIAL_US = 1620

WATCHDOG_S = 0.25
# Prazo de cada leitura do controle. A mao, curto o bastante para o laco girar
# (e o silencio ser percebido) mesmo sem relatorio. Na corrida, quase zero: o
# quadro do modelo ja leva o seu tempo, e esperar o controle o atrasaria.
LEITURA_MS = 20
LEITURA_CORRIDA_MS = 1
STATUS_S = 0.5
STATUS_CORRIDA_S = 1.0

_AVISOS = {
    "parado": "PARADO (Back/Home): solte os gatilhos para andar de novo",
    "silencio": "SEM NOTICIAS DO CONTROLE: parado ate chegar relatorio novo",
}

# Desligar o Jetson pelo controle (LB + RB). Mesmo comando da regra do sudo que
# o hardware/servico_carro.py instala; o sudo confere os argumentos.
DESLIGA_S = SEGURAR_S
COMANDO_DESLIGAR = ["/sbin/shutdown", "-h", "now"]
AVISO_RODAS_S = 0.3
_SEM_REGRA = ("LB + RB: desligar INDISPONIVEL -- o sudo pede senha. Instale a regra: "
              "python3 hardware/servico_carro.py instala")

_FALAS = {
    "largou": "=== CORRIDA COMECOU. Back para. ===",
    "parada": "=== corrida parada (Back). Modo manual: solte os gatilhos para andar. ===",
    "tempo": "=== corrida acabou o tempo. Modo manual: solte os gatilhos para andar. ===",
    "erro": "!!! ERRO NA CORRIDA: carro estacionado, de volta ao manual. !!!",
}


def _passos(microssegundos):
    """us -> passos de 12 bits do PCA9685 a 50 Hz (como no controle_teste.py)."""
    passos = int(microssegundos / (1000000.0 / PWM_FREQ / 4096))
    return max(0, min(4095, passos))


class Atuadores(object):
    """Servo e ESC no PCA9685 para o modo manual. O import fica aqui: a
    biblioteca so existe no Jetson."""

    def __init__(self):
        import Adafruit_PCA9685

        self._pwm = Adafruit_PCA9685.PCA9685(address=I2C_ADDRESS, busnum=I2C_BUSNUM)
        self._pwm.set_pwm_freq(PWM_FREQ)

    def manda(self, servo_us, esc_us):
        self._pwm.set_pwm(SERVO_CHANNEL, 0, _passos(servo_us))
        self._pwm.set_pwm(ESC_CHANNEL, 0, _passos(esc_us))

    def solta(self):
        self._pwm.set_pwm(SERVO_CHANNEL, 0, 0)
        self._pwm.set_pwm(ESC_CHANNEL, 0, 0)


def pode_desligar():
    """``True`` se o sudo deixa desligar sem senha (``sudo -n -l`` nao pergunta)."""
    try:
        return subprocess.call(["sudo", "-n", "-l"] + COMANDO_DESLIGAR,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    except OSError:
        return False


def desliga_o_jetson():
    """Manda o Jetson desligar. ``True`` se o comando foi aceito."""
    try:
        return subprocess.call(["sudo", "-n"] + COMANDO_DESLIGAR) == 0
    except OSError:
        return False


def avisa_com_as_rodas(manda, span_us):
    """Vira as rodas para os dois lados, duas vezes, com o ESC em neutro: o
    sinal, sem monitor, de que o Jetson vai desligar."""
    for lado in (-1, 1, -1, 1, 0):
        manda(STEER_CENTER_US + lado * span_us, ESC_NEUTRAL_US)
        time.sleep(AVISO_RODAS_S)


def carrega_corrida(a, extras):
    """Abre modelo, camera e LiDAR UMA vez, com as opcoes do jetson_runtime.py.

    Returns:
        ``(nova_corrida, fecha_tudo, opcoes_do_runtime)``. ``nova_corrida()``
        prepara uma corrida com estado novo (trava de arranque, fim de pista,
        filtro do esterco) sobre os componentes ja abertos.
    """
    # So existe no Jetson (TensorRT, pycuda): importado aqui, nao no topo.
    import jetson_runtime as jr

    opcoes = jr.cria_parser().parse_args(argumentos_da_corrida(
        engine=a.engine, config=a.config,
        out=pasta_da_corrida(a.engine, datetime.datetime.now()),
        cruise_us=a.cruise_us, beco_dist=a.beco_dist, extras=extras))
    arcos = jr.parse_arcs(opcoes.self_occlusion)
    cfg = jr.load_model_config(opcoes.config)
    max_range = jr.car_max_range(cfg, scale=opcoes.scale)
    camera, lidar, engine = jr.abre_componentes(opcoes, cfg)
    atuador = jr.Pca9685Actuator()

    def nova_corrida():
        # O LiDAR seguiu mandando enquanto o carro andava a mao: a fila e o passado.
        lidar.descarta_fila()
        if a.grava:
            logger = RunLogger(pasta_da_corrida(a.engine, datetime.datetime.now()),
                               meta=jr.meta_da_corrida(opcoes, cfg, max_range),
                               jpeg_every=opcoes.jpeg_every, log_inputs=opcoes.log_inputs)
        else:
            logger = LogNulo()
        laco = jr.cria_laco(opcoes, cfg, max_range, arcos, camera, lidar, engine,
                            atuador, logger)
        return types.SimpleNamespace(passo=laco.step, fecha=logger.close)

    def fecha_tudo():
        for componente in (camera, lidar):
            try:
                componente.close()
            except Exception as erro:
                print("aviso: ao fechar {0}: {1}".format(type(componente).__name__, erro))

    return nova_corrida, fecha_tudo, opcoes


def _linha_da_corrida(tele, quadros, segundos):
    alertas = ""
    if not tele.get("liberado", True):
        alertas += "  AGUARDANDO ARRANQUE"
    if tele.get("fim"):
        alertas += "  FIM DA PISTA"
    if tele.get("blocked"):
        alertas += "  PARADA"
    if tele.get("blind"):
        alertas += "  CAMERA CEGA"
    if tele.get("stale_lidar"):
        alertas += "  LIDAR VELHO"
    if tele.get("lidar_atrasado"):
        alertas += "  LIDAR ATRASADO"
    return "corrida: fps={0:5.1f}  steer={1:+.3f}  servo={2}us  esc={3}us{4}".format(
        quadros / segundos, tele.get("steer", 0.0), tele.get("servo_us"),
        tele.get("esc_us"), alertas)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Controle 8BitDo: dirige a mao; Start larga a corrida, Back para.",
        epilog="Qualquer outra flag vai direto para o jetson_runtime.py "
               "(ex.: --seconds 120).")
    p.add_argument("--max-us", type=int, default=MAX_INICIAL_US,
                   help="Teto do acelerador a mao (RT no fundo), de 1600 a %d us "
                        "(padrao %d)" % (ESC_MAX_US, MAX_INICIAL_US))
    p.add_argument("--span", type=int, default=SPAN_INICIAL_US,
                   help="Meia-faixa do servo a mao, em us (padrao %d; o modelo usa %d)"
                        % (SPAN_INICIAL_US, STEER_SPAN_US))
    p.add_argument("--silencio-s", type=float, default=0.5,
                   help="Segundos sem relatorio do controle que param o carro "
                        "andando a mao (padrao 0.5)")
    p.add_argument("--grava", action="store_true",
                   help="Corrida: grava cada corrida numa pasta nova em runs/ "
                        "(padrao: nao grava nada)")
    p.add_argument("--engine", default="models/driving_grade_v5d1.engine",
                   help="Corrida: arquivo .engine do TensorRT (padrao: o v5d1)")
    p.add_argument("--config", default="models/driving_grade_v5d1.json",
                   help="Corrida: sidecar .json do modelo")
    p.add_argument("--cruise-us", type=int, default=1610,
                   help="Corrida: PWM constante do ESC (padrao 1610)")
    p.add_argument("--beco-dist", type=float, default=0.5,
                   help="Corrida: parada de fim de pista (padrao 0.5)")
    a, extras = p.parse_known_args(argv)

    # Melhor descobrir agora do que depois de esperar o modelo carregar.
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        sys.exit("Nao rode com sudo. Como root a corrida nao acha as bibliotecas do "
                 "CUDA\n(libcurand.so.10, visto em 2026-10-14): o sudo limpa as "
                 "variaveis de ambiente.\nRode como o usuario normal, com a regra "
                 "de permissao do adaptador\n(docs/CONTROLE_8BITDO.md, secao 7).")
    for caminho in (a.engine, a.config):
        if not os.path.exists(os.path.join(_REPO, caminho)):
            sys.exit("nao achei {0} em {1}".format(caminho, _REPO))
    try:
        manual = Manual(max_us=a.max_us, span_us=a.span, silencio_s=a.silencio_s)
    except ValueError as erro:
        sys.exit(str(erro))

    try:
        controle = Controle(acha_adaptador())
        controle.abre()
    except ImportError:
        sys.exit("falta o pyusb: sudo apt install python3-usb")
    except ControlePerdido as erro:
        sys.exit(str(erro))
    except IOError as erro:
        if erro.errno == errno.EBUSY:
            sys.exit("o adaptador do controle ja esta em uso por outro programa: "
                     "provavelmente\nesta mesma copia, aberta no boot pelo servico "
                     "(hardware/servico_carro.py).\nPara rodar a mao: sudo systemctl "
                     "stop carro-controle")
        sys.exit("nao consegui abrir o adaptador: {0}\n"
                 "Sem sudo, e preciso da regra de permissao "
                 "(docs/CONTROLE_8BITDO.md, secao 7).".format(erro))

    atuadores = Atuadores()
    print("Carregando modelo, camera e LiDAR (uma vez so; pode levar um tempo)...")
    # O caminho dos arquivos e relativo ao repositorio, como no comando de sempre.
    os.chdir(_REPO)
    try:
        nova_corrida, fecha_tudo, opcoes = carrega_corrida(a, extras)
    except Exception as erro:
        try:
            atuadores.solta()
        except IOError:
            pass
        sys.exit("nao consegui preparar a corrida: {0}".format(erro))

    atuadores.manda(STEER_CENTER_US, ESC_NEUTRAL_US)
    if ESC_ARMADO:
        print("Armando o ESC: neutro por 2 s...")
        time.sleep(2.0)

    def estaciona():
        return para_com_insistencia(atuadores.manda, STEER_CENTER_US, ESC_NEUTRAL_US)

    def avisa(evento):
        print(_FALAS[evento])
        if evento == "erro" and unico.erro:
            print(unico.erro.rstrip())

    unico = ControleUnico(nova_corrida, manual, estaciona, avisa=avisa,
                          limite_s=opcoes.seconds)

    print("")
    print("=== Controle 8BitDo ===")
    print("A MAO: analogico esquerdo = volante (span {0} us) | RT = acelera 1600-{1} us | "
          "LT = re {2} us".format(a.span, min(a.max_us, ESC_MAX_US), ESC_RE_US))
    print("       Back ou Home = parada")
    print("DESLIGAR: LB + RB juntos por {0:.0f} s, sem corrida (as rodas mexem; espere o "
          "Jetson desligar antes de tirar a bateria)".format(DESLIGA_S))
    if not pode_desligar():
        print("  " + _SEM_REGRA)
    print("CORRIDA: Start = estaciona e larga o modelo | Back = para | limite {0:.0f} s{1}"
          .format(opcoes.seconds, " | gravando em runs/" if a.grava else ""))
    print("Ctrl+C: sai")
    if ESC_ARMADO:
        print("ESC ARMADO: o carro anda. Rodas no ar no primeiro teste, chave na mao.")
    else:
        print("ESC desarmado (ESC_ARMADO = False): a mao, so o volante responde.")
    print("")

    # Um jumper do I2C que solta por um instante nao derruba o modo manual: o
    # comando vai de novo no quadro seguinte (ai/car/escrita_tolerante.py).
    escrita = EscritaTolerante(atuadores.manda)
    watchdog = FrameWatchdog(WATCHDOG_S)
    desligar = SegurarParaDesligar(DESLIGA_S)
    ultimo_status, ultimo_aviso = 0.0, None
    quadros, inicio_quadros = 0, time.monotonic()
    motivo_saida = "interrompido"
    desligando = False
    try:
        while True:
            relatorio = controle.le(LEITURA_CORRIDA_MS if unico.correndo else LEITURA_MS)
            agora_estado = None if relatorio is None else estado(relatorio)
            cmd = unico.passo(agora_estado)
            if desligar.passo(agora_estado, unico.correndo):
                # Rodas mexendo tem que querer dizer "esta desligando": sem a
                # regra do sudo, nada de aviso.
                if not pode_desligar():
                    print(_SEM_REGRA)
                else:
                    estaciona()
                    print("=== LB + RB: DESLIGANDO O JETSON. Espere ele terminar antes de "
                          "tirar a bateria. ===")
                    avisa_com_as_rodas(escrita.manda, a.span)
                    # O desligamento do sistema manda SIGINT a este programa (o
                    # servico para assim); ignorado, a saida segura vai inteira.
                    signal.signal(signal.SIGINT, signal.SIG_IGN)
                    if desliga_o_jetson():
                        motivo_saida, desligando = "desligando o Jetson (LB + RB)", True
                        break
                    signal.signal(signal.SIGINT, signal.default_int_handler)
                    print("!!! O COMANDO DE DESLIGAR FALHOU: o Jetson continua ligado. "
                          "Nao tire a bateria. !!!")
                    continue      # o comando deste giro e de antes do aviso
            travou = watchdog.tick(time.monotonic())
            agora = time.monotonic()
            if cmd is None:
                # Quem escreveu na placa foi o laco do modelo (ou a corrida acabou
                # de parar e o carro foi estacionado).
                if unico.correndo and unico.telemetria is not None:
                    quadros += 1
                    if agora - inicio_quadros >= STATUS_CORRIDA_S:
                        print(_linha_da_corrida(unico.telemetria, quadros,
                                                agora - inicio_quadros))
                        quadros, inicio_quadros = 0, agora
                continue
            quadros, inicio_quadros = 0, agora
            esc_us = cmd.esc_us if (ESC_ARMADO and not travou) else ESC_NEUTRAL_US
            passou = escrita.manda(cmd.servo_us, esc_us)

            if not passou:
                aviso = "I2C FALHOU (fio solto?): repetindo"
            elif travou:
                aviso = "LACO TRAVOU: ESC em neutro"
            else:
                aviso = _AVISOS.get(cmd.motivo)
            if aviso != ultimo_aviso or agora - ultimo_status >= STATUS_S:
                falhas = "   [I2C: {0} falhas]".format(escrita.falhas) if escrita.falhas else ""
                print("volante {0:4d} us   esc {1:4d} us   {2}{3}".format(
                    cmd.servo_us, esc_us, aviso or "", falhas))
                ultimo_status, ultimo_aviso = agora, aviso
    except KeyboardInterrupt:
        pass
    except ControlePerdido as erro:
        motivo_saida = "controle perdido ({0})".format(erro)
    except PCA9685Perdido as erro:
        motivo_saida = str(erro)
    except Exception as erro:
        motivo_saida = "erro inesperado ({0!r})".format(erro)
        raise
    finally:
        seguro = unico.encerra()
        if not seguro:
            print("\n!!! NAO CONSEGUI POR O CARRO EM ESTADO SEGURO. "
                  "DESLIGUE A CHAVE DO CARRO. !!!")
        fecha_tudo()
        time.sleep(0.5)
        try:
            atuadores.solta()
        except IOError:
            pass
        print("\n{0}saindo: {1}".format("estado seguro. " if seguro else "", motivo_saida))
    return 0 if motivo_saida == "interrompido" or desligando else 1


if __name__ == "__main__":
    sys.exit(main())
