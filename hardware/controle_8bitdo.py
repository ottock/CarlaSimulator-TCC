#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""O programa do controle 8BitDo: dirige o carro a mao E larga a corrida autonoma.

Por que existe. Eram dois programas: este (o controle_teste.py com o controle
no lugar do teclado) e o largada_controle.py (Start lancava o
jetson_runtime.py, Back parava). O Rafael pediu um so em 2026-10-17. O
controle e lido direto da USB (docs/CONTROLE_8BITDO.md).

COMANDOS
  Sem corrida, voce dirige:
  analogico esquerdo, para os lados   volante, proporcional. Mesma conversao
                                      e mesmo sentido do runtime, mas com
                                      span 300 us para comecar (o modelo usa
                                      350; --span 350 iguala).
  RT                                  acelerador, de 1600 us (o minimo que move)
                                      ate --max-us (padrao 1630; o maximo
                                      calibrado e 1700). Soltou, para.
  LT                                  re, a 1420 us fixos (o teclado usa 1400).
  Back ou Home                        parada. Para andar de novo, solte os dois
                                      gatilhos e aperte outra vez.

  Start                               estaciona o carro (centro e neutro) e
                                      lanca a corrida autonoma: o
                                      jetson_runtime.py com o comando de
                                      sempre (ESTADO_IA.md secao 2). Durante a
                                      corrida este programa NAO escreve no
                                      PCA9685 -- ele e do runtime.
  Back, com corrida                   para a corrida (o mesmo sinal do Ctrl+C).
                                      Quando ela acaba, volta o modo manual,
                                      parado ate os gatilhos serem soltos.
  Ctrl+C no terminal                  para a corrida, se houver, estaciona o
                                      carro e sai.

PROTECOES
  - ESC_ARMADO, como no controle_teste.py. False: no modo manual o ESC nunca sai
    do neutro. (A corrida segue o ESC_ARMADO do jetson_runtime.py.)
  - Laco travado por mais de 0,25 s: ESC em neutro naquele quadro.
  - Controle mudo por mais de --silencio-s (0,5 s) com o carro andando a mao:
    para. O adaptador nao avisa quando o controle desliga e o ultimo estado
    fica valendo; sem isto um RT "preso" seguiria acelerando. O preco: segurar
    o RT no fundo com o analogico parado tambem e silencio, e pode parar o
    carro numa reta. Se acontecer demais, aumente --silencio-s.
  - Adaptador sumiu da USB: para a corrida, se houver, estaciona e sai.
  - Fio do I2C que solta por um instante (visto em 2026-10-14, Errno 121): o
    programa nao cai, o comando vai de novo no quadro seguinte e a tela conta
    as falhas. Falhando por 0,1 s seguidos, ele sai, tenta o neutro 10 vezes e,
    se nao conseguir, manda DESLIGAR A CHAVE: o PCA9685 continua mandando o
    ultimo pulso sozinho. O conserto de verdade e o fio.
  - Na corrida, as regras do largada: Start ignorado com corrida em andamento,
    3 s entre dois sinais de parada, e o runtime nunca e morto a forca.
  - Saida sempre tentando o estado seguro: servo no centro, ESC em neutro.

O QUE ELE NAO TEM / NAO RESOLVE
  - Parada frontal do LiDAR no modo manual. Quem dirige pode bater na parede.
  - O modelo e carregado de novo a cada corrida: entre o Start e o carro andar
    passa o mesmo tempo de quando o comando e digitado.
  - No fim da pista o carro para, mas o runtime continua ate o tempo dele
    acabar (--seconds, padrao 60). Aperte Back antes de voltar a dirigir.
  - Janela e radar: roda so no terminal, sem monitor no carro.

COMO USAR
  cd ~/CarlaSimulator-TCC
  python3 hardware/controle_8bitdo.py
  python3 hardware/controle_8bitdo.py --max-us 1650      # teto mais alto a mao
  python3 hardware/controle_8bitdo.py --seconds 120      # flags que este
                                                         # programa nao conhece
                                                         # vao para o runtime
  Controle ligado (chave traseira em 2.4G, luz do Home fixa). SEM sudo: precisa
  da regra de permissao do adaptador (docs/CONTROLE_8BITDO.md, secao 7). Com
  sudo o programa recusa rodar: em 2026-10-14 a corrida, como root, nao achou
  as bibliotecas do CUDA.

O modo manual e a largada funcionaram no carro em programas separados
(2026-10-14). JUNTOS, escrito em 2026-10-17 e ainda NAO rodado no Jetson:
primeira vez com as rodas no ar.
"""
import argparse
import datetime
import os
import signal
import subprocess
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

from ai.car.control_map import (ESC_MAX_US, ESC_NEUTRAL_US, STEER_CENTER_US,
                                STEER_SPAN_US, FrameWatchdog)
from ai.car.controle_unico import ControleUnico
from ai.car.escrita_tolerante import EscritaTolerante, PCA9685Perdido, para_com_insistencia
from ai.car.gamepad import Controle, ControlePerdido, acha_adaptador, estado
from ai.car.largada import comando_da_corrida, pasta_da_corrida
from ai.car.manual import ESC_RE_US, Manual

# ============================ SEGURANCA ============================
# True  -> o RT e o LT movem o carro. RODAS NO AR no primeiro teste.
# False -> o ESC fica em neutro o tempo todo; so o volante responde.
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

# Teto do acelerador (RT no fundo). Era 1700, o maximo calibrado do ESC; no
# teste de 2026-10-14 o Rafael achou rapido demais e baixou para 1630. A IA anda
# a 1610 fixos. Para subir: --max-us, ate 1700.
MAX_INICIAL_US = 1630

WATCHDOG_S = 0.25
# Prazo de cada leitura do controle. Curto, para o laco girar (e o silencio ser
# percebido) mesmo quando o controle nao manda nada.
LEITURA_MS = 20
STATUS_S = 0.5

_RUNTIME = os.path.join("hardware", "jetson_runtime.py")

_AVISOS = {
    "parado": "PARADO (Back/Home): solte os gatilhos para andar de novo",
    "silencio": "SEM NOTICIAS DO CONTROLE: parado ate chegar relatorio novo",
}

_FALAS = {
    "largou": "=== CORRIDA LANCADA (o modelo ainda vai carregar). Back para. ===",
    "parando": "=== Back: parando a corrida... ===",
    "acabou": "=== corrida encerrada. Modo manual: solte os gatilhos para andar. ===",
    "nao_parou": "!!! O RUNTIME NAO RESPONDEU. DESLIGUE A CHAVE DO CARRO. !!!",
}


def _passos(microssegundos):
    """us -> passos de 12 bits do PCA9685 a 50 Hz (como no controle_teste.py)."""
    passos = int(microssegundos / (1000000.0 / PWM_FREQ / 4096))
    return max(0, min(4095, passos))


class Atuadores(object):
    """Servo e ESC no PCA9685. O import fica aqui: a biblioteca so existe no Jetson."""

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


def _sigint_padrao():
    # Lancado em segundo plano (nohup, &), o shell deixa o SIGINT ignorado e o
    # filho herda isso: o runtime nunca veria o "Ctrl+C" do Back.
    signal.signal(signal.SIGINT, signal.SIG_DFL)


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
    p.add_argument("--engine", default="models/driving_grade_v5d1.engine",
                   help="Corrida: arquivo .engine do TensorRT (padrao: o v5d1)")
    p.add_argument("--config", default="models/driving_grade_v5d1.json",
                   help="Corrida: sidecar .json do modelo")
    p.add_argument("--cruise-us", type=int, default=1610,
                   help="Corrida: PWM constante do ESC passado ao runtime (padrao 1610)")
    p.add_argument("--beco-dist", type=float, default=0.5,
                   help="Corrida: parada de fim de pista passada ao runtime (padrao 0.5)")
    a, extras = p.parse_known_args(argv)

    # Melhor descobrir agora do que depois de apertar Start.
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
        sys.exit("nao consegui abrir o adaptador: {0}\n"
                 "Sem sudo, e preciso da regra de permissao "
                 "(docs/CONTROLE_8BITDO.md, secao 7).".format(erro))

    atuadores = Atuadores()
    atuadores.manda(STEER_CENTER_US, ESC_NEUTRAL_US)
    if ESC_ARMADO:
        print("Armando o ESC: neutro por 2 s...")
        time.sleep(2.0)

    def estaciona():
        return para_com_insistencia(atuadores.manda, STEER_CENTER_US, ESC_NEUTRAL_US)

    def inicia_runtime():
        out = pasta_da_corrida(a.engine, datetime.datetime.now())
        comando = comando_da_corrida(
            sys.executable, _RUNTIME, engine=a.engine, config=a.config, out=out,
            cruise_us=a.cruise_us, beco_dist=a.beco_dist, extras=extras)
        print("$ " + " ".join(comando))
        return subprocess.Popen(comando, cwd=_REPO, preexec_fn=_sigint_padrao)

    unico = ControleUnico(inicia_runtime, manual, estaciona,
                          avisa=lambda evento: print(_FALAS[evento]))

    print("")
    print("=== Controle 8BitDo ===")
    print("A MAO: analogico esquerdo = volante (span {0} us) | RT = acelera 1600-{1} us | "
          "LT = re {2} us".format(a.span, min(a.max_us, ESC_MAX_US), ESC_RE_US))
    print("       Back ou Home = parada")
    print("CORRIDA: Start = estaciona e larga o modelo | Back = para a corrida")
    print("Ctrl+C: sai")
    if ESC_ARMADO:
        print("ESC ARMADO: o carro anda. Rodas no ar no primeiro teste, chave na mao.")
    else:
        print("ESC desarmado (ESC_ARMADO = False): a mao, so o volante responde.")
    print("")

    # Um jumper do I2C que solta por um instante nao derruba o programa: o
    # comando vai de novo no quadro seguinte (ai/car/escrita_tolerante.py).
    escrita = EscritaTolerante(atuadores.manda)
    watchdog = FrameWatchdog(WATCHDOG_S)
    ultimo_status, ultimo_aviso = 0.0, None
    motivo_saida = "interrompido"
    sinaliza = True
    try:
        while True:
            relatorio = controle.le(LEITURA_MS)
            cmd = unico.passo(None if relatorio is None else estado(relatorio))
            travou = watchdog.tick(time.monotonic())
            if cmd is None:
                continue        # a placa e do runtime; a saida na tela e dele
            esc_us = cmd.esc_us if (ESC_ARMADO and not travou) else ESC_NEUTRAL_US
            passou = escrita.manda(cmd.servo_us, esc_us)

            if not passou:
                aviso = "I2C FALHOU (fio solto?): repetindo"
            elif travou:
                aviso = "LACO TRAVOU: ESC em neutro"
            else:
                aviso = _AVISOS.get(cmd.motivo)
            agora = time.monotonic()
            if aviso != ultimo_aviso or agora - ultimo_status >= STATUS_S:
                falhas = "   [I2C: {0} falhas]".format(escrita.falhas) if escrita.falhas else ""
                print("volante {0:4d} us   esc {1:4d} us   {2}{3}".format(
                    cmd.servo_us, esc_us, aviso or "", falhas))
                ultimo_status, ultimo_aviso = agora, aviso
    except KeyboardInterrupt:
        # O Ctrl+C do terminal chega ao runtime tambem (mesmo grupo): repetir o
        # sinal poderia cortar a limpeza dele.
        sinaliza = False
    except ControlePerdido as erro:
        motivo_saida = "controle perdido ({0})".format(erro)
    except PCA9685Perdido as erro:
        motivo_saida = str(erro)
    except Exception as erro:
        motivo_saida = "erro inesperado ({0!r})".format(erro)
        raise
    finally:
        seguro = unico.encerra(sinaliza=sinaliza)
        if not seguro:
            print("\n!!! NAO CONSEGUI POR O CARRO EM ESTADO SEGURO. "
                  "DESLIGUE A CHAVE DO CARRO. !!!")
        time.sleep(0.5)
        try:
            atuadores.solta()
        except IOError:
            pass
        print("\n{0}saindo: {1}".format("estado seguro. " if seguro else "", motivo_saida))
    return 0 if motivo_saida == "interrompido" else 1


if __name__ == "__main__":
    sys.exit(main())
