#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dirige o carro com o controle 8BitDo. E o controle_teste.py, trocando o
teclado pelo controle.

Por que existe. O controle_teste.py precisa de monitor e teclado ligados no
Jetson, e o volante dele e tudo ou nada (a seta vai direto ao batente). Com o
controle 8BitDo lido direto da USB (docs/CONTROLE_8BITDO.md), da para dirigir
de longe e com volante e acelerador proporcionais.

COMANDOS
  analogico esquerdo, para os lados   volante, proporcional. Mesma conversao
                                      e mesmo sentido do runtime, mas com
                                      span 300 us para comecar (o modelo usa
                                      350; --span 350 iguala).
  RT                                  acelerador, de 1600 us (o minimo que move)
                                      ate --max-us (padrao 1630; o maximo
                                      calibrado e 1700). Soltou, para.
  LT                                  re, a 1400 us fixos, como no teclado.
  Back ou Home                        parada. Para andar de novo, solte os dois
                                      gatilhos e aperte outra vez.
  Ctrl+C no terminal                  centraliza, ESC em neutro e sai.

PROTECOES
  - ESC_ARMADO, como no controle_teste.py. False: o ESC nunca sai do neutro.
  - Laco travado por mais de 0,25 s: ESC em neutro naquele quadro.
  - Controle mudo por mais de --silencio-s (0,5 s) com o carro andando: para.
    O adaptador nao avisa quando o controle desliga e o ultimo estado fica
    valendo; sem isto um RT "preso" seguiria acelerando. O preco: segurar o RT
    no fundo com o analogico parado tambem e silencio, e pode parar o carro
    numa reta. Se acontecer demais, aumente --silencio-s.
  - Adaptador sumiu da USB: para e sai.
  - Saida sempre em estado seguro: servo no centro, ESC em neutro.

O QUE ELE NAO TEM
  - Parada frontal do LiDAR. Quem dirige pode bater na parede, na velocidade
    que estiver mandando. O runtime tem; este programa nao le o LiDAR.
  - Janela e radar: roda so no terminal, sem monitor no carro.

COMO USAR
  cd ~/CarlaSimulator-TCC
  python3 hardware/controle_8bitdo.py
  python3 hardware/controle_8bitdo.py --max-us 1650     # teto mais alto
  Controle ligado (chave traseira em 2.4G, luz do Home fixa). Sem sudo: precisa
  da regra de permissao do adaptador (docs/CONTROLE_8BITDO.md, secao 7).

PRIMEIRA VEZ: rodas no ar. Escrito em 2026-10-14 e ainda NAO rodado no Jetson;
a logica (ai/car/manual.py) foi testada no PC. Conferir primeiro: analogico
para a direita vira as rodas para a direita? Se virar ao contrario, a troca e
SINAL_DO_ANALOGICO em src/ai/car/manual.py.
"""
import argparse
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

from ai.car.control_map import (ESC_MAX_US, ESC_NEUTRAL_US, STEER_CENTER_US,
                                STEER_SPAN_US, FrameWatchdog)
from ai.car.gamepad import Controle, ControlePerdido, acha_adaptador, estado
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

_AVISOS = {
    "parado": "PARADO (Back/Home): solte os gatilhos para andar de novo",
    "silencio": "SEM NOTICIAS DO CONTROLE: parado ate chegar relatorio novo",
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


def main(argv=None):
    p = argparse.ArgumentParser(description="Dirige o carro com o controle 8BitDo.")
    p.add_argument("--max-us", type=int, default=MAX_INICIAL_US,
                   help="Teto do acelerador (RT no fundo), de 1600 a %d us "
                        "(padrao %d)" % (ESC_MAX_US, MAX_INICIAL_US))
    p.add_argument("--span", type=int, default=SPAN_INICIAL_US,
                   help="Meia-faixa do servo em us (padrao %d; o modelo usa %d)"
                        % (SPAN_INICIAL_US, STEER_SPAN_US))
    p.add_argument("--silencio-s", type=float, default=0.5,
                   help="Segundos sem relatorio do controle que param o carro "
                        "andando (padrao 0.5)")
    a = p.parse_args(argv)

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

    print("")
    print("=== Controle 8BitDo ===")
    print("analogico esquerdo: volante (span {0} us) | RT: acelera 1600-{1} us | "
          "LT: re {2} us".format(a.span, min(a.max_us, ESC_MAX_US), ESC_RE_US))
    print("Back ou Home: parada | Ctrl+C: sai")
    if ESC_ARMADO:
        print("ESC ARMADO: o carro anda. Rodas no ar no primeiro teste, chave na mao.")
    else:
        print("ESC desarmado (ESC_ARMADO = False): so o volante responde.")
    print("")

    watchdog = FrameWatchdog(WATCHDOG_S)
    ultimo_status, ultimo_aviso = 0.0, None
    motivo_saida = "interrompido"
    try:
        while True:
            relatorio = controle.le(LEITURA_MS)
            if relatorio is not None:
                manual.recebe(estado(relatorio))
            cmd = manual.comando()
            travou = watchdog.tick(time.monotonic())
            esc_us = cmd.esc_us if (ESC_ARMADO and not travou) else ESC_NEUTRAL_US
            atuadores.manda(cmd.servo_us, esc_us)

            aviso = "LACO TRAVOU: ESC em neutro" if travou else _AVISOS.get(cmd.motivo)
            agora = time.monotonic()
            if aviso != ultimo_aviso or agora - ultimo_status >= STATUS_S:
                print("volante {0:4d} us   esc {1:4d} us   {2}".format(
                    cmd.servo_us, esc_us, aviso or ""))
                ultimo_status, ultimo_aviso = agora, aviso
    except KeyboardInterrupt:
        pass
    except ControlePerdido as erro:
        motivo_saida = "controle perdido ({0})".format(erro)
    finally:
        atuadores.manda(STEER_CENTER_US, ESC_NEUTRAL_US)
        time.sleep(0.5)
        atuadores.solta()
        print("\nestado seguro. saindo: {0}".format(motivo_saida))
    return 0 if motivo_saida == "interrompido" else 1


if __name__ == "__main__":
    sys.exit(main())
