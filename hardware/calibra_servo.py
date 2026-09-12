#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Acha o batente MECANICO do servo de esterco, passo a passo e sem forcar.

Por que existe, tendo `controle_pwm_steering.py`: aquele faz uma VARREDURA
automatica ate 1100/1900. Se o batente real estiver antes disso, ele fica
empurrando contra o limite a cada ciclo -- e o proprio arquivo avisa que isso
queima o servo. Aqui cada passo e' comandado por voce, e voce para no instante
em que a roda parar de responder.

Para que serve o numero: o modelo aprendeu `steer = +/-1` significando BATENTE
TOTAL. Hoje o codigo mapeia +/-1 para +/-200 us, valor herdado do controle por
teclado -- e conservador, nao medido. Se o batente real for maior, todo comando
de esterco sai reduzido na mesma proporcao e o carro subvira em TUDO.

Nao precisa de monitor: le do teclado pelo terminal, entao funciona por SSH.

    RODAS NO AR.

Uso:
    python3 hardware/calibra_servo.py
"""
import sys

try:
    import Adafruit_PCA9685
except ImportError:
    print("ERRO: Adafruit_PCA9685 nao instalado.")
    sys.exit(1)

I2C_ADDRESS = 0x40
I2C_BUSNUM = 1
PWM_FREQ = 50
SERVO_CHANNEL = 15

CENTER_US = 1500
STEP_US = 10
# Limite ABSOLUTO de seguranca. Nao e o batente -- e o ponto alem do qual nem
# tentamos, porque servos de RC raramente aceitam mais que isso.
HARD_MIN_US = 1000
HARD_MAX_US = 2000

PERIOD_US = 1000000.0 / PWM_FREQ
STEPS = 4096

pwm = Adafruit_PCA9685.PCA9685(address=I2C_ADDRESS, busnum=I2C_BUSNUM)
pwm.set_pwm_freq(PWM_FREQ)


def set_us(us):
    passos = int(us / (PERIOD_US / STEPS))
    pwm.set_pwm(SERVO_CHANNEL, 0, max(0, min(4095, passos)))


def main():
    us = CENTER_US
    set_us(us)
    print("")
    print("=== Calibracao do batente do servo (RODAS NO AR) ===")
    print("")
    print("  d + Enter  -> +%d us (ESQUERDA neste carro: o servo e espelhado)" % STEP_US)
    print("  a + Enter  -> -%d us (DIREITA neste carro)" % STEP_US)
    print("  c + Enter  -> volta ao centro (%d)" % CENTER_US)
    print("  q + Enter  -> sai, centralizando")
    print("")
    print("COMO MEDIR: va de 'd' em 'd' ate a roda PARAR de girar mais, ou ate")
    print("ouvir o servo forcando (zumbido). Esse e o batente. Anote o ultimo")
    print("valor em que ela AINDA se movia e volte 20 us -- essa e a margem.")
    print("Repita para o outro lado com 'a'. Depois use a MENOR das duas")
    print("distancias ao centro como --steer-span-us.")
    print("")
    try:
        while True:
            print("  us = %d   (centro %d, delta %+d)" % (us, CENTER_US, us - CENTER_US))
            cmd = sys.stdin.readline().strip().lower()
            if cmd == "q":
                break
            elif cmd == "c":
                us = CENTER_US
            elif cmd == "d":
                us = min(HARD_MAX_US, us + STEP_US)
            elif cmd == "a":
                us = max(HARD_MIN_US, us - STEP_US)
            else:
                continue
            set_us(us)
    except KeyboardInterrupt:
        pass
    finally:
        set_us(CENTER_US)
        print("")
        print("centralizado em %d us. Encerrado." % CENTER_US)


if __name__ == "__main__":
    main()
