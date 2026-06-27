#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste de steering via PCA9685 (I2C) - sweep suave entre extremos.
Projeto TCC Carro Autonomo RC - Instituto Maua.

Migra o controle do servo de Jetson.GPIO para o PCA9685.
Ambiente: Jetson Nano, JetPack 4.6.x, Ubuntu 18.04, Python 3.6.

SEGURANCA:
- Ground comum: GND Jetson + GND PCA9685 + GND ESC/servo no mesmo ponto.
- V+ do PCA9685 vem do BEC do ESC (5V), NUNCA do GPIO do Jetson.
- Servo na canal 1 (steering), conforme MASTER.
"""

import time
import board
import busio
from adafruit_pca9685 import PCA9685

# ----------------------------------------------------------------------
# CONFIGURACAO
# ----------------------------------------------------------------------

SERVO_CHANNEL = 1        # canal do servo (steering) no PCA9685
PWM_FREQ = 50            # Hz - padrao RC

# Limites em microssegundos (largura de pulso).
# COMECE CONSERVADOR e abra aos poucos na etapa de calibracao.
# 1500us = centro nominal. NAO va direto pra 1000/2000 sem testar:
# o servo pode bater no batente mecanico e queimar.
US_MIN = 1100            # extremo de um lado (ex: esquerda)
US_CENTER = 1500         # centro
US_MAX = 1900            # extremo do outro lado (ex: direita)

# Suavidade do sweep
STEP_US = 5              # passo em us por iteracao (menor = mais suave)
STEP_DELAY = 0.02        # segundos entre passos

# ----------------------------------------------------------------------
# CONVERSAO us -> duty cycle 16 bits
# ----------------------------------------------------------------------
# O PCA9685 (lib Adafruit) usa duty_cycle de 16 bits (0..65535).
# Periodo do sinal = 1/PWM_FREQ. Em 50Hz -> 20000us por ciclo.
# fracao do periodo = largura_us / periodo_us
# duty_16bit = fracao * 65535

PERIOD_US = 1000000 / PWM_FREQ   # 20000 us em 50Hz


def us_to_duty(microseconds):
    """Converte largura de pulso (us) para duty cycle de 16 bits."""
    duty = int((microseconds / PERIOD_US) * 65535)
    # trava nos limites validos do registrador
    return max(0, min(65535, duty))


# ----------------------------------------------------------------------
# SETUP DO PCA9685
# ----------------------------------------------------------------------

i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = PWM_FREQ


def set_us(channel, microseconds):
    """Define a largura de pulso em microssegundos num canal."""
    pca.channels[channel].duty_cycle = us_to_duty(microseconds)


# ----------------------------------------------------------------------
# SWEEP
# ----------------------------------------------------------------------

def sweep_once():
    """Varre de MIN -> MAX -> MIN suavemente."""
    # MIN -> MAX
    us = US_MIN
    while us <= US_MAX:
        set_us(SERVO_CHANNEL, us)
        us += STEP_US
        time.sleep(STEP_DELAY)
    # MAX -> MIN
    us = US_MAX
    while us >= US_MIN:
        set_us(SERVO_CHANNEL, us)
        us -= STEP_US
        time.sleep(STEP_DELAY)


def main():
    print("Teste de steering via PCA9685")
    print("Freq: {} Hz | MIN: {}us | CENTRO: {}us | MAX: {}us".format(
        PWM_FREQ, US_MIN, US_CENTER, US_MAX))
    print("Centralizando servo...")
    set_us(SERVO_CHANNEL, US_CENTER)
    time.sleep(1.0)

    print("Iniciando sweep. Ctrl+C para parar.")
    try:
        while True:
            sweep_once()
    except KeyboardInterrupt:
        print("\nParando. Centralizando servo...")
        set_us(SERVO_CHANNEL, US_CENTER)
        time.sleep(0.5)
        # zera o canal pra nao deixar o servo segurando posicao
        pca.channels[SERVO_CHANNEL].duty_cycle = 0
        pca.deinit()
        print("Finalizado.")


if __name__ == "__main__":
    main()