#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste de steering via PCA9685 (I2C) - sweep suave entre extremos.
Projeto TCC Carro Autonomo RC - Instituto Maua.

Biblioteca: Adafruit_PCA9685 (legacy, compativel com Python 3.6).
Ambiente: Jetson Nano, JetPack 4.6.x, Ubuntu 18.04, Python 3.6.

SEGURANCA:
- Ground comum: GND Jetson + GND PCA9685 + GND ESC/servo no mesmo ponto.
- V+ do PCA9685 vem do BEC do ESC (5V), NUNCA do GPIO do Jetson.
- Servo no canal 1 (steering), conforme MASTER.
- Rodas fora do chao.
"""

import time
import Adafruit_PCA9685

# ----------------------------------------------------------------------
# CONFIGURACAO
# ----------------------------------------------------------------------

I2C_ADDRESS = 0x40       # endereco do PCA9685 (confirmado via i2cdetect)
I2C_BUSNUM = 1           # barramento I2C do Jetson (i2cdetect -y -r 1)

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
# CONVERSAO us -> passos de 12 bits (0..4095)
# ----------------------------------------------------------------------
# O PCA9685 divide cada periodo PWM em 4096 passos (12 bits).
# Periodo em 50Hz = 20000us. Cada passo = 20000/4096 ~= 4.88us.
# passos = largura_us / us_por_passo

PERIOD_US = 1000000.0 / PWM_FREQ   # 20000 us em 50Hz
STEPS = 4096


def us_to_steps(microseconds):
    """Converte largura de pulso (us) para passos de 12 bits (0..4095)."""
    us_per_step = PERIOD_US / STEPS
    steps = int(microseconds / us_per_step)
    # trava nos limites validos do registrador
    return max(0, min(4095, steps))


# ----------------------------------------------------------------------
# SETUP DO PCA9685
# ----------------------------------------------------------------------
# busnum=1 evita o autodetect que falha no Jetson
# ("Could not determine default I2C bus for platform").

pwm = Adafruit_PCA9685.PCA9685(address=I2C_ADDRESS, busnum=I2C_BUSNUM)
pwm.set_pwm_freq(PWM_FREQ)


def set_us(channel, microseconds):
    """Define a largura de pulso em microssegundos num canal."""
    off = us_to_steps(microseconds)
    # on=0 -> pulso comeca no inicio do ciclo; off=quando desliga
    pwm.set_pwm(channel, 0, off)


def stop(channel):
    """Solta o canal (sem segurar torque)."""
    pwm.set_pwm(channel, 0, 0)


# ----------------------------------------------------------------------
# SWEEP
# ----------------------------------------------------------------------

def sweep_once():
    """Varre de MIN -> MAX -> MIN suavemente."""
    us = US_MIN
    while us <= US_MAX:
        set_us(SERVO_CHANNEL, us)
        us += STEP_US
        time.sleep(STEP_DELAY)
    us = US_MAX
    while us >= US_MIN:
        set_us(SERVO_CHANNEL, us)
        us -= STEP_US
        time.sleep(STEP_DELAY)


def main():
    print("Teste de steering via PCA9685 (Adafruit_PCA9685 legacy)")
    print("Endereco: 0x{:02X} | Barramento I2C: {}".format(
        I2C_ADDRESS, I2C_BUSNUM))
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
        stop(SERVO_CHANNEL)
        print("Finalizado.")


if __name__ == "__main__":
    main()