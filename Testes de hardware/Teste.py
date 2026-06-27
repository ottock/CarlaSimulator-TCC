#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Teste BRUTO de steering - posicoes fixas com pausa longa."""

import time
import Adafruit_PCA9685

I2C_ADDRESS = 0x40
I2C_BUSNUM = 1
SERVO_CHANNEL = 1
PWM_FREQ = 50

PERIOD_US = 1000000.0 / PWM_FREQ
STEPS = 4096


def us_to_steps(us):
    return max(0, min(4095, int(us / (PERIOD_US / STEPS))))


pwm = Adafruit_PCA9685.PCA9685(address=I2C_ADDRESS, busnum=I2C_BUSNUM)
pwm.set_pwm_freq(PWM_FREQ)


def set_us(us):
    print("  -> {}us  (passos: {})".format(us, us_to_steps(us)))
    pwm.set_pwm(SERVO_CHANNEL, 0, us_to_steps(us))


try:
    while True:
        print("CENTRO (1500us)")
        set_us(1500)
        time.sleep(1.5)
        print("UM LADO (1200us)")
        set_us(1200)
        time.sleep(1.5)
        print("CENTRO (1500us)")
        set_us(1500)
        time.sleep(1.5)
        print("OUTRO LADO (1800us)")
        set_us(1800)
        time.sleep(1.5)
except KeyboardInterrupt:
    set_us(1500)
    time.sleep(0.5)
    pwm.set_pwm(SERVO_CHANNEL, 0, 0)
    print("\nFinalizado.")