#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Controle_PWM_Throttle.py
# Validacao do ESC via PCA9685 (I2C, canal 0) - Etapa 2.3
#
# OBJETIVO desta versao: so confirmar que o ESC arma e que o motor gira
# pra frente devagar, no sentido certo. Re/freio ficam pra depois.
#
# >>> SEGURANCA <<<
#   - RODAS FORA DO CHAO. Carro apoiado, 4 rodas girando livres.
#   - KILL SWITCH / switch fisico do carro NA MAO.
#   - Quando o ESC armar e o valor passar do neutro, a roda gira de verdade.
#
# Ambiente: Jetson Nano (Yahboom) / JetPack 4.6.x / Python 3.6
#   - lib legacy Adafruit_PCA9685 com busnum=1 (autodetect nao reconhece o Jetson)
#   - sem separador de milhar em literais (3.6 da SyntaxError)

import time
import Adafruit_PCA9685

# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------
ESC_CHANNEL = 12       # canal fisico onde o cabo do ESC esta plugado no PCA9685
I2C_ADDRESS = 0x40     # confirmado com: i2cdetect -y -r 1
I2C_BUSNUM  = 1        # Jetson: precisa ser explicito
PWM_FREQ_HZ = 50       # padrao RC

# Larguras de pulso em microssegundos.
NEUTRAL_US  = 1500     # ESC parado / neutro
ARM_HOLD_S  = 15.0      # tempo segurando neutro para o ESC armar

# Faixa de teste PROPOSITALMENTE estreita. So pra confirmar que anda.
# Depois de validado, abrir ate ~1900 conforme a calibracao (etapa 2.4).
TEST_MIN_US = 1500     # neutro
TEST_MAX_US = 1800     # teto conservador pro primeiro teste

# Periodo do ciclo PWM em microssegundos: 1/50Hz = 20000us
PERIOD_US = 1000000 // PWM_FREQ_HZ   # 20000

# ---------------------------------------------------------------------------
# Conversao us -> passos (12 bits / 4096 por ciclo)
# ---------------------------------------------------------------------------
def us_to_steps(pulse_us):
    """Converte largura de pulso (us) para passo 0-4095 do PCA9685."""
    steps = int(round(pulse_us * 4096.0 / PERIOD_US))
    if steps < 0:
        steps = 0
    if steps > 4095:
        steps = 4095
    return steps

def set_pulse(pwm, channel, pulse_us):
    """Aplica uma largura de pulso (us) num canal."""
    pwm.set_pwm(channel, 0, us_to_steps(pulse_us))

# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main():
    pwm = Adafruit_PCA9685.PCA9685(address=I2C_ADDRESS, busnum=I2C_BUSNUM)
    pwm.set_pwm_freq(PWM_FREQ_HZ)

    try:
        # 1) ARMING: segura neutro por alguns segundos.
        #    O ESC normalmente apita confirmando que armou.
        print("Armando ESC: enviando neutro ({}us) por {}s...".format(
            NEUTRAL_US, ARM_HOLD_S))
        print(">>> Confira: rodas fora do chao? kill switch na mao? <<<")
        set_pulse(pwm, ESC_CHANNEL, NEUTRAL_US)
        time.sleep(ARM_HOLD_S)
        print("ESC armado (deve ter apitado). Motor em neutro.")

        # 2) Teste manual: o usuario digita um valor em us dentro da faixa.
        #    Enter vazio = neutro. 'q' = sair.
        print("")
        print("Digite uma largura de pulso em us entre {} e {}.".format(
            TEST_MIN_US, TEST_MAX_US))
        print("Enter vazio volta ao neutro ({}us). 'q' encerra.".format(
            NEUTRAL_US))

        while True:
            entrada = input("us> ").strip()
            if entrada == "q":
                break
            if entrada == "":
                set_pulse(pwm, ESC_CHANNEL, NEUTRAL_US)
                print("  neutro")
                continue
            try:
                valor = int(entrada)
            except ValueError:
                print("  valor invalido")
                continue
            if valor < TEST_MIN_US or valor > TEST_MAX_US:
                print("  fora da faixa de teste ({}-{}). ignorado.".format(
                    TEST_MIN_US, TEST_MAX_US))
                continue
            set_pulse(pwm, ESC_CHANNEL, valor)
            print("  aplicado: {}us".format(valor))

    finally:
        # 3) SEMPRE volta a neutro ao sair (inclui Ctrl-C e qualquer erro).
        set_pulse(pwm, ESC_CHANNEL, NEUTRAL_US)
        print("\nNeutro reaplicado. Encerrado com seguranca.")


if __name__ == "__main__":
    main()