#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Confere o SINAL do esterco: steer negativo tem de virar para a ESQUERDA do carro.

Por que este teste existe: a convencao do CARLA, em que o modelo foi treinado, e
`steer = -1` para a ESQUERDA e `+1` para a DIREITA. Se o servo estiver montado ao
contrario, ou se o mapa para microssegundos tiver o sinal trocado, o carro esterca
espelhado -- ve parede a esquerda e vira PARA ela. Seria a mesma classe de erro
que ja encontramos no LiDAR (o sensor girava ao contrario), e e silenciosa: os
numeros no log parecem perfeitamente normais.

Usa `ai.car.control_map.steer_to_us`, a MESMA funcao do caminho do modelo. Uma
copia com os valores repetidos aqui poderia divergir e o teste passaria enquanto
o carro erra.

O ESC nao e tocado (fica em neutro), entao o carro nao anda. Rodas no ar ou no
chao, tanto faz.

Nao precisa de monitor: so imprime no terminal, funciona por SSH.

Uso:
    python3 hardware/teste_esquerda.py
"""
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

try:
    import Adafruit_PCA9685
except ImportError:
    print("ERRO: Adafruit_PCA9685 nao instalado.")
    sys.exit(1)

from ai.car.control_map import ESC_NEUTRAL_US, STEER_CENTER_US, steer_to_us

I2C_ADDRESS = 0x40
I2C_BUSNUM = 1
PWM_FREQ = 50
SERVO_CHANNEL = 15
ESC_CHANNEL = 12

PERIOD_US = 1000000.0 / PWM_FREQ
STEPS = 4096
RAMPA_US = 4          # us por passo, para nao dar tranco no servo
RAMPA_DELAY = 0.01

pwm = Adafruit_PCA9685.PCA9685(address=I2C_ADDRESS, busnum=I2C_BUSNUM)
pwm.set_pwm_freq(PWM_FREQ)


def set_us(channel, us):
    passos = int(us / (PERIOD_US / STEPS))
    pwm.set_pwm(channel, 0, max(0, min(4095, passos)))


def ir_para(atual, alvo):
    """Move o servo suavemente ate o alvo e devolve onde parou."""
    passo = RAMPA_US if alvo > atual else -RAMPA_US
    while (passo > 0 and atual < alvo) or (passo < 0 and atual > alvo):
        atual += passo
        if (passo > 0 and atual > alvo) or (passo < 0 and atual < alvo):
            atual = alvo
        set_us(SERVO_CHANNEL, atual)
        time.sleep(RAMPA_DELAY)
    return atual


def main():
    set_us(ESC_CHANNEL, ESC_NEUTRAL_US)      # o carro NAO anda neste teste
    atual = STEER_CENTER_US
    set_us(SERVO_CHANNEL, atual)

    esq = steer_to_us(-1.0)
    dir_ = steer_to_us(+1.0)
    print("")
    print("=== Teste do sinal do esterco ===")
    print("  steer -1.0 -> %d us   (deve ser ESQUERDA do carro)" % esq)
    print("  steer   0  -> %d us" % STEER_CENTER_US)
    print("  steer +1.0 -> %d us   (deve ser DIREITA do carro)" % dir_)
    print("")
    print("Olhe as RODAS DA FRENTE, com o carro apontado para longe de voce")
    print("(a esquerda do carro e a MESMA que a sua nessa posicao).")
    print("Ctrl+C encerra centralizando.")
    print("")

    sequencia = [("CENTRO", STEER_CENTER_US, 2.0),
                 ("<<<<< ESQUERDA (steer -1)", esq, 3.0),
                 ("CENTRO", STEER_CENTER_US, 2.0),
                 ("DIREITA (steer +1) >>>>>", dir_, 3.0)]
    try:
        while True:
            for nome, alvo, espera in sequencia:
                print("  %-28s  %d us" % (nome, alvo))
                atual = ir_para(atual, alvo)
                time.sleep(espera)
            print("  --- repetindo (Ctrl+C para sair) ---")
    except KeyboardInterrupt:
        pass
    finally:
        ir_para(atual, STEER_CENTER_US)
        set_us(ESC_CHANNEL, ESC_NEUTRAL_US)
        print("")
        print("centralizado. Encerrado.")
        print("")
        print("SE AS RODAS FORAM PARA O LADO ERRADO: o esterco esta espelhado, e o")
        print("carro vira PARA a parede que deveria evitar. Me avise -- o conserto e")
        print("inverter o sinal em control_map.steer_to_us, nao mexer no hardware.")


if __name__ == "__main__":
    main()
