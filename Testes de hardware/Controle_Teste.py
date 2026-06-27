#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Controle_Manual_Teclado.py
# Fase 3 (etapa 3.2) - Controle manual do carro via teclado.
# Projeto TCC Carro Autonomo RC - Instituto Maua.
#
# Ambiente: Jetson Nano (Yahboom) / JetPack 4.6.x / Ubuntu 18.04 / Python 3.6
# Biblioteca PWM: Adafruit_PCA9685 (legacy) com busnum=1.
# Entrada de teclado: pygame (ja usado no Coin_d6.py). Precisa de uma
# janela em foco -> requer monitor HDMI no Jetson (nao roda por SSH puro).
#
# ============================ SEGURANCA ============================
#  - RODAS FORA DO CHAO em qualquer teste com o ESC/motor. Sempre.
#  - KILL SWITCH na mao quando o ESC estiver armado.
#  - Ground comum: GND Jetson + GND PCA9685 + GND ESC/servo no mesmo ponto.
#  - V+ do PCA9685 vem do BEC do ESC (5V), NUNCA do GPIO do Jetson.
#  - O ESC comeca DESARMADO (ESC_ARMADO = False). Enquanto desarmado,
#    NENHUM comando de throttle e enviado: da pra testar so o servo
#    com seguranca total. So mude para True quando as rodas estiverem
#    no ar e o kill switch estiver na sua mao.
#  - Controle "hold-to-move": o carro so age ENQUANTO a tecla esta
#    pressionada. Soltou -> servo centraliza e throttle volta a neutro.
# ==================================================================
#
# Teclas:
#   SETAS ESQ/DIR  -> estercamento (servo)
#   SETA CIMA      -> acelera (throttle)   [so se ESC_ARMADO=True]
#   SETA BAIXO     -> re/freio             [so se ESC_ARMADO=True]
#   BARRA DE ESPACO-> PARADA TOTAL (neutro + centro) enquanto segurada
#   ESC ou Q       -> sair (deixa tudo em neutro/centro)

import sys
import time
import os

# --- Workaround do pygame/SDL neste Jetson (JetPack 4.6.x, SDL 2.0.8) ---
# Sem isto, o pygame aborta com "Aborted (core dumped)" ao abrir a janela:
#   SDL_VIDEODRIVER=x11   -> forca o backend de video X11 (o que roda aqui)
#   DBUS_FATAL_WARNINGS=0 -> impede que um aviso do D-Bus derrube o processo
# Tem que vir ANTES de 'import pygame'. (Validado na bancada Jun 2026.)
os.environ.setdefault("SDL_VIDEODRIVER", "x11")
os.environ.setdefault("DBUS_FATAL_WARNINGS", "0")

try:
    import pygame
except ImportError:
    print("ERRO: pygame nao instalado.")
    print("Instalado via: python3 -m pip install --user pygame==2.1.2")
    print("(precisa de libsdl2-dev, libjpeg-dev etc. via apt antes de compilar)")
    sys.exit(1)

try:
    import Adafruit_PCA9685
except ImportError:
    print("ERRO: Adafruit_PCA9685 nao instalado.")
    print("Rode: python3 -m pip install --user Adafruit_PCA9685")
    sys.exit(1)


# ======================================================================
# CONFIGURACAO
# ======================================================================

# --- Trava de seguranca do ESC ---
# False  -> throttle NUNCA e enviado (so testa servo). Use assim primeiro.
# True   -> habilita throttle. So com rodas no ar e kill switch na mao.
ESC_ARMADO = True

# --- I2C / PCA9685 ---
I2C_ADDRESS = 0x40
I2C_BUSNUM  = 1
PWM_FREQ    = 50          # Hz, padrao RC

# --- Canais (conforme MASTER: servo=15, ESC=12) ---
SERVO_CHANNEL = 15
ESC_CHANNEL   = 12

# --- Limites do servo (steering), em microssegundos ---
# Conservador. Abra so depois de confirmar que nao bate no batente.
STEER_CENTER_US = 1500
STEER_LEFT_US   = 1300    # um lado
STEER_RIGHT_US  = 1700    # outro lado

# --- Limites do ESC (throttle), em microssegundos ---
# Valores de BANCADA ja calibrados (ver MASTER):
#   1500 = neutro | deadband ate ~1600 | 1600-1700 = faixa de bancada.
# THROTTLE_MAX_US recria em software o limite que vivia no radio original.
THROTTLE_NEUTRO_US = 1500
THROTTLE_MIN_US    = 1600    # velocidade minima real (forward)
THROTTLE_MAX_US    = 1700    # teto de seguranca de bancada
# Re/freio: conservador, espelhando abaixo do neutro. A re do ESC ainda
# nao foi validada (ver MASTER) -> mantemos faixa curta ate testar.
THROTTLE_RE_US     = 1400

# --- Suavidade (rampas) ---
# Em vez de pular direto pro valor, caminhamos suave ate ele. Isso evita
# trancos no servo e socos de corrente no ESC.
STEER_STEP_US    = 8      # us por frame ao mover o servo
THROTTLE_STEP_US = 4      # us por frame ao mudar o throttle (mais suave)

# --- Loop ---
FPS = 50                  # frames por segundo do laco de controle


# ======================================================================
# CONVERSAO us -> passos de 12 bits (0..4095)
# ======================================================================
PERIOD_US = 1000000.0 / PWM_FREQ      # 20000 us em 50Hz
STEPS = 4096


def us_to_steps(microseconds):
    """Converte largura de pulso (us) em passos de 12 bits (0..4095)."""
    us_per_step = PERIOD_US / STEPS
    steps = int(microseconds / us_per_step)
    return max(0, min(4095, steps))


# ======================================================================
# SETUP DO PCA9685
# ======================================================================
pwm = Adafruit_PCA9685.PCA9685(address=I2C_ADDRESS, busnum=I2C_BUSNUM)
pwm.set_pwm_freq(PWM_FREQ)


def set_us(channel, microseconds):
    """Aplica uma largura de pulso (us) num canal do PCA9685."""
    pwm.set_pwm(channel, 0, us_to_steps(microseconds))


def soltar(channel):
    """Solta o canal (sem segurar)."""
    pwm.set_pwm(channel, 0, 0)


def aproximar(atual, alvo, passo):
    """Caminha 'atual' em direcao a 'alvo' no maximo 'passo' por chamada."""
    if atual < alvo:
        return min(atual + passo, alvo)
    if atual > alvo:
        return max(atual - passo, alvo)
    return atual


# ======================================================================
# ARMING DO ESC
# ======================================================================
def armar_esc():
    """Mantem neutro por ~2s para o ESC aceitar comandos (sequencia de arming)."""
    print("Armando ESC: enviando neutro ({0}us) por 2s...".format(THROTTLE_NEUTRO_US))
    set_us(ESC_CHANNEL, THROTTLE_NEUTRO_US)
    time.sleep(2.0)
    print("ESC armado.")


# ======================================================================
# LOOP PRINCIPAL
# ======================================================================
def main():
    pygame.init()
    # Janela pequena: o pygame precisa dela em foco para capturar teclas.
    tela = pygame.display.set_mode((480, 240))
    pygame.display.set_caption("Controle Manual - TCC (ESC {0})".format(
        "ARMADO" if ESC_ARMADO else "DESARMADO"))
    fonte = pygame.font.SysFont("monospace", 16)
    fonte_b = pygame.font.SysFont("monospace", 18, bold=True)
    clock = pygame.time.Clock()

    # Estado atual dos atuadores (em us). Comecam neutros/centrados.
    steer_us = STEER_CENTER_US
    throttle_us = THROTTLE_NEUTRO_US

    # Centraliza o servo e poe o ESC em neutro antes de comecar.
    set_us(SERVO_CHANNEL, steer_us)
    if ESC_ARMADO:
        armar_esc()
    else:
        # Mesmo desarmado, deixamos o canal do ESC em neutro definido.
        set_us(ESC_CHANNEL, THROTTLE_NEUTRO_US)

    print("")
    print("=== Controle manual iniciado ===")
    print("Setas: dirigir | Espaco: parada total | ESC/Q: sair")
    print("ESC armado: {0}".format(ESC_ARMADO))
    print("Mantenha as RODAS FORA DO CHAO e o KILL SWITCH na mao.")
    print("")

    rodando = True
    try:
        while rodando:
            # ---- 1. Eventos (sair) ----
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    rodando = False
                if e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q):
                    rodando = False

            # ---- 2. Estado das teclas (hold-to-move) ----
            teclas = pygame.key.get_pressed()

            parada_total = teclas[pygame.K_SPACE]

            # Alvo do servo conforme setas (sem tecla = centro).
            if parada_total:
                steer_alvo = STEER_CENTER_US
            elif teclas[pygame.K_LEFT]:
                steer_alvo = STEER_LEFT_US
            elif teclas[pygame.K_RIGHT]:
                steer_alvo = STEER_RIGHT_US
            else:
                steer_alvo = STEER_CENTER_US

            # Alvo do throttle conforme setas (sem tecla = neutro).
            # So consideramos throttle se o ESC estiver armado.
            if (not ESC_ARMADO) or parada_total:
                throttle_alvo = THROTTLE_NEUTRO_US
            elif teclas[pygame.K_UP]:
                throttle_alvo = THROTTLE_MIN_US   # comeca no minimo real;
                # (poderiamos modular ate THROTTLE_MAX_US; mantemos simples e seguro)
            elif teclas[pygame.K_DOWN]:
                throttle_alvo = THROTTLE_RE_US
            else:
                throttle_alvo = THROTTLE_NEUTRO_US

            # ---- 3. Rampa suave ate os alvos ----
            steer_us = aproximar(steer_us, steer_alvo, STEER_STEP_US)
            set_us(SERVO_CHANNEL, steer_us)

            if ESC_ARMADO:
                throttle_us = aproximar(throttle_us, throttle_alvo, THROTTLE_STEP_US)
                # Trava dura: nunca passa do teto nem do limite de re.
                throttle_us = max(THROTTLE_RE_US, min(THROTTLE_MAX_US, throttle_us))
                set_us(ESC_CHANNEL, throttle_us)
            else:
                throttle_us = THROTTLE_NEUTRO_US

            # ---- 4. HUD na tela ----
            tela.fill((18, 18, 24))
            cor_esc = (255, 80, 80) if ESC_ARMADO else (120, 200, 120)
            linhas = [
                (fonte_b, "CONTROLE MANUAL - TCC", (230, 235, 245)),
                (fonte, "Steer:    {0} us".format(steer_us), (200, 200, 210)),
                (fonte, "Throttle: {0} us".format(throttle_us), (200, 200, 210)),
                (fonte, "ESC: {0}".format("ARMADO" if ESC_ARMADO else "DESARMADO"), cor_esc),
                (fonte, "Espaco=parar  ESC/Q=sair", (140, 145, 160)),
            ]
            y = 20
            for f, txt, cor in linhas:
                tela.blit(f.render(txt, True, cor), (20, y))
                y += 34
            pygame.display.flip()

            clock.tick(FPS)

    except KeyboardInterrupt:
        pass
    finally:
        # Estado seguro ao sair: servo centrado, ESC neutro, depois solta.
        print("Encerrando: centralizando servo e neutralizando ESC...")
        set_us(SERVO_CHANNEL, STEER_CENTER_US)
        set_us(ESC_CHANNEL, THROTTLE_NEUTRO_US)
        time.sleep(0.5)
        soltar(SERVO_CHANNEL)
        soltar(ESC_CHANNEL)
        pygame.quit()
        print("Finalizado em estado seguro.")


if __name__ == "__main__":
    main()