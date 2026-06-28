#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Controle_Integrado.py
# Fase 3 (etapa 3.4 parcial) - Radar COIN-D6 + controle manual numa janela so.
# Projeto TCC Carro Autonomo RC - Instituto Maua.
#
# Junta dois programas ja validados:
#   - Coin_d6.py        (visualizador do LiDAR COIN-D6)
#   - Controle_Teste.py (controle manual por teclado via PCA9685)
# ...num laco UNICO, numa janela UNICA. A camera fica para uma proxima etapa.
#
# Ambiente: Jetson Nano (Yahboom) / JetPack 4.6.x / Ubuntu 18.04 / Python 3.6
#   - PWM:   Adafruit_PCA9685 (legacy) com busnum=1
#   - LiDAR: pyserial em /dev/ttyUSB* (precisa do grupo 'dialout' p/ rodar
#            sem sudo; ver README/MASTER)
#   - Janela grafica: monitor HDMI no Jetson.
#
# ============================ SEGURANCA ============================
#  - ESC comeca DESARMADO (ESC_ARMADO = False). Primeiro VALIDE que o
#    radar desenha fluido E o servo responde sem atraso. So depois,
#    com RODAS NO AR e KILL SWITCH na mao, troque para True.
#  - Ordem dentro do frame: 1o atualiza atuadores (controle), 2o le a
#    serial e desenha o radar. Assim o comando do frame sai ANTES do
#    desenho pesado, e um engasgo grafico nao trava o throttle.
#  - Watchdog: se um frame demorar demais (laco travou), o ESC volta a
#    neutro automaticamente.
#  - Controle "hold-to-move": solta a tecla -> servo centraliza e
#    throttle volta a neutro.
# ==================================================================
#
# Teclas:
#   SETAS ESQ/DIR   -> estercamento (servo)
#   SETA CIMA       -> acelera        [so se ESC_ARMADO=True]
#   SETA BAIXO      -> re/freio        [so se ESC_ARMADO=True]
#   BARRA DE ESPACO -> PARADA TOTAL enquanto segurada
#   [ + ] / [ - ]   -> zoom do radar (range)
#   ESC ou Q        -> sair em estado seguro

import sys
import os
import time
import math
import struct
import glob

# --- Workaround pygame/SDL neste Jetson (ver Controle_Teste.py) ---
os.environ.setdefault("SDL_VIDEODRIVER", "x11")
os.environ.setdefault("DBUS_FATAL_WARNINGS", "0")

try:
    import serial
except ImportError:
    print("ERRO: pyserial nao instalado. Rode: python3 -m pip install --user pyserial")
    sys.exit(1)

try:
    import pygame
except ImportError:
    print("ERRO: pygame nao instalado (pip install --user pygame==2.1.2).")
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
# False -> throttle NUNCA enviado. Valide a fluidez assim PRIMEIRO.
# True  -> habilita throttle. So com rodas no ar e kill switch na mao.
ESC_ARMADO = False

# --- I2C / PCA9685 ---
I2C_ADDRESS = 0x40
I2C_BUSNUM  = 1
PWM_FREQ    = 50

# --- Canais (MASTER: servo=15, ESC=12) ---
SERVO_CHANNEL = 15
ESC_CHANNEL   = 12

# --- Limites do servo (us) ---
STEER_CENTER_US = 1500
STEER_LEFT_US   = 1300
STEER_RIGHT_US  = 1700

# --- Limites do ESC (us) - faixa de bancada calibrada (ver MASTER) ---
THROTTLE_NEUTRO_US = 1500
THROTTLE_MIN_US    = 1600
THROTTLE_MAX_US    = 1700
THROTTLE_RE_US     = 1400

# --- Suavidade (rampas) ---
STEER_STEP_US    = 8
THROTTLE_STEP_US = 4

# --- Loop ---
FPS = 30                  # 30 e suficiente; o radar e pesado a mais que isso
WATCHDOG_S = 0.25         # se um frame passar disso, ESC volta a neutro

# --- LiDAR COIN-D6 ---
BAUD_RATE = 230400
CMD_START = bytes([0xAA, 0x55, 0xF0, 0x0F])
CMD_STOP  = bytes([0xAA, 0x55, 0xF5, 0x0A])
LIDAR_MAX_M = 12.0        # alcance fisico do sensor


# ======================================================================
# CONVERSAO us -> passos 12 bits
# ======================================================================
PERIOD_US = 1000000.0 / PWM_FREQ
STEPS = 4096


def us_to_steps(microseconds):
    steps = int(microseconds / (PERIOD_US / STEPS))
    return max(0, min(4095, steps))


# ======================================================================
# PCA9685
# ======================================================================
pwm = Adafruit_PCA9685.PCA9685(address=I2C_ADDRESS, busnum=I2C_BUSNUM)
pwm.set_pwm_freq(PWM_FREQ)


def set_us(channel, microseconds):
    pwm.set_pwm(channel, 0, us_to_steps(microseconds))


def soltar(channel):
    pwm.set_pwm(channel, 0, 0)


def aproximar(atual, alvo, passo):
    if atual < alvo:
        return min(atual + passo, alvo)
    if atual > alvo:
        return max(atual - passo, alvo)
    return atual


def armar_esc():
    print("Armando ESC: neutro ({0}us) por 2s...".format(THROTTLE_NEUTRO_US))
    set_us(ESC_CHANNEL, THROTTLE_NEUTRO_US)
    time.sleep(2.0)
    print("ESC armado.")


# ======================================================================
# PARSER DO COIN-D6 (identico ao Coin_d6.py validado)
# ======================================================================
class CoinD6Parser:
    def __init__(self):
        self.buffer = bytearray()
        self.current_scan = []
        self.scan_count = 0
        self.parse_errors = 0

    def feed(self, data):
        self.buffer.extend(data)
        scans = []
        while True:
            idx = self.buffer.find(b'\xAA\x55')
            if idx < 0 or idx + 4 > len(self.buffer):
                if idx < 0:
                    self.buffer = self.buffer[-1:]
                break
            if idx > 0:
                self.buffer = self.buffer[idx:]

            sample_count = self.buffer[3]
            if sample_count == 0 or sample_count > 50:
                self.buffer = self.buffer[2:]
                self.parse_errors += 1
                continue

            pkt_len = 10 + sample_count * 3
            if len(self.buffer) < pkt_len:
                break

            pkt = bytes(self.buffer[:pkt_len])
            self.buffer = self.buffer[pkt_len:]

            start_angle = struct.unpack_from('<H', pkt, 4)[0] * 0.01
            end_angle   = struct.unpack_from('<H', pkt, 6)[0] * 0.01
            start_angle = start_angle % 360.0
            end_angle   = end_angle % 360.0

            angle_diff = end_angle - start_angle
            if angle_diff < 0:
                angle_diff += 360.0
            if angle_diff > 180:
                angle_diff -= 360.0
            step = angle_diff / max(sample_count - 1, 1)

            for i in range(sample_count):
                offset = 10 + i * 3
                if offset + 2 >= len(pkt):
                    break
                dist_mm = pkt[offset + 1] | (pkt[offset + 2] << 8)
                if dist_mm <= 1:
                    continue
                dist_m = dist_mm / 1000.0
                angle  = (start_angle + i * step) % 360.0
                if dist_m <= LIDAR_MAX_M:
                    self.current_scan.append((angle, dist_m))

            if len(self.current_scan) >= 300:
                scans.append(self.current_scan)
                self.current_scan = []
                self.scan_count += 1
        return scans


# ======================================================================
# CORES
# ======================================================================
BG       = (12, 12, 18)
RING     = (30, 38, 48)
CROSS    = (40, 45, 55)
TXT_DIM  = (100, 105, 120)
TXT      = (190, 195, 210)
TXT_HI   = (230, 235, 245)
CYAN     = (0, 200, 220)
GREEN    = (0, 210, 90)
YELLOW   = (255, 210, 0)
ORANGE   = (255, 130, 0)
RED      = (255, 50, 50)
BLUE     = (60, 130, 255)
SAFETY_R = (80, 25, 25)

DIST_COLORS = [RED, ORANGE, YELLOW, GREEN, CYAN, BLUE]
DIST_THRESHOLDS = [1.0, 2.0, 3.0, 5.0, 8.0]


def color_for_dist(d):
    for i, t in enumerate(DIST_THRESHOLDS):
        if d < t:
            return DIST_COLORS[i]
    return DIST_COLORS[-1]


# ======================================================================
# PROCURAR PORTA DO LIDAR
# ======================================================================
def achar_porta_lidar():
    for pat in ['/dev/ttyUSB*', '/dev/ttyACM*']:
        achadas = sorted(glob.glob(pat))
        if achadas:
            return achadas[0]
    return None


# ======================================================================
# APP INTEGRADO
# ======================================================================
class AppIntegrado:
    def __init__(self, w=980, h=680, max_range=12.0):
        pygame.init()
        pygame.display.set_caption("Controle Integrado - LiDAR + Controle (TCC)")
        self.screen = pygame.display.set_mode((w, h))
        self.w, self.h = w, h
        self.max_range = max_range
        self.clock = pygame.time.Clock()

        # Area do radar (quadrada, a esquerda).
        self.r_size = min(h - 40, w - 320)
        self.cx = self.r_size // 2 + 20
        self.cy = h // 2
        self.radius = self.r_size // 2 - 15
        # Inicio do painel (HUD), a direita do radar.
        self.px = self.r_size + 50

        self.f_big = pygame.font.SysFont("monospace", 20, bold=True)
        self.f_med = pygame.font.SysFont("monospace", 15)
        self.f_sml = pygame.font.SysFont("monospace", 12)

        self.scan = []
        self.sectors = [99.9] * 8
        self.scan_hz = 0.0
        self._last_scan_t = time.monotonic()

        self.bg = pygame.Surface((w, h))
        self._draw_bg()

    def _draw_bg(self):
        self.bg.fill(BG)
        cx, cy, r = self.cx, self.cy, self.radius
        n_rings = 6
        for i in range(1, n_rings + 1):
            rr = int(r * i / n_rings)
            pygame.draw.circle(self.bg, RING, (cx, cy), rr, 1)
        pygame.draw.line(self.bg, CROSS, (cx - r, cy), (cx + r, cy), 1)
        pygame.draw.line(self.bg, CROSS, (cx, cy - r), (cx, cy + r), 1)
        for i in range(8):
            a = math.radians(i * 45)
            ex = cx + int(r * math.cos(a))
            ey = cy - int(r * math.sin(a))
            pygame.draw.line(self.bg, (25, 30, 40), (cx, cy), (ex, ey), 1)
        sr = int(3.0 / self.max_range * r)
        if sr < r:
            pygame.draw.circle(self.bg, SAFETY_R, (cx, cy), sr, 1)

    def polar2px(self, angle_deg, dist_m):
        if dist_m > self.max_range or dist_m <= 0:
            return None
        s = self.radius / self.max_range
        rad = math.radians(angle_deg)
        x = self.cx + int(dist_m * math.cos(rad) * s)
        y = self.cy - int(dist_m * math.sin(rad) * s)
        return (x, y)

    def update_scan(self, scan):
        self.scan = scan
        now = time.monotonic()
        dt = now - self._last_scan_t
        self.scan_hz = (1.0 / dt) if dt > 0 else 0.0
        self._last_scan_t = now
        self.sectors = [99.9] * 8
        for angle, dist in scan:
            idx = int(angle / 45.0) % 8
            if dist < self.sectors[idx]:
                self.sectors[idx] = dist

    def zoom(self, delta):
        self.max_range = max(1.0, min(LIDAR_MAX_M, self.max_range + delta))
        self._draw_bg()

    def draw(self, steer_us, throttle_us, esc_armado, laco_fps, watchdog_ativo):
        self.screen.blit(self.bg, (0, 0))

        # Pontos do radar.
        for angle, dist in self.scan:
            pt = self.polar2px(angle, dist)
            if pt:
                c = color_for_dist(dist)
                sz = max(1, 4 - int(dist / self.max_range * 3))
                if sz <= 1:
                    self.screen.set_at(pt, c)
                else:
                    pygame.draw.circle(self.screen, c, pt, sz)

        pygame.draw.circle(self.screen, TXT_HI, (self.cx, self.cy), 4)

        # ---- Painel / HUD ----
        px, py = self.px, 15
        self.screen.blit(self.f_big.render("CONTROLE INTEGRADO", True, CYAN), (px, py)); py += 24
        self.screen.blit(self.f_sml.render("TCC Veiculos Autonomos", True, TXT_DIM), (px, py)); py += 24
        pygame.draw.line(self.screen, CROSS, (px, py), (self.w - 15, py)); py += 12

        # Estado do controle.
        cor_esc = (255, 80, 80) if esc_armado else (120, 200, 120)
        self.screen.blit(self.f_med.render("Steer:    {0} us".format(steer_us), True, TXT), (px, py)); py += 22
        self.screen.blit(self.f_med.render("Throttle: {0} us".format(throttle_us), True, TXT), (px, py)); py += 22
        self.screen.blit(self.f_med.render("ESC: {0}".format("ARMADO" if esc_armado else "DESARMADO"), True, cor_esc), (px, py)); py += 26

        pygame.draw.line(self.screen, CROSS, (px, py), (self.w - 15, py)); py += 12

        # Saude do laco (essencial p/ decidir se pode armar).
        cor_fps = GREEN if laco_fps >= 20 else (ORANGE if laco_fps >= 12 else RED)
        self.screen.blit(self.f_sml.render("Laco FPS:", True, TXT_DIM), (px, py))
        self.screen.blit(self.f_med.render("{0:.0f}".format(laco_fps), True, cor_fps), (px + 90, py - 1)); py += 20
        self.screen.blit(self.f_sml.render("Scan Hz:", True, TXT_DIM), (px, py))
        self.screen.blit(self.f_med.render("{0:.1f}".format(self.scan_hz), True, TXT), (px + 90, py - 1)); py += 20
        self.screen.blit(self.f_sml.render("Range:", True, TXT_DIM), (px, py))
        self.screen.blit(self.f_med.render("{0:.0f}m".format(self.max_range), True, TXT), (px + 90, py - 1)); py += 24

        if watchdog_ativo:
            self.screen.blit(self.f_med.render("! WATCHDOG: ESC neutro !", True, RED), (px, py)); py += 24

        pygame.draw.line(self.screen, CROSS, (px, py), (self.w - 15, py)); py += 12

        # Setor mais proximo (safety).
        mn = min(self.sectors)
        if mn < 3.0:
            self.screen.blit(self.f_med.render("! OBSTACULO {0:.2f}m !".format(mn), True, RED), (px, py)); py += 24
        else:
            self.screen.blit(self.f_med.render("Safety: OK", True, GREEN), (px, py)); py += 24

        pygame.draw.line(self.screen, CROSS, (px, py), (self.w - 15, py)); py += 12
        for txt in ["Setas: dirigir", "Espaco: parar tudo",
                    "[+/-]: zoom radar", "ESC/Q: sair"]:
            self.screen.blit(self.f_sml.render(txt, True, TXT_DIM), (px, py)); py += 16

        pygame.display.flip()


# ======================================================================
# MAIN
# ======================================================================
def main():
    # --- Conectar LiDAR ---
    porta = achar_porta_lidar()
    lidar = None
    parser = CoinD6Parser()
    if porta:
        try:
            lidar = serial.Serial(port=porta, baudrate=BAUD_RATE, timeout=0)
            lidar.reset_input_buffer()
            time.sleep(0.1)
            lidar.write(CMD_START)
            time.sleep(0.3)
            print("LiDAR conectado em {0}".format(porta))
        except Exception as e:
            print("AVISO: nao consegui abrir o LiDAR ({0}). Sigo sem radar.".format(e))
            lidar = None
    else:
        print("AVISO: nenhuma porta de LiDAR encontrada. Sigo sem radar.")

    app = AppIntegrado()

    # --- Estado dos atuadores ---
    steer_us = STEER_CENTER_US
    throttle_us = THROTTLE_NEUTRO_US
    set_us(SERVO_CHANNEL, steer_us)
    if ESC_ARMADO:
        armar_esc()
    else:
        set_us(ESC_CHANNEL, THROTTLE_NEUTRO_US)

    print("")
    print("=== App integrado iniciado ===")
    print("ESC armado: {0}".format(ESC_ARMADO))
    if ESC_ARMADO:
        print("RODAS NO AR e KILL SWITCH na mao.")
    else:
        print("ESC desarmado: valide a fluidez do radar + servo antes de armar.")
    print("")

    rodando = True
    t_frame_ant = time.monotonic()
    laco_fps = 0.0

    try:
        while rodando:
            t0 = time.monotonic()
            dt_frame = t0 - t_frame_ant
            t_frame_ant = t0
            if dt_frame > 0:
                # media simples p/ suavizar o numero exibido
                laco_fps = 0.8 * laco_fps + 0.2 * (1.0 / dt_frame)
            watchdog_ativo = dt_frame > WATCHDOG_S

            # ---- 1. Eventos ----
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    rodando = False
                if e.type == pygame.KEYDOWN:
                    if e.key in (pygame.K_ESCAPE, pygame.K_q):
                        rodando = False
                    elif e.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        app.zoom(-1.0)   # menos range = zoom in
                    elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        app.zoom(+1.0)

            # ---- 2. Teclado (hold-to-move) ----
            teclas = pygame.key.get_pressed()
            parada_total = teclas[pygame.K_SPACE]

            if parada_total:
                steer_alvo = STEER_CENTER_US
            elif teclas[pygame.K_LEFT]:
                steer_alvo = STEER_LEFT_US
            elif teclas[pygame.K_RIGHT]:
                steer_alvo = STEER_RIGHT_US
            else:
                steer_alvo = STEER_CENTER_US

            if (not ESC_ARMADO) or parada_total or watchdog_ativo:
                throttle_alvo = THROTTLE_NEUTRO_US
            elif teclas[pygame.K_UP]:
                throttle_alvo = THROTTLE_MIN_US
            elif teclas[pygame.K_DOWN]:
                throttle_alvo = THROTTLE_RE_US
            else:
                throttle_alvo = THROTTLE_NEUTRO_US

            # ---- 3. ATUADORES PRIMEIRO (antes do desenho pesado) ----
            steer_us = aproximar(steer_us, steer_alvo, STEER_STEP_US)
            set_us(SERVO_CHANNEL, steer_us)

            if ESC_ARMADO and not watchdog_ativo:
                throttle_us = aproximar(throttle_us, throttle_alvo, THROTTLE_STEP_US)
                throttle_us = max(THROTTLE_RE_US, min(THROTTLE_MAX_US, throttle_us))
                set_us(ESC_CHANNEL, throttle_us)
            else:
                throttle_us = THROTTLE_NEUTRO_US
                set_us(ESC_CHANNEL, THROTTLE_NEUTRO_US)

            # ---- 4. LiDAR: ler serial e atualizar scan ----
            if lidar is not None:
                try:
                    waiting = lidar.in_waiting
                    if waiting > 0:
                        data = lidar.read(min(waiting, 4096))
                        scans = parser.feed(data)
                        if scans:
                            app.update_scan(scans[-1])
                except Exception as e:
                    print("AVISO: erro lendo LiDAR: {0}".format(e))

            # ---- 5. Desenhar (parte pesada, por ultimo) ----
            app.draw(steer_us, throttle_us, ESC_ARMADO, laco_fps, watchdog_ativo)
            app.clock.tick(FPS)

    except KeyboardInterrupt:
        pass
    finally:
        print("Encerrando em estado seguro...")
        set_us(SERVO_CHANNEL, STEER_CENTER_US)
        set_us(ESC_CHANNEL, THROTTLE_NEUTRO_US)
        time.sleep(0.5)
        soltar(SERVO_CHANNEL)
        soltar(ESC_CHANNEL)
        if lidar is not None:
            try:
                lidar.write(CMD_STOP)
                time.sleep(0.05)
                lidar.close()
            except Exception:
                pass
        pygame.quit()
        print("Finalizado.")


if __name__ == "__main__":
    main()