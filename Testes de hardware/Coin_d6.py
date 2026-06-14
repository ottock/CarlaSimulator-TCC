#!/usr/bin/env python3
"""
COIN-D6 LiDAR — Visualizador Pygame
=====================================
Mostra o scan 360° do COIN-D6 em tempo real com interface grafica.

Requisitos:
  sudo pip3 install pyserial pygame

Uso:
  sudo python3 coin_d6_visual.py
  sudo python3 coin_d6_visual.py --port /dev/ttyUSB0
  sudo python3 coin_d6_visual.py --range 4.0
  sudo python3 coin_d6_visual.py --fullscreen
"""

import sys
import time
import math
import struct
import argparse
import glob
import os

try:
    import serial
except ImportError:
    print("[ERRO] pyserial nao encontrado: sudo pip3 install pyserial")
    sys.exit(1)

try:
    import pygame
    import pygame.gfxdraw
except ImportError:
    print("[ERRO] pygame nao encontrado: sudo pip3 install pygame")
    sys.exit(1)


# ============================================================
# Protocolo COIN-D6
# ============================================================
BAUD_RATE = 230400
CMD_START = bytes([0xAA, 0x55, 0xF0, 0x0F])
CMD_STOP  = bytes([0xAA, 0x55, 0xF5, 0x0A])
HEADER    = bytes([0xAA, 0x55])


# ============================================================
# Cores
# ============================================================
BLACK      = (15, 15, 20)
DARK_GRAY  = (30, 30, 38)
MID_GRAY   = (55, 55, 65)
LIGHT_GRAY = (120, 120, 135)
WHITE      = (220, 220, 230)

GREEN      = (0, 220, 100)
CYAN       = (0, 200, 220)
YELLOW     = (255, 210, 0)
ORANGE     = (255, 130, 0)
RED        = (255, 50, 50)
BLUE       = (60, 120, 255)
PURPLE     = (160, 80, 255)

RING_COLOR    = (35, 45, 55)
SECTOR_LINE   = (40, 40, 50)
GRID_COLOR    = (25, 30, 38)

# Gradiente por distancia (perto -> longe)
DIST_COLORS = [
    RED,        # < 1m
    ORANGE,     # 1-2m
    YELLOW,     # 2-3m
    GREEN,      # 3-5m
    CYAN,       # 5-8m
    BLUE,       # 8-12m
]


def dist_to_color(distance_m):
    """Retorna cor baseada na distancia."""
    if distance_m < 1.0:
        return DIST_COLORS[0]
    elif distance_m < 2.0:
        return DIST_COLORS[1]
    elif distance_m < 3.0:
        return DIST_COLORS[2]
    elif distance_m < 5.0:
        return DIST_COLORS[3]
    elif distance_m < 8.0:
        return DIST_COLORS[4]
    else:
        return DIST_COLORS[5]


# ============================================================
# Parser (mesmo do script anterior, validado)
# ============================================================
class CoinD6Parser:
    def __init__(self):
        self.buffer = bytearray()
        self.current_scan = []
        self.scan_count = 0
        self.point_count = 0
        self.parse_errors = 0
        self.raw_packets = 0

    def feed(self, data):
        self.buffer.extend(data)
        scans = []

        while len(self.buffer) >= 4:
            idx = self.buffer.find(HEADER)
            if idx < 0:
                self.buffer.clear()
                break
            if idx > 0:
                self.buffer = self.buffer[idx:]

            if len(self.buffer) < 4:
                break

            packet_type = self.buffer[2]
            sample_count = self.buffer[3]

            if sample_count == 0 or sample_count > 50:
                self.buffer = self.buffer[2:]
                self.parse_errors += 1
                continue

            expected_len = 2 + 1 + 1 + 2 + (sample_count * 3) + 2 + 2 + 1

            if len(self.buffer) < expected_len:
                break

            packet = self.buffer[:expected_len]
            self.buffer = self.buffer[expected_len:]
            self.raw_packets += 1

            try:
                start_angle = struct.unpack_from('<H', packet, 4)[0] * 0.01
                end_angle_offset = 4 + 2 + sample_count * 3
                end_angle = struct.unpack_from('<H', packet, end_angle_offset)[0] * 0.01

                angle_diff = end_angle - start_angle
                if angle_diff < 0:
                    angle_diff += 360.0
                angle_step = angle_diff / (sample_count - 1) if sample_count > 1 else 0

                for i in range(sample_count):
                    offset = 6 + i * 3
                    if offset + 2 >= len(packet):
                        break
                    distance_mm = struct.unpack_from('<H', packet, offset)[0]
                    intensity = packet[offset + 2]
                    distance_m = distance_mm / 1000.0
                    angle = (start_angle + i * angle_step) % 360.0

                    if 0 < distance_mm < 12000:
                        self.current_scan.append({
                            'angle': angle,
                            'distance': distance_m,
                            'intensity': intensity,
                        })
                        self.point_count += 1

                if len(self.current_scan) >= 350:
                    scans.append(self.current_scan)
                    self.current_scan = []
                    self.scan_count += 1

            except (struct.error, IndexError):
                self.parse_errors += 1

        return scans


# ============================================================
# Conexao serial
# ============================================================
def find_serial_port():
    for pattern in ['/dev/ttyUSB*', '/dev/ttyACM*']:
        ports = glob.glob(pattern)
        if ports:
            return sorted(ports)[0]
    return None


def connect_lidar(port):
    ser = serial.Serial(
        port=port, baudrate=BAUD_RATE,
        bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE, timeout=0.01,  # non-blocking
    )
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    time.sleep(0.1)
    ser.write(CMD_START)
    time.sleep(0.3)
    return ser


# ============================================================
# Visualizador Pygame
# ============================================================
class LidarVisualizer:
    def __init__(self, width=900, height=700, max_range=6.0, fullscreen=False):
        pygame.init()
        pygame.display.set_caption("COIN-D6 LiDAR — TCC Veiculos Autonomos")

        self.width = width
        self.height = height
        self.max_range = max_range

        flags = 0
        if fullscreen:
            flags = pygame.FULLSCREEN
            info = pygame.display.Info()
            self.width = info.current_w
            self.height = info.current_h

        self.screen = pygame.display.set_mode((self.width, self.height), flags)
        self.clock = pygame.time.Clock()

        # Area do radar (quadrado, centralizado na metade esquerda)
        self.radar_size = min(self.height - 80, self.width - 300) 
        self.radar_cx = self.radar_size // 2 + 30
        self.radar_cy = self.height // 2
        self.radar_radius = self.radar_size // 2 - 20

        # Painel lateral
        self.panel_x = self.radar_size + 60

        # Fontes
        self.font_big = pygame.font.SysFont("monospace", 22, bold=True)
        self.font_med = pygame.font.SysFont("monospace", 16)
        self.font_sml = pygame.font.SysFont("monospace", 13)

        # Dados
        self.last_scan = []
        self.sectors = [99.0] * 8
        self.scan_count = 0
        self.point_count = 0
        self.fps = 0
        self.scan_rate = 0
        self.last_scan_time = time.monotonic()
        self.parse_errors = 0

        # Pre-render do fundo do radar (nao muda)
        self.bg_surface = None
        self._render_background()

    def _render_background(self):
        """Pre-renderiza o fundo estatico do radar."""
        self.bg_surface = pygame.Surface((self.width, self.height))
        self.bg_surface.fill(BLACK)

        cx, cy, r = self.radar_cx, self.radar_cy, self.radar_radius

        # Aneis de distancia
        num_rings = 6
        for i in range(1, num_rings + 1):
            ring_r = int(r * i / num_rings)
            pygame.draw.circle(self.bg_surface, RING_COLOR, (cx, cy), ring_r, 1)
            # Label de distancia
            dist_label = f"{self.max_range * i / num_rings:.0f}m"
            txt = self.font_sml.render(dist_label, True, LIGHT_GRAY)
            self.bg_surface.blit(txt, (cx + ring_r + 3, cy - 8))

        # Linhas dos 8 setores (cada 45°)
        for i in range(8):
            angle = math.radians(i * 45)
            x1 = cx + int(r * math.cos(angle))
            y1 = cy - int(r * math.sin(angle))
            pygame.draw.line(self.bg_surface, SECTOR_LINE, (cx, cy), (x1, y1), 1)

        # Cruz central
        pygame.draw.line(self.bg_surface, MID_GRAY, (cx - r, cy), (cx + r, cy), 1)
        pygame.draw.line(self.bg_surface, MID_GRAY, (cx, cy - r), (cx, cy + r), 1)

        # Labels de direcao
        dirs = [("0°", cx + r + 5, cy - 8),
                ("90°", cx - 12, cy - r - 18),
                ("180°", cx - r - 35, cy - 8),
                ("270°", cx - 12, cy + r + 4)]
        for label, x, y in dirs:
            txt = self.font_sml.render(label, True, LIGHT_GRAY)
            self.bg_surface.blit(txt, (x, y))

    def polar_to_screen(self, angle_deg, distance_m):
        """Converte coordenada polar para pixel na tela."""
        if distance_m > self.max_range:
            return None, None
        rad = math.radians(angle_deg)
        scale = self.radar_radius / self.max_range
        x = self.radar_cx + int(distance_m * math.cos(rad) * scale)
        y = self.radar_cy - int(distance_m * math.sin(rad) * scale)
        return x, y

    def update(self, scan, parser):
        """Atualiza com novos dados."""
        if scan:
            self.last_scan = scan
            self.scan_count = parser.scan_count
            self.point_count = parser.point_count
            self.parse_errors = parser.parse_errors

            now = time.monotonic()
            dt = now - self.last_scan_time
            if dt > 0:
                self.scan_rate = 1.0 / dt
            self.last_scan_time = now

            # Calcular setores (8 x 45°)
            self.sectors = [99.0] * 8
            for pt in scan:
                idx = int(pt['angle'] / 45.0) % 8
                self.sectors[idx] = min(self.sectors[idx], pt['distance'])

    def draw(self):
        """Desenha tudo."""
        # Fundo pre-renderizado
        self.screen.blit(self.bg_surface, (0, 0))

        cx, cy, r = self.radar_cx, self.radar_cy, self.radar_radius

        # ── Pontos do scan ──
        for pt in self.last_scan:
            x, y = self.polar_to_screen(pt['angle'], pt['distance'])
            if x is not None:
                color = dist_to_color(pt['distance'])
                # Ponto com tamanho variavel (mais perto = maior)
                size = max(1, 4 - int(pt['distance'] / self.max_range * 3))
                if size <= 1:
                    self.screen.set_at((x, y), color)
                else:
                    pygame.draw.circle(self.screen, color, (x, y), size)

        # ── Centro (posicao do LiDAR) ──
        pygame.draw.circle(self.screen, WHITE, (cx, cy), 5, 0)
        pygame.draw.circle(self.screen, CYAN, (cx, cy), 5, 1)

        # ── Zona de safety net (3m) ──
        safety_r = int(3.0 / self.max_range * r)
        pygame.draw.circle(self.screen, (60, 20, 20), (cx, cy), safety_r, 1)

        # ══════════════════════════════
        # Painel lateral
        # ══════════════════════════════
        px = self.panel_x
        py = 20

        # Titulo
        title = self.font_big.render("COIN-D6 LiDAR", True, CYAN)
        self.screen.blit(title, (px, py))
        py += 28

        subtitle = self.font_sml.render("TCC — Veiculos Autonomos", True, LIGHT_GRAY)
        self.screen.blit(subtitle, (px, py))
        py += 30

        # Separador
        pygame.draw.line(self.screen, MID_GRAY, (px, py), (self.width - 20, py), 1)
        py += 15

        # Stats
        stats = [
            ("Scans",    f"{self.scan_count}"),
            ("Pontos",   f"{len(self.last_scan)}"),
            ("Scan Hz",  f"{self.scan_rate:.1f}"),
            ("FPS",      f"{self.fps:.0f}"),
            ("Range",    f"{self.max_range:.1f}m"),
            ("Erros",    f"{self.parse_errors}"),
        ]
        for label, value in stats:
            txt_l = self.font_sml.render(f"{label}:", True, LIGHT_GRAY)
            txt_v = self.font_med.render(value, True, WHITE)
            self.screen.blit(txt_l, (px, py))
            self.screen.blit(txt_v, (px + 80, py - 1))
            py += 22

        py += 10
        pygame.draw.line(self.screen, MID_GRAY, (px, py), (self.width - 20, py), 1)
        py += 15

        # ── Distancias por setor ──
        header = self.font_med.render("Dist. por Setor", True, YELLOW)
        self.screen.blit(header, (px, py))
        py += 25

        sector_labels = [
            "0°   Frente ",
            "45°  Fr-Dir ",
            "90°  Direita",
            "135° Tr-Dir ",
            "180° Tras   ",
            "225° Tr-Esq ",
            "270° Esquer.",
            "315° Fr-Esq ",
        ]

        bar_max_w = self.width - px - 100

        for i, label in enumerate(sector_labels):
            dist = self.sectors[i]
            dist_str = f"{dist:.1f}m" if dist < 50 else " --"

            # Cor baseada na distancia
            if dist < 1.0:
                color = RED
            elif dist < 3.0:
                color = ORANGE
            elif dist < 5.0:
                color = YELLOW
            else:
                color = GREEN

            # Label
            txt = self.font_sml.render(label, True, LIGHT_GRAY)
            self.screen.blit(txt, (px, py))

            # Valor
            txt_v = self.font_sml.render(dist_str, True, color)
            self.screen.blit(txt_v, (px + 110, py))

            # Barra
            if dist < 50:
                bar_w = max(1, int((dist / self.max_range) * bar_max_w))
                bar_w = min(bar_w, bar_max_w)
                bar_rect = pygame.Rect(px + 160, py + 2, bar_w, 12)
                pygame.draw.rect(self.screen, color, bar_rect)
                pygame.draw.rect(self.screen, DARK_GRAY, bar_rect, 1)

            py += 20

        py += 15
        pygame.draw.line(self.screen, MID_GRAY, (px, py), (self.width - 20, py), 1)
        py += 15

        # ── Safety net status ──
        min_dist = min(self.sectors)
        if min_dist < 3.0:
            safety_txt = self.font_med.render("! SAFETY NET ATIVO !", True, RED)
            safety_detail = self.font_sml.render(
                f"Obstaculo a {min_dist:.2f}m — freio!", True, ORANGE)
            self.screen.blit(safety_txt, (px, py))
            self.screen.blit(safety_detail, (px, py + 22))
        else:
            safety_txt = self.font_med.render("Safety net: OK", True, GREEN)
            self.screen.blit(safety_txt, (px, py))
        py += 45

        # ── Legenda de cores ──
        legend = self.font_sml.render("Legenda:", True, LIGHT_GRAY)
        self.screen.blit(legend, (px, py))
        py += 18

        legend_items = [
            (RED,    "< 1m"),
            (ORANGE, "1-2m"),
            (YELLOW, "2-3m"),
            (GREEN,  "3-5m"),
            (CYAN,   "5-8m"),
            (BLUE,   "8-12m"),
        ]

        for color, label in legend_items:
            pygame.draw.rect(self.screen, color, (px, py + 2, 12, 12))
            txt = self.font_sml.render(label, True, LIGHT_GRAY)
            self.screen.blit(txt, (px + 18, py))
            py += 17

        py += 10

        # ── Controles ──
        controls = [
            "[+/-]  Zoom in/out",
            "[R]    Reset range",
            "[S]    Screenshot",
            "[ESC]  Sair",
        ]
        for ctrl in controls:
            txt = self.font_sml.render(ctrl, True, MID_GRAY)
            self.screen.blit(txt, (px, py))
            py += 16

        # Circulo vermelho de safety net (legenda)
        safety_label = self.font_sml.render("Circulo vermelho = 3m (safety net)", True, (60, 20, 20))
        self.screen.blit(safety_label, (30, self.height - 20))

        pygame.display.flip()
        self.fps = self.clock.get_fps()
        self.clock.tick(30)

    def handle_events(self):
        """Retorna False se deve sair."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    return False
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.max_range = max(1.0, self.max_range - 1.0)
                    self._render_background()
                elif event.key == pygame.K_MINUS:
                    self.max_range = min(12.0, self.max_range + 1.0)
                    self._render_background()
                elif event.key == pygame.K_r:
                    self.max_range = 6.0
                    self._render_background()
                elif event.key == pygame.K_s:
                    fname = f"lidar_screenshot_{int(time.time())}.png"
                    pygame.image.save(self.screen, fname)
                    print(f"  Screenshot salvo: {fname}")
        return True


# ============================================================
# Main loop
# ============================================================
def main():
    argp = argparse.ArgumentParser(description="Visualizador COIN-D6 LiDAR")
    argp.add_argument("--port", type=str, default=None)
    argp.add_argument("--range", type=float, default=6.0,
                      help="Range maximo em metros (default: 6)")
    argp.add_argument("--fullscreen", action="store_true")
    args = argp.parse_args()

    # Porta serial
    port = args.port or find_serial_port()
    if not port:
        print("[ERRO] Porta serial nao encontrada. Use --port /dev/ttyUSB0")
        sys.exit(1)

    print(f"  Porta: {port}")
    print(f"  Range: {args.range}m")
    print(f"  Conectando ao COIN-D6...")

    ser = connect_lidar(port)
    parser = CoinD6Parser()
    viz = LidarVisualizer(max_range=args.range, fullscreen=args.fullscreen)

    print(f"  Visualizador iniciado! Janela aberta.")
    print(f"  [ESC] para sair | [+/-] zoom | [S] screenshot\n")

    running = True
    try:
        while running:
            # Ler dados seriais
            data = ser.read(1024)
            if data:
                scans = parser.feed(data)
                if scans:
                    viz.update(scans[-1], parser)

            # Eventos e desenho
            running = viz.handle_events()
            viz.draw()

    except KeyboardInterrupt:
        pass
    finally:
        print("\n  Encerrando...")
        ser.write(CMD_STOP)
        time.sleep(0.1)
        ser.close()
        pygame.quit()
        print("  Pronto!")


if __name__ == "__main__":
    main()