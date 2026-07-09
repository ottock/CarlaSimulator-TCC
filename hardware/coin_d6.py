#!/usr/bin/env python3
"""
COIN-D6 LiDAR — Visualizador Corrigido
========================================
Parser baseado na analise real dos pacotes do COIN-D6.

Formato do pacote (85 bytes tipico):
  [0-1]  AA 55         header
  [2]    00             tipo
  [3]    19 (=25)       quantidade de pontos
  [4-5]  start angle    LE, 0.01 graus
  [6-7]  end angle      LE, 0.01 graus
  [8-9]  timestamp/crc
  [10+]  dados: [intensidade, dist_lo, dist_hi] x N
         distancia em mm, LE. 0 ou 1 = sem leitura.

Uso:
  sudo python3 coin_d6_v2.py
  sudo python3 coin_d6_v2.py --range 4
  sudo python3 coin_d6_v2.py --fullscreen
"""

import sys
import time
import math
import struct
import argparse
import glob

try:
    import serial
except ImportError:
    print("Instale: sudo pip3 install pyserial")
    sys.exit(1)

try:
    import pygame
except ImportError:
    print("Instale: sudo pip3 install pygame")
    sys.exit(1)


BAUD_RATE = 230400
CMD_START = bytes([0xAA, 0x55, 0xF0, 0x0F])
CMD_STOP  = bytes([0xAA, 0x55, 0xF5, 0x0A])


# ============================================================
# Parser corrigido
# ============================================================
class CoinD6Parser:
    def __init__(self):
        self.buffer = bytearray()
        self.current_scan = []
        self.scan_count = 0
        self.point_count = 0
        self.parse_errors = 0

    def feed(self, data):
        self.buffer.extend(data)
        scans = []

        while True:
            # Procurar header AA 55
            idx = self.buffer.find(b'\xAA\x55')
            if idx < 0 or idx + 4 > len(self.buffer):
                if idx < 0:
                    self.buffer = self.buffer[-1:]  # manter ultimo byte
                break
            if idx > 0:
                self.buffer = self.buffer[idx:]

            # Ler count no byte [3]
            sample_count = self.buffer[3]
            if sample_count == 0 or sample_count > 50:
                self.buffer = self.buffer[2:]
                self.parse_errors += 1
                continue

            # Tamanho: header(2) + type(1) + count(1) + start(2) + end(2) + extra(2) + data(N*3)
            pkt_len = 10 + sample_count * 3

            if len(self.buffer) < pkt_len:
                break  # esperar mais dados

            pkt = bytes(self.buffer[:pkt_len])
            self.buffer = self.buffer[pkt_len:]

            # Parse angulos (LE, unidade 0.01 grau)
            start_angle = struct.unpack_from('<H', pkt, 4)[0] * 0.01
            end_angle   = struct.unpack_from('<H', pkt, 6)[0] * 0.01

            # Normalizar angulos > 360
            start_angle = start_angle % 360.0
            end_angle   = end_angle % 360.0

            # Calcular step
            angle_diff = end_angle - start_angle
            if angle_diff < 0:
                angle_diff += 360.0
            if angle_diff > 180:  # pacote cruzando 0 graus
                angle_diff -= 360.0
            step = angle_diff / max(sample_count - 1, 1)

            # Parse dados: cada ponto = [intensity, dist_lo, dist_hi]
            for i in range(sample_count):
                offset = 10 + i * 3
                if offset + 2 >= len(pkt):
                    break

                intensity = pkt[offset]
                dist_mm   = pkt[offset + 1] | (pkt[offset + 2] << 8)

                # Filtrar: 0 ou 1 mm = sem leitura
                if dist_mm <= 1:
                    continue

                dist_m = dist_mm / 1000.0
                angle  = (start_angle + i * step) % 360.0

                if dist_m <= 12.0:
                    self.current_scan.append((angle, dist_m, intensity))
                    self.point_count += 1

            # Detectar scan completo (~400 pontos por volta, 10Hz * ~16 pacotes)
            if len(self.current_scan) >= 300:
                scans.append(self.current_scan)
                self.current_scan = []
                self.scan_count += 1

        return scans


# ============================================================
# Cores
# ============================================================
BG        = (12, 12, 18)
RING      = (30, 38, 48)
CROSS     = (40, 45, 55)
TXT_DIM   = (100, 105, 120)
TXT       = (190, 195, 210)
TXT_HI    = (230, 235, 245)
CYAN      = (0, 200, 220)
GREEN     = (0, 210, 90)
YELLOW    = (255, 210, 0)
ORANGE    = (255, 130, 0)
RED       = (255, 50, 50)
BLUE      = (60, 130, 255)
SAFETY_R  = (80, 25, 25)

DIST_COLORS = [RED, ORANGE, YELLOW, GREEN, CYAN, BLUE]
DIST_THRESHOLDS = [1.0, 2.0, 3.0, 5.0, 8.0]

def color_for_dist(d):
    for i, t in enumerate(DIST_THRESHOLDS):
        if d < t:
            return DIST_COLORS[i]
    return DIST_COLORS[-1]


# ============================================================
# Visualizador
# ============================================================
class Visualizer:
    def __init__(self, w=900, h=680, max_range=12.0, fullscreen=False):
        pygame.init()
        pygame.display.set_caption("COIN-D6 LiDAR")

        if fullscreen:
            info = pygame.display.Info()
            w, h = info.current_w, info.current_h
            self.screen = pygame.display.set_mode((w, h), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((w, h))

        self.w, self.h = w, h
        self.max_range = max_range
        self.clock = pygame.time.Clock()

        # Radar area
        self.r_size = min(h - 40, w - 280)
        self.cx = self.r_size // 2 + 20
        self.cy = h // 2
        self.radius = self.r_size // 2 - 15

        # Panel
        self.px = self.r_size + 50

        # Fontes
        self.f_big = pygame.font.SysFont("monospace", 20, bold=True)
        self.f_med = pygame.font.SysFont("monospace", 15)
        self.f_sml = pygame.font.SysFont("monospace", 12)

        # State
        self.scan = []
        self.sectors = [99.9] * 8
        self.stats = {"scans": 0, "pts": 0, "hz": 0.0, "errs": 0}
        self.last_t = time.monotonic()

        # Pre-render background
        self.bg = pygame.Surface((w, h))
        self._draw_bg()

    def _draw_bg(self):
        self.bg.fill(BG)
        cx, cy, r = self.cx, self.cy, self.radius
        n_rings = 6

        for i in range(1, n_rings + 1):
            rr = int(r * i / n_rings)
            pygame.draw.circle(self.bg, RING, (cx, cy), rr, 1)
            # Rotulos numericos dos aneis removidos a pedido (visual mais limpo).

        # Cross
        pygame.draw.line(self.bg, CROSS, (cx - r, cy), (cx + r, cy), 1)
        pygame.draw.line(self.bg, CROSS, (cx, cy - r), (cx, cy + r), 1)

        # Sector lines
        for i in range(8):
            a = math.radians(i * 45)
            ex = cx + int(r * math.cos(a))
            ey = cy - int(r * math.sin(a))
            pygame.draw.line(self.bg, (25, 30, 40), (cx, cy), (ex, ey), 1)

        # Safety ring (3m)
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

    def update(self, scan, parser):
        self.scan = scan
        self.stats["scans"] = parser.scan_count
        self.stats["pts"] = len(scan)
        self.stats["errs"] = parser.parse_errors
        now = time.monotonic()
        dt = now - self.last_t
        self.stats["hz"] = 1.0 / dt if dt > 0 else 0
        self.last_t = now

        self.sectors = [99.9] * 8
        for angle, dist, _ in scan:
            idx = int(angle / 45.0) % 8
            if dist < self.sectors[idx]:
                self.sectors[idx] = dist

    def draw(self):
        self.screen.blit(self.bg, (0, 0))
        cx, cy = self.cx, self.cy

        # Points
        for angle, dist, intensity in self.scan:
            pt = self.polar2px(angle, dist)
            if pt:
                c = color_for_dist(dist)
                sz = max(1, 4 - int(dist / self.max_range * 3))
                if sz <= 1:
                    self.screen.set_at(pt, c)
                else:
                    pygame.draw.circle(self.screen, c, pt, sz)

        # Center dot
        pygame.draw.circle(self.screen, TXT_HI, (cx, cy), 4)
        pygame.draw.circle(self.screen, CYAN, (cx, cy), 4, 1)

        # === Panel ===
        px, py = self.px, 15

        t = self.f_big.render("COIN-D6 LiDAR", True, CYAN)
        self.screen.blit(t, (px, py)); py += 22
        t = self.f_sml.render("TCC Veiculos Autonomos", True, TXT_DIM)
        self.screen.blit(t, (px, py)); py += 22

        pygame.draw.line(self.screen, CROSS, (px, py), (self.w - 15, py)); py += 10

        for label, val in [
            ("Scans",   str(self.stats["scans"])),
            ("Pontos",  str(self.stats["pts"])),
            ("Scan Hz", f"{self.stats['hz']:.1f}"),
            ("FPS",     f"{self.clock.get_fps():.0f}"),
            ("Range",   f"{self.max_range:.1f}m"),
        ]:
            self.screen.blit(self.f_sml.render(f"{label}:", True, TXT_DIM), (px, py))
            self.screen.blit(self.f_med.render(val, True, TXT), (px + 75, py - 1))
            py += 20

        py += 8
        pygame.draw.line(self.screen, CROSS, (px, py), (self.w - 15, py)); py += 10

        self.screen.blit(self.f_med.render("Dist. por setor", True, YELLOW), (px, py)); py += 22

        names = ["0 Frente","45 Fr-Dir","90 Direita","135 Tr-Dir",
                 "180 Tras","225 Tr-Esq","270 Esquer.","315 Fr-Esq"]
        bw = self.w - px - 105

        for i, name in enumerate(names):
            d = self.sectors[i]
            c = color_for_dist(d) if d < 50 else TXT_DIM
            ds = f"{d:.1f}m" if d < 50 else " --"

            self.screen.blit(self.f_sml.render(name, True, TXT_DIM), (px, py))
            self.screen.blit(self.f_sml.render(ds, True, c), (px + 95, py))

            if d < 50:
                bar = max(1, min(int(d / self.max_range * bw), bw))
                pygame.draw.rect(self.screen, c, (px + 145, py + 2, bar, 11))

            py += 18

        py += 8
        pygame.draw.line(self.screen, CROSS, (px, py), (self.w - 15, py)); py += 10

        mn = min(self.sectors)
        if mn < 3.0:
            self.screen.blit(self.f_med.render("! SAFETY NET !", True, RED), (px, py))
            self.screen.blit(self.f_sml.render(f"Obstaculo a {mn:.2f}m", True, ORANGE), (px, py + 20))
        else:
            self.screen.blit(self.f_med.render("Safety: OK", True, GREEN), (px, py))
        py += 42

        # Legend
        pygame.draw.line(self.screen, CROSS, (px, py), (self.w - 15, py)); py += 10
        items = [(RED,"<1m"),(ORANGE,"1-2"),(YELLOW,"2-3"),(GREEN,"3-5"),(CYAN,"5-8"),(BLUE,"8-12")]
        for c, lb in items:
            pygame.draw.rect(self.screen, c, (px, py + 1, 10, 10))
            self.screen.blit(self.f_sml.render(lb, True, TXT_DIM), (px + 14, py))
            py += 15

        py += 8
        for txt in ["[+/-] Zoom", "[S] Screenshot", "[ESC] Sair"]:
            self.screen.blit(self.f_sml.render(txt, True, (40, 42, 50)), (px, py))
            py += 14

        # Footer
        self.screen.blit(self.f_sml.render("Circulo = 3m safety net", True, SAFETY_R),
                         (25, self.h - 18))

        pygame.display.flip()
        self.clock.tick(30)

    def events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
            if e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                if e.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self.max_range = max(1.0, self.max_range - 1.0)
                    self._draw_bg()
                if e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.max_range = min(12.0, self.max_range + 1.0)
                    self._draw_bg()
                if e.key == pygame.K_r:
                    self.max_range = 12.0
                    self._draw_bg()
                if e.key == pygame.K_s:
                    fn = f"lidar_{int(time.time())}.png"
                    pygame.image.save(self.screen, fn)
                    print(f"  Salvo: {fn}")
        return True


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None)
    ap.add_argument("--range", type=float, default=12.0)
    ap.add_argument("--fullscreen", action="store_true")
    args = ap.parse_args()

    port = args.port
    if not port:
        for pat in ['/dev/ttyUSB*', '/dev/ttyACM*']:
            found = glob.glob(pat)
            if found:
                port = sorted(found)[0]
                break
    if not port:
        print("Porta nao encontrada. Use --port /dev/ttyUSB0")
        sys.exit(1)

    print(f"  Porta: {port}")
    ser = serial.Serial(port=port, baudrate=BAUD_RATE, timeout=0)
    ser.reset_input_buffer()
    time.sleep(0.1)
    ser.write(CMD_START)
    time.sleep(0.3)

    parser = CoinD6Parser()
    viz = Visualizer(max_range=args.range, fullscreen=args.fullscreen)

    print("  Rodando! [ESC] sair | [+/-] zoom | [S] screenshot")

    try:
        while viz.events():
            # Ler tudo disponivel no buffer serial (rapido, sem lag)
            waiting = ser.in_waiting
            if waiting > 0:
                data = ser.read(min(waiting, 4096))
                scans = parser.feed(data)
                if scans:
                    viz.update(scans[-1], parser)
            viz.draw()
    except KeyboardInterrupt:
        pass
    finally:
        ser.write(CMD_STOP)
        time.sleep(0.05)
        ser.close()
        pygame.quit()
        print("  Encerrado.")

if __name__ == "__main__":
    main()