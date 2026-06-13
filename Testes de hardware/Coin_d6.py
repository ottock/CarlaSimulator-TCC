#!/usr/bin/env python3
"""
COIN-D6 LiDAR Test Script para Jetson Nano
============================================
Testa o LiDAR COIN-D6 (Guoke Optoelectronics) via USB serial (CH340).

COMO CONECTAR:
  1. Plugue o adaptador CH340 USB no Jetson Nano
  2. O LiDAR deve comecar a girar automaticamente ao receber 5V pelo USB
  3. Rode: sudo python3 coin_d6_test.py

MODOS:
  python3 coin_d6_test.py                  # Auto-detecta porta e mostra scan
  python3 coin_d6_test.py --port /dev/ttyUSB0  # Porta especifica
  python3 coin_d6_test.py --raw             # Modo hex dump (debug)
  python3 coin_d6_test.py --csv 30          # Grava 30s de dados em CSV

PROTOCOLO (do manual):
  Baud rate: 230400
  Start:  AA 55 F0 0F
  Stop:   AA 55 F5 0A
  Fios:   Amarelo=5V, Branco=GND, Vermelho=TXD, Preto=RXD
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
    print("[ERRO] pyserial nao encontrado.")
    print("       Instale com: pip3 install pyserial")
    print("       No Jetson:   sudo pip3 install pyserial")
    sys.exit(1)


# ============================================================
# Constantes do protocolo COIN-D6
# ============================================================
BAUD_RATE = 230400
CMD_START = bytes([0xAA, 0x55, 0xF0, 0x0F])
CMD_STOP  = bytes([0xAA, 0x55, 0xF5, 0x0A])
HEADER    = bytes([0xAA, 0x55])


def find_serial_port():
    """Auto-detecta a porta serial do CH340."""
    patterns = [
        '/dev/ttyUSB*',
        '/dev/ttyACM*',
        '/dev/serial/by-id/*CH340*',
    ]
    for pattern in patterns:
        ports = glob.glob(pattern)
        if ports:
            return sorted(ports)[0]
    return None


def connect_lidar(port, timeout=2.0):
    """Conecta ao LiDAR e envia comando de start."""
    print(f"  Conectando em {port} @ {BAUD_RATE} baud...")
    ser = serial.Serial(
        port=port,
        baudrate=BAUD_RATE,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=timeout,
    )

    # Limpar buffer
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    time.sleep(0.1)

    # Enviar comando de start
    print(f"  Enviando comando START (AA 55 F0 0F)...")
    ser.write(CMD_START)
    time.sleep(0.5)

    # Verificar se dados estao chegando
    waiting = ser.in_waiting
    if waiting > 0:
        print(f"  [OK] Dados chegando! ({waiting} bytes no buffer)")
    else:
        print(f"  [!!] Nenhum dado recebido ainda. Aguardando...")
        time.sleep(1.0)
        waiting = ser.in_waiting
        if waiting > 0:
            print(f"  [OK] Dados chegando agora ({waiting} bytes)")
        else:
            print(f"  [!!] Sem dados. Verifique conexao e alimentacao 5V.")

    return ser


def stop_lidar(ser):
    """Envia comando de stop."""
    try:
        ser.write(CMD_STOP)
        time.sleep(0.1)
        ser.close()
        print("  LiDAR parado e porta fechada.")
    except Exception:
        pass


# ============================================================
# Modo 1: Hex Dump (debug / descobrir protocolo)
# ============================================================
def run_hex_dump(ser, duration=10):
    """Le bytes brutos e mostra hex dump para debug."""
    print(f"\n  === MODO HEX DUMP ({duration}s) ===")
    print(f"  Procurando por headers AA 55...\n")

    start_time = time.monotonic()
    total_bytes = 0
    packet_count = 0

    try:
        while (time.monotonic() - start_time) < duration:
            data = ser.read(256)
            if not data:
                continue

            total_bytes += len(data)

            # Mostrar hex com destaque nos headers
            hex_str = ""
            for i, b in enumerate(data):
                if i < len(data) - 1 and data[i] == 0xAA and data[i+1] == 0x55:
                    hex_str += f"\n  >>> HEADER >>> "
                    packet_count += 1
                hex_str += f"{b:02X} "

            print(hex_str, end="", flush=True)

    except KeyboardInterrupt:
        pass

    elapsed = time.monotonic() - start_time
    print(f"\n\n  --- Resumo ---")
    print(f"  Tempo:     {elapsed:.1f}s")
    print(f"  Bytes:     {total_bytes}")
    print(f"  Rate:      {total_bytes/elapsed:.0f} bytes/s")
    print(f"  Headers:   {packet_count} pacotes AA 55 encontrados")


# ============================================================
# Modo 2: Parse de pacotes CSPC
# ============================================================
class CoinD6Parser:
    """
    Parser para o protocolo CSPC do COIN-D6.

    Formato tipico de pacote CSPC:
      [AA 55] [type] [sample_count] [start_angle_lo] [start_angle_hi]
      [dados: distance_lo distance_hi intensity] * sample_count
      [end_angle_lo] [end_angle_hi]
      [timestamp_lo] [timestamp_hi]
      [crc]

    Se o formato real for diferente, o modo --raw ajuda a descobrir.
    """

    def __init__(self):
        self.buffer = bytearray()
        self.scan_data = {}  # angulo -> distancia
        self.current_scan = []
        self.scan_count = 0
        self.point_count = 0
        self.parse_errors = 0
        self.raw_packets = 0

    def feed(self, data):
        """Alimenta bytes e retorna scans completos."""
        self.buffer.extend(data)
        scans = []

        while len(self.buffer) >= 4:
            # Procurar header AA 55
            idx = self.buffer.find(HEADER)
            if idx < 0:
                self.buffer.clear()
                break
            if idx > 0:
                self.buffer = self.buffer[idx:]

            # Precisamos de pelo menos o header + type + count
            if len(self.buffer) < 4:
                break

            packet_type = self.buffer[2]
            sample_count = self.buffer[3]

            # Validar sample_count razoavel (1-40 pontos por pacote)
            if sample_count == 0 or sample_count > 50:
                # Nao eh um pacote valido, pular esse header
                self.buffer = self.buffer[2:]
                self.parse_errors += 1
                continue

            # Tamanho esperado: header(2) + type(1) + count(1) +
            # start_angle(2) + [distance(2) + intensity(1)] * count +
            # end_angle(2) + timestamp(2) + crc(1)
            expected_len = 2 + 1 + 1 + 2 + (sample_count * 3) + 2 + 2 + 1

            if len(self.buffer) < expected_len:
                break  # Esperar mais dados

            # Extrair pacote
            packet = self.buffer[:expected_len]
            self.buffer = self.buffer[expected_len:]
            self.raw_packets += 1

            # Parse
            try:
                start_angle = struct.unpack_from('<H', packet, 4)[0] * 0.01
                end_angle_offset = 4 + 2 + sample_count * 3
                end_angle = struct.unpack_from('<H', packet, end_angle_offset)[0] * 0.01

                # Calcular step angular
                angle_diff = end_angle - start_angle
                if angle_diff < 0:
                    angle_diff += 360.0
                if sample_count > 1:
                    angle_step = angle_diff / (sample_count - 1)
                else:
                    angle_step = 0

                for i in range(sample_count):
                    offset = 6 + i * 3
                    if offset + 2 >= len(packet):
                        break
                    distance_mm = struct.unpack_from('<H', packet, offset)[0]
                    intensity = packet[offset + 2]
                    distance_m = distance_mm / 1000.0

                    angle = start_angle + i * angle_step
                    if angle >= 360:
                        angle -= 360

                    if 0 < distance_mm < 12000:  # Filtrar dados invalidos
                        self.current_scan.append({
                            'angle': angle,
                            'distance': distance_m,
                            'intensity': intensity,
                        })
                        self.point_count += 1

                # Detectar scan completo (volta completa ~ 400 pontos)
                if len(self.current_scan) >= 350:
                    scans.append(self.current_scan)
                    self.current_scan = []
                    self.scan_count += 1

            except (struct.error, IndexError):
                self.parse_errors += 1

        return scans


def angle_to_grid(angle, distance, grid_size=40, max_range=6.0):
    """Converte coordenada polar para posicao no grid ASCII."""
    if distance > max_range or distance <= 0:
        return None, None
    rad = math.radians(angle)
    x = distance * math.cos(rad)
    y = distance * math.sin(rad)
    gx = int((x / max_range) * (grid_size // 2) + grid_size // 2)
    gy = int((-y / max_range) * (grid_size // 2) + grid_size // 2)
    if 0 <= gx < grid_size and 0 <= gy < grid_size:
        return gx, gy
    return None, None


def render_ascii_scan(scan_points, grid_size=41, max_range=6.0):
    """Renderiza um scan como mapa ASCII no terminal."""
    grid = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]
    center = grid_size // 2

    # Marcar centro (posicao do LiDAR)
    grid[center][center] = '◉'

    # Plotar pontos
    for pt in scan_points:
        gx, gy = angle_to_grid(pt['angle'], pt['distance'], grid_size, max_range)
        if gx is not None:
            if pt['distance'] < 1.0:
                grid[gy][gx] = '█'  # Muito perto
            elif pt['distance'] < 3.0:
                grid[gy][gx] = '▓'  # Perto
            else:
                grid[gy][gx] = '░'  # Longe

    lines = []
    for row in grid:
        lines.append('  ' + ''.join(row))
    return '\n'.join(lines)


def clear_screen():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def run_scan_display(ser, max_range=6.0):
    """Modo principal: mostra scan em tempo real."""
    parser = CoinD6Parser()
    last_scan = None
    start_time = time.monotonic()

    print("\n  Recebendo dados do COIN-D6...")
    print("  Aguardando primeiro scan completo...\n")

    try:
        while True:
            data = ser.read(512)
            if not data:
                continue

            scans = parser.feed(data)

            if scans:
                last_scan = scans[-1]

            # Atualizar display a cada ~200ms
            elapsed = time.monotonic() - start_time
            if last_scan and elapsed > 0.5:
                clear_screen()
                print("=" * 60)
                print("  COIN-D6 LiDAR — Scan em Tempo Real")
                print("  TCC: Digital Twins para Veiculos Autonomos")
                print("=" * 60)
                print()

                # Estatisticas
                print(f"  Scans completos: {parser.scan_count}")
                print(f"  Pontos totais:   {parser.point_count}")
                print(f"  Pacotes raw:     {parser.raw_packets}")
                print(f"  Erros de parse:  {parser.parse_errors}")
                print(f"  Pontos no scan:  {len(last_scan)}")
                print()

                # Distancias por regiao (util pro projeto)
                sectors = [999.0] * 8
                for pt in last_scan:
                    a = pt['angle']
                    d = pt['distance']
                    # Mapear 0-360 para 8 setores
                    sector_idx = int(a / 45.0) % 8
                    sectors[sector_idx] = min(sectors[sector_idx], d)

                labels = ["Frente", "Fr-Dir", "Direita", "Tr-Dir",
                          "Tras  ", "Tr-Esq", "Esquerda", "Fr-Esq"]
                print("  Distancia minima por setor:")
                for i, (label, dist) in enumerate(zip(labels, sectors)):
                    bar_len = min(int(dist * 3), 30)
                    bar = '█' * bar_len
                    dist_str = f"{dist:.2f}m" if dist < 100 else " ---"
                    print(f"    {label}: {dist_str:>7s}  {bar}")
                print()

                # Mapa ASCII
                print(f"  Mapa (range {max_range}m, ◉ = LiDAR):")
                print(render_ascii_scan(last_scan, 41, max_range))
                print()
                print(f"  [Ctrl+C para sair]  [Range: {max_range}m]")

                last_scan = None
                start_time = time.monotonic()

    except KeyboardInterrupt:
        print("\n\n  Parando...")


# ============================================================
# Modo 3: Gravar CSV
# ============================================================
def run_csv_logger(ser, duration=30):
    """Grava dados do LiDAR em CSV."""
    import csv

    parser = CoinD6Parser()
    filename = f"lidar_scan_{int(time.time())}.csv"

    print(f"\n  Gravando {duration}s de scans em {filename}...")
    print(f"  Ctrl+C para parar antes.\n")

    start = time.monotonic()
    all_points = []

    try:
        while (time.monotonic() - start) < duration:
            data = ser.read(512)
            if data:
                scans = parser.feed(data)
                for scan in scans:
                    for pt in scan:
                        pt['scan_id'] = parser.scan_count
                        pt['timestamp'] = time.monotonic() - start
                        all_points.append(pt)

            elapsed = time.monotonic() - start
            pct = elapsed / duration * 100
            sys.stdout.write(f"\r  [{int(pct):3d}%] Scans: {parser.scan_count}, Pontos: {len(all_points)}")
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n  Gravacao interrompida.")

    # Salvar
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['scan_id', 'timestamp_s', 'angle_deg', 'distance_m', 'intensity'])
        for pt in all_points:
            writer.writerow([
                pt['scan_id'],
                f"{pt['timestamp']:.4f}",
                f"{pt['angle']:.2f}",
                f"{pt['distance']:.4f}",
                pt['intensity'],
            ])

    print(f"\n\n  Salvo: {filename}")
    print(f"  {len(all_points)} pontos em {parser.scan_count} scans")
    print(f"  Abra no Excel ou plote com matplotlib para visualizar.")


# ============================================================
# Modo 4: Teste rapido de conectividade
# ============================================================
def run_quick_test(ser):
    """Teste rapido: mostra os primeiros pacotes recebidos."""
    print("\n  === TESTE RAPIDO DE CONECTIVIDADE ===\n")

    parser = CoinD6Parser()
    start = time.monotonic()
    test_duration = 5  # 5 segundos

    while (time.monotonic() - start) < test_duration:
        data = ser.read(256)
        if data:
            parser.feed(data)

        elapsed = time.monotonic() - start
        sys.stdout.write(
            f"\r  [{elapsed:.1f}s] Pacotes: {parser.raw_packets} | "
            f"Pontos: {parser.point_count} | "
            f"Scans: {parser.scan_count} | "
            f"Erros: {parser.parse_errors}"
        )
        sys.stdout.flush()

    print("\n")

    if parser.raw_packets > 0 and parser.parse_errors < parser.raw_packets * 0.5:
        print("  ✅ LIDAR FUNCIONANDO!")
        print(f"     {parser.raw_packets} pacotes recebidos e parseados.")
        print(f"     {parser.scan_count} scans completos (360°).")
        print(f"     {parser.point_count} pontos de distancia validos.")
        print(f"\n     Pronto para usar no carro autonomo!")
        return True
    elif parser.raw_packets > 0:
        print("  ⚠  LIDAR RESPONDENDO mas com muitos erros de parse.")
        print(f"     {parser.raw_packets} pacotes, {parser.parse_errors} erros.")
        print(f"     O formato do pacote pode ser diferente do esperado.")
        print(f"     Rode com --raw para ver os bytes e ajustar o parser.")
        return True
    else:
        print("  ❌ SEM DADOS DO LIDAR")
        print("     Verifique:")
        print("     1. O adaptador CH340 USB esta conectado ao Jetson?")
        print("     2. O LiDAR esta girando? (ele precisa de 5V pelo USB)")
        print("     3. Os fios estao corretos?")
        print("        Amarelo=5V, Branco=GND, Vermelho=TXD, Preto=RXD")
        print("     4. A porta serial esta correta? Tente: ls /dev/ttyUSB*")
        return False


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Teste do LiDAR COIN-D6 no Jetson Nano",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python3 coin_d6_test.py                     # Teste rapido + scan visual
  python3 coin_d6_test.py --raw               # Hex dump para debug
  python3 coin_d6_test.py --csv 60            # Gravar 60s em CSV
  python3 coin_d6_test.py --port /dev/ttyUSB1 # Porta especifica
  python3 coin_d6_test.py --range 3.0         # Mapa com range de 3m
        """
    )
    parser.add_argument("--port", type=str, default=None,
                        help="Porta serial (default: auto-detecta)")
    parser.add_argument("--raw", action="store_true",
                        help="Modo hex dump (debug)")
    parser.add_argument("--csv", type=int, default=None,
                        help="Gravar N segundos de dados em CSV")
    parser.add_argument("--range", type=float, default=6.0,
                        help="Range maximo do mapa em metros (default: 6)")
    parser.add_argument("--test", action="store_true",
                        help="Apenas teste rapido de conectividade (5s)")

    args = parser.parse_args()

    # Encontrar porta
    port = args.port
    if port is None:
        port = find_serial_port()
        if port is None:
            print("\n  [ERRO] Nenhuma porta serial encontrada!")
            print("  Verifique se o adaptador CH340 USB esta conectado.")
            print("  Dica: rode 'ls /dev/ttyUSB*' para verificar.")
            print("  Se nao aparecer, pode precisar instalar o driver:")
            print("    sudo apt-get install linux-modules-extra-$(uname -r)")
            sys.exit(1)
        print(f"  Auto-detectada porta: {port}")

    # Verificar permissoes
    if os.path.exists(port) and not os.access(port, os.R_OK | os.W_OK):
        print(f"\n  [ERRO] Sem permissao para acessar {port}")
        print(f"  Rode com sudo: sudo python3 {sys.argv[0]}")
        print(f"  Ou adicione seu usuario ao grupo dialout:")
        print(f"    sudo usermod -aG dialout $USER")
        print(f"    (precisa relogar depois)")
        sys.exit(1)

    # Conectar
    try:
        ser = connect_lidar(port)
    except serial.SerialException as e:
        print(f"\n  [ERRO] Nao conseguiu abrir {port}: {e}")
        sys.exit(1)

    try:
        if args.test:
            run_quick_test(ser)
        elif args.raw:
            run_hex_dump(ser, duration=10)
        elif args.csv:
            run_csv_logger(ser, duration=args.csv)
        else:
            # Primeiro faz um teste rapido
            print()
            ok = run_quick_test(ser)
            if ok:
                print("\n  Iniciando scan visual em 2s...")
                print("  (Ctrl+C para sair a qualquer momento)\n")
                time.sleep(2)
                run_scan_display(ser, max_range=args.range)
    finally:
        stop_lidar(ser)


if __name__ == "__main__":
    main()