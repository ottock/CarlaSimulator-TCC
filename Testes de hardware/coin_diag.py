#!/usr/bin/env python3
"""
COIN-D6 — Diagnostico de protocolo
====================================
Captura bytes brutos, identifica pacotes e mostra a estrutura.
Rode, espere 5s, e me mande a saida.
"""

import sys
import time
import glob
import serial

BAUD_RATE = 230400
CMD_START = bytes([0xAA, 0x55, 0xF0, 0x0F])

def find_port():
    for p in ['/dev/ttyUSB*', '/dev/ttyACM*']:
        ports = glob.glob(p)
        if ports:
            return sorted(ports)[0]
    return None

port = find_port()
if not port:
    print("ERRO: porta nao encontrada")
    sys.exit(1)

print(f"Porta: {port}")
ser = serial.Serial(port=port, baudrate=BAUD_RATE, timeout=0.5)
ser.reset_input_buffer()
time.sleep(0.1)
ser.write(CMD_START)
time.sleep(0.5)

# Capturar 2 segundos de dados brutos
print("Capturando 2s de dados...\n")
raw = bytearray()
t0 = time.monotonic()
while (time.monotonic() - t0) < 2.0:
    chunk = ser.read(1024)
    if chunk:
        raw.extend(chunk)

ser.write(bytes([0xAA, 0x55, 0xF5, 0x0A]))
ser.close()

print(f"Total capturado: {len(raw)} bytes")
print(f"Taxa: {len(raw)/2:.0f} bytes/s\n")

# Encontrar todos os headers AA 55
positions = []
for i in range(len(raw) - 1):
    if raw[i] == 0xAA and raw[i+1] == 0x55:
        positions.append(i)

print(f"Headers AA 55 encontrados: {len(positions)}\n")

if len(positions) < 2:
    print("Poucos headers. Primeiros 200 bytes brutos:")
    for i in range(0, min(200, len(raw)), 16):
        hex_part = ' '.join(f'{b:02X}' for b in raw[i:i+16])
        print(f"  {i:04d}: {hex_part}")
    sys.exit(0)

# Analisar distancia entre headers (= tamanho do pacote)
gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
unique_gaps = sorted(set(gaps))

print("=== TAMANHO DOS PACOTES ===")
print(f"Gaps entre headers: {unique_gaps}")
for g in unique_gaps:
    count = gaps.count(g)
    print(f"  {g} bytes: {count}x ({count/len(gaps)*100:.0f}%)")

# Mostrar os primeiros 10 pacotes completos
print(f"\n=== PRIMEIROS 10 PACOTES (hex) ===")
for i in range(min(10, len(positions)-1)):
    start = positions[i]
    end = positions[i+1]
    pkt = raw[start:end]
    hex_str = ' '.join(f'{b:02X}' for b in pkt[:40])
    if len(pkt) > 40:
        hex_str += f' ... ({len(pkt)} bytes total)'
    print(f"\nPacote {i} @ offset {start} ({len(pkt)} bytes):")
    print(f"  {hex_str}")
    
    # Tentar interpretar campos comuns
    if len(pkt) >= 6:
        print(f"  [2]={pkt[2]:02X} [3]={pkt[3]:02X} [4]={pkt[4]:02X} [5]={pkt[5]:02X}")
        
        # Testar se bytes 4-5 sao angulo (little-endian, 0.01 graus)
        angle_le = (pkt[5] << 8 | pkt[4]) * 0.01
        angle_be = (pkt[4] << 8 | pkt[5]) * 0.01
        print(f"  Bytes 4-5 como angulo: LE={angle_le:.2f}° BE={angle_be:.2f}°")

        # Se tem mais dados, mostrar possiveis distancias
        if len(pkt) >= 10:
            # Testar pares de bytes como distancia (mm)
            print(f"  Possiveis distancias (mm) a partir do byte 6:")
            dists = []
            for j in range(6, min(len(pkt)-1, 30), 2):
                d_le = pkt[j] | (pkt[j+1] << 8)
                dists.append(f"  [{j}-{j+1}] LE={d_le}")
            print("    " + " | ".join(dists[:8]))

# Mostrar um pacote completo byte a byte
if len(positions) >= 2:
    print(f"\n=== PACOTE COMPLETO BYTE A BYTE (pacote 0) ===")
    start = positions[0]
    end = positions[1]
    pkt = raw[start:end]
    for i in range(0, len(pkt), 16):
        hex_part = ' '.join(f'{b:02X}' for b in pkt[i:i+16])
        ascii_part = ''.join(chr(b) if 32 <= b < 127 else '.' for b in pkt[i:i+16])
        print(f"  {i:3d}: {hex_part:<50s} {ascii_part}")

print(f"\n=== COPIE TODA ESSA SAIDA E ME ENVIE ===")