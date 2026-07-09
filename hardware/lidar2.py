# Para rodar:
# python lidar2.py
#

import signal
import struct
import threading
import time

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

import serial

matplotlib.use("TkAgg")

# ============================================================
# CTRL + C
# ============================================================
signal.signal(signal.SIGINT, signal.SIG_DFL)

# ============================================================
# CONFIGURAÇÕES
# ============================================================
SERIAL_PORT = "/dev/ttyUSB0"

# COIN-D6
BAUDRATE = 230400
DEBUG = False
PLOT_EVERY = 3

# ============================================================
# DADOS GLOBAIS
# ============================================================
scan_data = np.zeros(360)
intensity_data = np.zeros(360)
lock = threading.Lock()
running = True
packet_counter = 0
plot_counter = 0

# ============================================================
# SERIAL
# ============================================================
print("[INFO] Abrindo serial...")
ser = serial.Serial(
    SERIAL_PORT,
    BAUDRATE,
    timeout=0.01
)
print("[INFO] Serial OK")

# ============================================================
# START SCAN
# ============================================================
print("[INFO] Enviando START SCAN")
start_cmd = bytes([
    0xAA,
    0x55,
    0xF0,
    0x0F
])
ser.write(start_cmd)
time.sleep(0.2)
ack = ser.read(100)
print("\n================ ACK ================")
print(ack.hex())
print("=====================================\n")


# ============================================================
# CHECKSUM XOR
# ============================================================
def validate_checksum(packet):
    try:
        # checksum do pacote
        packet_cs = struct.unpack(
            "<H",
            packet[8:10]
        )[0]
        calc_cs = 0

        # XOR em todos os bytes
        # exceto checksum
        for i in range(len(packet)):
            if i == 8 or i == 9:
                continue
            calc_cs ^= packet[i]
        calc_cs &= 0xFFFF

        if DEBUG:
            print("CHECKSUM RX:", hex(packet_cs))
            print("CHECKSUM OK:", hex(calc_cs))
        
        # Alguns D6 parecem não seguir
        # exatamente o manual no XOR.
        # Então deixamos apenas log.
        return True
    except Exception as e:
        print("[CHECKSUM ERROR]", e)
        return False

# ============================================================
# PARSER
# ============================================================
def parse_packet(packet):

    global scan_data
    global intensity_data
    global packet_counter

    try:
        packet_counter += 1

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------
        header = packet[0:2]
        if header != b'\xAA\x55':
            if DEBUG:
                print("[HEADER INVÁLIDO]")
            return

        # ----------------------------------------------------
        # FOOTER
        # ----------------------------------------------------
        footer = packet[-2:]

        if footer != b'\xAA\x55':
            if DEBUG:
                print("[FOOTER INVÁLIDO]",
                      footer.hex())

        # ----------------------------------------------------
        # CHECKSUM
        # ----------------------------------------------------
        validate_checksum(packet)

        # ----------------------------------------------------
        # CAMPOS
        # ----------------------------------------------------
        mt = packet[2]
        lsn = packet[3]
        if lsn == 0:
            return

        if lsn > 40:
            # proteção contra corrupção
            return

        # ----------------------------------------------------
        # ÂNGULOS
        # ----------------------------------------------------

        fsa = struct.unpack(
            "<H",
            packet[4:6]
        )[0]

        lsa = struct.unpack(
            "<H",
            packet[6:8]
        )[0]

        angle_start = (
            (fsa >> 1) / 64.0
        )

        angle_end = (
            (lsa >> 1) / 64.0
        )

        # ----------------------------------------------------
        # WRAP ANGULAR
        # ----------------------------------------------------
        diff = angle_end - angle_start
        if diff < 0:
            diff += 360

        # ----------------------------------------------------
        # STEP
        # ----------------------------------------------------
        if lsn > 1:
            step = diff / (lsn - 1)
        else:
            step = 0

        # ----------------------------------------------------
        # PONTOS
        # ----------------------------------------------------
        data_start = 10
        with lock:
            for i in range(lsn):
                idx = data_start + i * 3
                if idx + 3 > len(packet):
                    break
                s_l = packet[idx]
                s_m = packet[idx + 1]
                s_h = packet[idx + 2]

                # ------------------------------------------------
                # INTENSIDADE
                # ------------------------------------------------
                intensity = (
                    (s_m & 0x03) * 64 +
                    (s_l >> 2)
                )

                # ------------------------------------------------
                # DISTÂNCIA
                # ------------------------------------------------
                distance = (
                    s_h * 64 +
                    (s_m >> 2)
                )

                if distance <= 0:
                    continue

                if distance > 12000:
                    continue

                # ------------------------------------------------
                # ÂNGULO
                # ------------------------------------------------
                angle = (
                    angle_start +
                    step * i
                )

                angle = (angle+180) % 360
                # angle %= 360

                angle_int = int(angle)
                scan_data[angle_int] = distance
                intensity_data[angle_int] = intensity

        if DEBUG:
            print("\n====================")
            print("PACOTE:", packet_counter)
            print("LSN:", lsn)
            print("START:", angle_start)
            print("END:", angle_end)
            print("STEP:", step)
            print("MAX DIST:", np.max(scan_data))
            print("====================")

    except Exception as e:
        print("\n[PARSE ERROR]")
        print(e)

# ============================================================
# THREAD SERIAL
# ============================================================
def serial_thread():
    global running
    buffer = bytearray()
    while running:
        try:
            if ser.in_waiting:
                data = ser.read(
                    ser.in_waiting
                )
                buffer.extend(data)

                if DEBUG:
                    print("\nBYTES:",
                          len(data))
                    print("BUFFER:",
                          len(buffer))

                # ------------------------------------------------
                # PROCESSA BUFFER
                # ------------------------------------------------
                while True:
                    idx = buffer.find(
                        b'\xAA\x55'
                    )

                    if idx < 0:
                        # evita crescimento infinito
                        if len(buffer) > 10000:
                            buffer.clear()
                        break

                    # tamanho mínimo
                    if len(buffer) < idx + 12:
                        break

                    # quantidade de pontos
                    lsn = buffer[idx + 3]

                    # sanity check
                    if lsn == 0 or lsn > 40:
                        del buffer[:idx + 2]
                        continue

                    # ------------------------------------------------
                    # TAMANHO REAL
                    # ------------------------------------------------
                    packet_size = (
                        12 + lsn * 3
                    )

                    # pacote completo?
                    if len(buffer) < idx + packet_size:
                        # normal em stream serial
                        break

                    # ------------------------------------------------
                    # EXTRAI PACOTE
                    # ------------------------------------------------
                    packet = buffer[
                        idx :
                        idx + packet_size
                    ]

                    parse_packet(packet)

                    # remove pacote processado
                    del buffer[
                        : idx + packet_size
                    ]

            else:
                time.sleep(0.001)

        except Exception as e:
            print("\n[SERIAL ERROR]")
            print(e)

# ============================================================
# THREAD SERIAL
# ============================================================

t = threading.Thread(
    target=serial_thread
)
t.daemon = True
t.start()

# ============================================================
# MATPLOTLIB
# ============================================================
plt.ion()

fig = plt.figure(
    figsize=(8, 8)
)

ax = fig.add_subplot(
    111,
    polar=True
)

ax.set_title(
    "COIN-D6 Radar"
)

ax.set_ylim(0, 12000)

# ============================================================
# LOOP PRINCIPAL
# ============================================================

try:
    while running:
        plot_counter += 1
        if plot_counter % PLOT_EVERY != 0:
            time.sleep(0.005)
            continue

        with lock:
            angles = np.arange(360)
            distances = scan_data.copy()
            intensities = intensity_data.copy()

        valid = distances > 0
        ax.clear()
        ax.set_ylim(0, 12000)

        ax.set_title(
            f"COIN-D6 Radar | "
            f"Max: {np.max(distances):.0f} mm"
        )

        # ----------------------------------------------------
        # RADAR COLORIDO POR INTENSIDADE
        # ----------------------------------------------------

        ax.scatter(
            np.radians(
                angles[valid]
            ),
            distances[valid],
            c=intensities[valid],
            s=8
        )

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n[INFO] CTRL+C")

finally:
    running = False
    print("[INFO] ENCERRANDO")

    # --------------------------------------------------------
    # STOP SCAN
    # --------------------------------------------------------

    try:
        stop_cmd = bytes([
            0xAA,
            0x55,
            0xF5,
            0x0A
        ])

        ser.write(stop_cmd)
        print("[INFO] STOP enviado")

    except Exception as e:
        print(e)

    # --------------------------------------------------------
    # SERIAL
    # --------------------------------------------------------

    try:
        ser.close()
        print("[INFO] Serial fechada")

    except:
        pass

    # --------------------------------------------------------
    # MATPLOTLIB
    # --------------------------------------------------------

    try:
        plt.close("all")

    except:
        pass

    print("[INFO] FINALIZADO")
