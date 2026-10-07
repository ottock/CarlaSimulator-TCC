"""COIN-D6 (WitMotion) 2D LiDAR packet parser -- the canonical copy.

Emits ``(angle_deg, dist_m)`` tuples, which is exactly what
``ai.shared.lidar_pipeline.scan_to_sectors_m`` consumes. Python 3.6-safe.

Two divergent copies of this parser already exist under ``hardware/`` (one of them
returns 3-tuples with intensity). This is the single source; the hardware scripts
should migrate to it rather than a third copy being made.

Deliberate difference from both: this parser does NOT group points into scans. The
old copies closed a "scan" every 300 points, which is not a full revolution -- a
sector never swept would read ``max_range`` = "free", a hole in the wall. Grouping
by revolution belongs to :mod:`ai.car.scan_assembly`.
"""
import struct

MAX_RANGE_M = 12.0          # alcance fisico do sensor
MAX_SAMPLES_PER_PACKET = 50
HEADER = b"\xAA\x55"


# Alcance MINIMO do sensor, ja na unidade CORRETA (ver _points_from_packet).
# Nas corridas de 2026-09-12, 1.4% dos pontos ficam abaixo de 1,25 mm (lixo: o
# sensor usa ~0 para "sem leitura") e nenhum cai entre 1,25 e 50 mm -- 50 mm e o
# alcance minimo fisico do COIN-D6 (confirmado a mao em 2026-10-07: um objeto
# empurrado ate encostar para em 0,050 m). 20 mm fica no meio dessa faixa vazia.
#
# HISTORICO: este valor era 0.05 e o comentario falava em "0.20 m" porque o
# parser lia a distancia 4x maior. Nessa escala errada 0.05 equivalia a 12,5 mm.
MIN_RANGE_M = 0.02


class CoinD6Parser:
    """Feed it raw serial bytes, get back polar points."""

    def __init__(self, max_range_m=MAX_RANGE_M, min_range_m=MIN_RANGE_M,
                 dist_offset_m=0.0):
        """
        Args:
            dist_offset_m: subtraido de toda distancia DEPOIS de decodificar.
                Padrao 0: aqui fica so o protocolo. O valor medido deste sensor
                e calibracao, e mora com as outras em hardware/jetson_runtime.py
                (LIDAR_DIST_OFFSET_M).
        """
        self.max_range_m = max_range_m
        self.min_range_m = min_range_m
        self.dist_offset_m = float(dist_offset_m)
        self.buffer = bytearray()
        self.parse_errors = 0

    def feed(self, data):
        """Consume bytes and return the ``(angle_deg, dist_m)`` points completed."""
        self.buffer.extend(data)
        points = []
        while True:
            idx = self.buffer.find(HEADER)
            if idx < 0 or idx + 4 > len(self.buffer):
                if idx < 0:
                    # Guarda o ultimo byte: o header pode estar partido entre reads.
                    self.buffer = self.buffer[-1:]
                break
            if idx > 0:
                self.buffer = self.buffer[idx:]

            sample_count = self.buffer[3]
            if sample_count == 0 or sample_count > MAX_SAMPLES_PER_PACKET:
                self.buffer = self.buffer[2:]   # descarta este header e recomeca
                self.parse_errors += 1
                continue

            pkt_len = 10 + sample_count * 3
            if len(self.buffer) < pkt_len:
                break                            # pacote incompleto: espera mais
            pkt = bytes(self.buffer[:pkt_len])
            self.buffer = self.buffer[pkt_len:]
            points.extend(self._points_from_packet(pkt, sample_count))
        return points

    def _points_from_packet(self, pkt, sample_count):
        start_angle = (struct.unpack_from("<H", pkt, 4)[0] * 0.01) % 360.0
        end_angle = (struct.unpack_from("<H", pkt, 6)[0] * 0.01) % 360.0

        angle_diff = end_angle - start_angle
        if angle_diff < 0:
            angle_diff += 360.0
        if angle_diff > 180:                     # pacote cruzando o zero
            angle_diff -= 360.0
        step = angle_diff / max(sample_count - 1, 1)

        out = []
        for i in range(sample_count):
            offset = 10 + i * 3
            if offset + 2 >= len(pkt):
                break
            # Os 16 bits NAO sao milimetros: os 2 bits de baixo sao flags e a
            # distancia e o resto, `>> 2`. E o formato de LiDAR de triangulacao
            # com intensidade do protocolo YDLIDAR, que este pacote segue byte a
            # byte (AA 55, CT, LSN, FSA, LSA, CS, amostras de 3 bytes):
            #     Distance(i) = uint16_t(S(2) << 8 | S(1)) >> 2   [mm]
            # Lido sem o shift, TUDO saia 4x mais longe. Confirmado de tres
            # jeitos em 2026-10-07: (1) objeto a ~13 cm do centro do sensor lia
            # 0,53; (2) objeto encostado lia 0,200, que e o alcance minimo de
            # 5 cm; (3) nas 9 rodadas de pista, esquerda+direita somava 2,33 m
            # num corredor de 0,56 m -- /4 da 0,583.
            # Consequencia: em toda corrida ate aqui as paredes da pista (~28 cm)
            # chegavam ao modelo a ~1,1 m, alem do max_range de 1,0 m, ou seja
            # como espaco LIVRE. O braco de LiDAR nunca viu a pista.
            raw = pkt[offset + 1] | (pkt[offset + 2] << 8)
            dist_mm = raw >> 2
            # O desconto vem ANTES dos filtros: o "sem leitura" (~0) fica
            # negativo e cai no min_range, como deve.
            dist_m = dist_mm / 1000.0 - self.dist_offset_m
            # Abaixo do alcance minimo NAO e um obstaculo colado no carro, e
            # ruido: uma parede a 0 m e fisicamente impossivel, e normalizada
            # vira 0.0 = "encostado", a leitura mais perigosa possivel.
            if dist_m < self.min_range_m:
                continue
            if dist_m > self.max_range_m:
                continue
            out.append(((start_angle + i * step) % 360.0, dist_m))
        return out
