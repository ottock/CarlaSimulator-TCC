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


class CoinD6Parser:
    """Feed it raw serial bytes, get back polar points."""

    def __init__(self, max_range_m=MAX_RANGE_M):
        self.max_range_m = max_range_m
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
            dist_mm = pkt[offset + 1] | (pkt[offset + 2] << 8)
            if dist_mm <= 1:                     # 0/1 mm = sem leitura
                continue
            dist_m = dist_mm / 1000.0
            if dist_m > self.max_range_m:
                continue
            out.append(((start_angle + i * step) % 360.0, dist_m))
        return out
