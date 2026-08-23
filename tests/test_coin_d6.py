"""Parser do LiDAR COIN-D6 (WitMotion), a copia canonica.

Devolve (angulo_graus, distancia_m) -- exatamente a assinatura que
ai.shared.lidar_pipeline.scan_to_sectors_m consome. Diferente das copias antigas
em hardware/, este parser NAO agrupa pontos em scans: quem fecha a volta e o
ScanAssembler, porque cortar a cada 300 pontos nao e uma volta completa.
"""
import struct

import pytest

from ai.car.coin_d6 import CoinD6Parser


def _packet(start_deg, end_deg, dists_mm):
    """Monta um pacote COIN-D6 valido com as distancias dadas (em milimetros)."""
    n = len(dists_mm)
    pkt = bytearray(b"\xAA\x55")
    pkt += bytes([0x00, n])
    pkt += struct.pack("<H", int(round(start_deg * 100)))
    pkt += struct.pack("<H", int(round(end_deg * 100)))
    pkt += b"\x00\x00"
    for d in dists_mm:
        pkt += bytes([0x30, d & 0xFF, (d >> 8) & 0xFF])
    return bytes(pkt)


def test_single_packet_yields_angle_and_distance_in_metres():
    p = CoinD6Parser()
    pts = p.feed(_packet(10.0, 20.0, [1000, 2000]))
    assert len(pts) == 2
    assert pts[0][0] == pytest.approx(10.0)
    assert pts[0][1] == pytest.approx(1.0)      # 1000 mm -> 1.0 m
    assert pts[1][0] == pytest.approx(20.0)
    assert pts[1][1] == pytest.approx(2.0)


def test_zero_and_one_millimetre_mean_no_return():
    # o sensor usa 0/1 mm para "sem leitura"; deixar passar viraria um obstaculo
    # colado no carro, que e a leitura mais perigosa possivel
    p = CoinD6Parser()
    pts = p.feed(_packet(0.0, 10.0, [0, 1, 1500]))
    assert len(pts) == 1
    assert pts[0][1] == pytest.approx(1.5)


def test_distance_beyond_max_range_is_dropped():
    p = CoinD6Parser(max_range_m=12.0)
    pts = p.feed(_packet(0.0, 10.0, [20000, 3000]))
    assert len(pts) == 1
    assert pts[0][1] == pytest.approx(3.0)


def test_packet_split_across_two_reads_is_buffered():
    # a serial entrega pedacos arbitrarios; um pacote partido nao pode virar lixo
    p = CoinD6Parser()
    pkt = _packet(30.0, 40.0, [1000, 1100])
    assert p.feed(pkt[:7]) == []
    pts = p.feed(pkt[7:])
    assert len(pts) == 2


def test_garbage_before_the_header_is_skipped():
    p = CoinD6Parser()
    pts = p.feed(b"\x01\x02\x03" + _packet(50.0, 60.0, [1200]))
    assert len(pts) == 1
    assert pts[0][0] == pytest.approx(50.0)


def test_two_packets_in_one_read():
    p = CoinD6Parser()
    pts = p.feed(_packet(0.0, 10.0, [1000]) + _packet(20.0, 30.0, [2000]))
    assert [round(a) for a, _ in pts] == [0, 20]


def test_angles_wrap_into_zero_360():
    p = CoinD6Parser()
    pts = p.feed(_packet(350.0, 359.0, [1000, 1000]))
    for ang, _ in pts:
        assert 0.0 <= ang < 360.0


def test_absurd_sample_count_is_rejected_without_hanging():
    p = CoinD6Parser()
    bad = bytearray(b"\xAA\x55\x00\xFF")        # count 255 > limite
    bad += b"\x00" * 6
    pts = p.feed(bytes(bad) + _packet(10.0, 20.0, [1000]))
    assert len(pts) == 1                        # recupera e acha o pacote bom
    assert p.parse_errors >= 1
