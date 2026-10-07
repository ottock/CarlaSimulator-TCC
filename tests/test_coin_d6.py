"""Parser do LiDAR COIN-D6 (WitMotion), a copia canonica.

Devolve (angulo_graus, distancia_m) -- exatamente a assinatura que
ai.shared.lidar_pipeline.scan_to_sectors_m consome. Diferente das copias antigas
em hardware/, este parser NAO agrupa pontos em scans: quem fecha a volta e o
ScanAssembler, porque cortar a cada 300 pontos nao e uma volta completa.
"""
import struct

import pytest

from ai.car.coin_d6 import CoinD6Parser


def _packet_raw(start_deg, end_deg, raws):
    """Pacote COIN-D6 com os 16 bits CRUS de cada amostra, como saem do fio."""
    n = len(raws)
    pkt = bytearray(b"\xAA\x55")
    pkt += bytes([0x00, n])
    pkt += struct.pack("<H", int(round(start_deg * 100)))
    pkt += struct.pack("<H", int(round(end_deg * 100)))
    pkt += b"\x00\x00"
    for r in raws:
        pkt += bytes([0x30, r & 0xFF, (r >> 8) & 0xFF])
    return bytes(pkt)


def _packet(start_deg, end_deg, dists_mm):
    """Pacote com as distancias dadas em milimetros VERDADEIROS.

    No fio a distancia vai deslocada 2 bits (os 2 de baixo sao flags). A versao
    anterior deste ajudante escrevia os milimetros direto nos 16 bits -- isto e,
    codificava o MESMO erro do parser, e por isso os testes concordavam com ele
    enquanto o carro via tudo 4x mais longe.
    """
    return _packet_raw(start_deg, end_deg, [int(d) << 2 for d in dists_mm])


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
    pts = p.feed(_packet(0.0, 10.0, [15000, 3000]))    # 16 bits >> 2 vai ate 16,4 m
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


# ---------------------------------------------------------------------------
# Alcance MINIMO (medido nos logs de pista de 2026-09-12, em unidade CORRETA)
#
# O filtro de 0/1 mm nao bastava: 2 mm passava e virava "obstaculo colado".
# Nos 257.948 pontos das corridas de pista a separacao e perfeita:
#     0.000   - 0.00125 m :   3609 pontos (1.40%)   <- ruido
#     0.00125 - 0.050   m :      0 pontos           <- vazio absoluto
#     0.050 m ou mais     : 254339 pontos (98.6%)   <- medidas reais
# (Na escala errada antiga esses limites liam 0.005 e 0.200.) 50 mm e o alcance
# minimo do sensor, confirmado a mao: um objeto encostado para em 0,050 m.
# ---------------------------------------------------------------------------

def test_returns_below_the_minimum_range_are_noise():
    # 2 mm passava pelo filtro antigo e a rede lia parede encostada no carro
    p = CoinD6Parser()
    pts = p.feed(_packet(0.0, 10.0, [2, 3, 1500]))
    assert len(pts) == 1
    assert pts[0][1] == pytest.approx(1.5)


def test_real_short_readings_are_kept():
    # A pista tem 0.53 m de largura: com o carro no meio as paredes ficam a
    # ~0.26 m. Um filtro agressivo demais apagaria a propria pista.
    p = CoinD6Parser()
    pts = p.feed(_packet(0.0, 10.0, [50, 260, 300]))
    assert len(pts) == 3                         # inclusive os 50 mm do alcance minimo


def test_the_minimum_range_is_configurable():
    p = CoinD6Parser(min_range_m=0.30)
    pts = p.feed(_packet(0.0, 10.0, [250, 400]))
    assert len(pts) == 1
    assert pts[0][1] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# O formato da amostra (2026-10-07). Os 2 bits de baixo sao flags.
#
# Medido no carro, com o parser antigo (16 bits / 1000):
#   - objeto a ~13 cm do centro do sensor lia 0,53 m
#   - objeto encostado lia 0,200 m
#   - nas rodadas de pista, esquerda + direita somava 2,33 m num corredor de 0,56
# Os tres fecham com distancia = 16 bits >> 2, que e o formato de LiDAR de
# triangulacao com intensidade do protocolo YDLIDAR.
# ---------------------------------------------------------------------------

def test_the_low_two_bits_are_flags_not_distance():
    p = CoinD6Parser()
    pts = p.feed(_packet_raw(0.0, 10.0, [(1000 << 2) | 0b11, (1000 << 2)]))
    assert [d for _, d in pts] == [pytest.approx(1.0), pytest.approx(1.0)]


def test_what_the_operator_saw_at_13_cm_now_reads_13_cm():
    # O parser antigo devolvia 0,530 para estes bytes.
    p = CoinD6Parser()
    [(_, d)] = p.feed(_packet_raw(0.0, 0.0, [530]))
    assert d == pytest.approx(0.132)


def test_the_sensor_floor_is_five_centimetres():
    # Antigo "0,200": era o alcance minimo do sensor, e e uma medida valida.
    p = CoinD6Parser()
    [(_, d)] = p.feed(_packet_raw(0.0, 0.0, [200]))
    assert d == pytest.approx(0.050)


def test_the_track_wall_reads_at_half_the_lane():
    # Parede mais proxima num scan de pista: 1,115 no parser antigo. A meia
    # largura da pista e 0,28 m.
    p = CoinD6Parser()
    [(_, d)] = p.feed(_packet_raw(0.0, 0.0, [1115]))
    assert d == pytest.approx(0.278, abs=0.001)
