"""Agrupa os pontos do LiDAR em VOLTAS completas (Fase 6b).

O parser entrega pontos continuamente. Se a gente fechasse um "scan" por contagem
de pontos (como as versoes antigas em hardware/ faziam, a cada 300), um setor nunca
varrido entraria no vetor como max_range = "livre" -- um buraco na parede, que e
justamente o sinal que faz o carro bater nela. Aqui a volta fecha no WRAP do
angulo (359 -> 0).
"""
from ai.car.scan_assembly import ScanAssembler


def _sweep(start, stop, step=1.0, dist=2.0):
    """Pontos de `start` a `stop` (exclusivo) de `step` em `step` graus."""
    pts = []
    a = start
    while a < stop:
        pts.append((a, dist))
        a += step
    return pts


def test_no_wrap_emits_nothing():
    asm = ScanAssembler()
    assert asm.feed(_sweep(0.0, 90.0)) == []


def test_first_partial_revolution_is_discarded():
    # Comecamos a ouvir a serial no meio de uma volta: esse pedaco NAO e uma volta
    # completa e emiti-lo entregaria ao modelo um vetor cheio de buracos.
    asm = ScanAssembler(min_points=10)
    scans = asm.feed(_sweep(300.0, 360.0) + _sweep(0.0, 60.0))
    assert scans == []


def test_full_revolution_after_the_first_wrap_is_emitted():
    asm = ScanAssembler(min_points=10)
    asm.feed(_sweep(300.0, 360.0))          # fragmento inicial, descartado
    scans = asm.feed(_sweep(0.0, 360.0) + [(1.0, 2.0)])   # volta inteira + wrap
    assert len(scans) == 1
    assert len(scans[0]) == 360


def test_two_revolutions_yield_two_scans():
    asm = ScanAssembler(min_points=10)
    asm.feed(_sweep(350.0, 360.0))          # fragmento inicial
    scans = asm.feed(_sweep(0.0, 360.0) + _sweep(0.0, 360.0) + [(0.5, 2.0)])
    assert len(scans) == 2
    assert all(len(s) == 360 for s in scans)


def test_points_accumulate_across_feed_calls():
    # a serial entrega pedaços; uma volta quase sempre chega em varios reads
    asm = ScanAssembler(min_points=10)
    asm.feed(_sweep(350.0, 360.0))
    asm.feed(_sweep(0.0, 180.0))
    scans = asm.feed(_sweep(180.0, 360.0) + [(2.0, 2.0)])
    assert len(scans) == 1
    assert len(scans[0]) == 360


def test_revolution_with_too_few_points_is_dropped():
    # LiDAR engasgando: uma "volta" com 3 pontos nao descreve o mundo
    asm = ScanAssembler(min_points=50)
    asm.feed(_sweep(350.0, 360.0))
    scans = asm.feed([(10.0, 2.0), (200.0, 2.0), (350.0, 2.0), (1.0, 2.0)])
    assert scans == []


def test_scan_keeps_the_angles_and_distances_intact():
    asm = ScanAssembler(min_points=2)
    asm.feed([(350.0, 1.0), (5.0, 9.9)])            # wrap inicial, descarta
    # o (355, 7.7) antes do (1.0) e necessario: so uma QUEDA maior que 180 graus
    # conta como wrap, entao 20 -> 1 nao fecharia a volta
    scans = asm.feed([(10.0, 3.3), (20.0, 4.4), (355.0, 7.7), (1.0, 5.5)])
    assert scans[0][0] == (5.0, 9.9)
    assert (10.0, 3.3) in scans[0]
    assert (355.0, 7.7) in scans[0]
    assert (1.0, 5.5) not in scans[0]               # ja pertence a volta seguinte
