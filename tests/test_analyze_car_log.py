"""Analise do log do carro, no PC (Fase 6b).

E isto que transforma o teste com rodas no ar nas medicoes de bancada #1 (arco de
oclusao da carroceria) e #2 (zero e sentido do angulo) -- com numero.
"""
import numpy as np
import pytest

from ai.car.run_log import RunLogger
from scripts_analyze import (blocked_runs, fps_stats, load_run, occlusion_arc,
                             sector_to_deg)


def test_occlusion_arc_finds_sectors_that_are_always_blocked():
    # A carroceria e o unico obstaculo que nunca se move: distancia curta E
    # praticamente constante. O mundo la fora varia conforme o carro gira.
    rng = np.random.default_rng(0)
    sectors = rng.uniform(0.5, 1.0, (200, 72)).astype(np.float32)
    sectors[:, 20:50] = 0.12                     # carroceria: curta e constante
    arc = occlusion_arc(sectors, max_range=1.0)
    assert set(arc) == set(range(20, 50))


def test_occlusion_arc_ignores_a_close_but_moving_wall():
    # uma parede perto varia conforme o carro se move: nao e a carroceria
    rng = np.random.default_rng(1)
    sectors = rng.uniform(0.5, 1.0, (200, 72)).astype(np.float32)
    sectors[:, 10:15] = rng.uniform(0.10, 0.40, (200, 5))
    assert occlusion_arc(sectors, max_range=1.0) == []


def test_occlusion_arc_empty_when_nothing_is_blocked():
    sectors = np.ones((50, 72), dtype=np.float32)
    assert occlusion_arc(sectors, max_range=1.0) == []


def test_sector_to_degrees_uses_the_sector_centre():
    # setor i cobre [i*5, (i+1)*5); o centro e (i+0.5)*5 -- a mesma convencao do
    # apply_fov_mask, senao o angulo relatado nao casaria com a mascara
    assert sector_to_deg(0, 72) == pytest.approx(2.5)
    assert sector_to_deg(36, 72) == pytest.approx(182.5)


def test_fps_stats_reports_the_median_and_the_worst_frame():
    frames = [{"dt": 0.05}, {"dt": 0.05}, {"dt": 0.05}, {"dt": 0.40}]
    st = fps_stats(frames)
    assert st["fps_median"] == pytest.approx(20.0)
    assert st["dt_max"] == pytest.approx(0.40)
    assert st["n"] == 4


def test_fps_stats_ignores_the_first_frame_with_zero_dt():
    # o primeiro step nao tem frame anterior, entao dt=0 e nao e um FPS infinito
    st = fps_stats([{"dt": 0.0}, {"dt": 0.05}, {"dt": 0.05}])
    assert st["fps_median"] == pytest.approx(20.0)


def test_load_run_reads_back_what_the_logger_wrote(tmp_path):
    lg = RunLogger(str(tmp_path / "r"), meta={"fov_deg": 180.0}, jpeg_every=100)
    lg.log_scan(t=0.0, points=[(0.0, 0.5)])
    for i in range(3):
        lg.log_frame(t=float(i), sectors=np.full(72, 0.5, dtype=np.float32),
                     control=(0.2, 0.0, 0.0), servo_us=1540, dt=0.05)
    lg.close()
    run = load_run(str(tmp_path / "r"))
    assert run["meta"]["fov_deg"] == 180.0
    assert run["sectors"].shape == (3, 72)
    assert len(run["frames"]) == 3
    assert len(run["scans"]) == 1


# --- agrupamento do arco em trechos contiguos (o relatorio de min/max mentia) ---

def test_blocked_runs_groups_a_simple_contiguous_arc():
    assert blocked_runs([20, 21, 22, 23], 72) == [[20, 21, 22, 23]]


def test_blocked_runs_joins_the_arc_that_crosses_zero():
    # A carroceria pode ocluir em volta da FRENTE. Reportar min/max daria
    # "2.5 a 357.5 graus", que se le como o circulo inteiro bloqueado, quando
    # sao 6 setores = 30 graus.
    assert blocked_runs([0, 1, 2, 69, 70, 71], 72) == [[69, 70, 71, 0, 1, 2]]


def test_blocked_runs_keeps_disjoint_arcs_separate():
    # um setor solto + um bloco: sao dois trechos, nao um arco de 5 a 41
    assert blocked_runs([5, 30, 31, 32], 72) == [[5], [30, 31, 32]]


def test_blocked_runs_empty_when_nothing_is_blocked():
    assert blocked_runs([], 72) == []


def test_blocked_runs_handles_the_whole_circle_without_duplicating():
    # borda: tudo bloqueado comeca em 0 E termina em 71, mas nao deve juntar
    # consigo mesmo e repetir setores
    runs = blocked_runs(list(range(72)), 72)
    assert len(runs) == 1
    assert sorted(runs[0]) == list(range(72))
