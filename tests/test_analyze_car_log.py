"""Analise do log do carro, no PC (Fase 6b).

E isto que transforma o teste com rodas no ar nas medicoes de bancada #1 (arco de
oclusao da carroceria) e #2 (zero e sentido do angulo) -- com numero.
"""
import numpy as np
import pytest

from ai.car.run_log import RunLogger
from scripts_analyze import fps_stats, load_run, occlusion_arc, sector_to_deg


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
