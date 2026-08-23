"""Log da corrida do carro (Fase 6b).

O log E o entregavel deste teste: e dele que saem, no PC, o arco de oclusao da
carroceria e o sentido do angulo do LiDAR. Formatos deliberadamente banais (jsonl
e npy) para nao precisar de dependencia nenhuma para ler.
"""
import io
import json
import os

import numpy as np
import pytest

from ai.car.run_log import RunLogger


def _read_jsonl(path):
    with io.open(path, "r", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def test_creates_the_run_directory_and_meta(tmp_path):
    lg = RunLogger(str(tmp_path / "run1"), meta={"fov_deg": 180.0, "esc_armado": False})
    lg.close()
    with io.open(str(tmp_path / "run1" / "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["fov_deg"] == 180.0
    assert meta["esc_armado"] is False


def test_one_jsonl_line_per_frame(tmp_path):
    lg = RunLogger(str(tmp_path / "r"), meta={})
    for i in range(3):
        lg.log_frame(t=float(i), sectors=np.ones(72, dtype=np.float32),
                     control=(0.1, 0.5, 0.0), servo_us=1520, dt=0.05)
    lg.close()
    rows = _read_jsonl(str(tmp_path / "r" / "frames.jsonl"))
    assert len(rows) == 3
    assert rows[0]["steer"] == 0.1
    assert rows[0]["servo_us"] == 1520
    assert rows[0]["dt"] == 0.05


def test_sectors_are_saved_as_a_matrix(tmp_path):
    lg = RunLogger(str(tmp_path / "r"), meta={})
    for i in range(4):
        lg.log_frame(t=float(i), sectors=np.full(72, i, dtype=np.float32),
                     control=(0.0, 0.0, 0.0), servo_us=1500, dt=0.05)
    lg.close()
    arr = np.load(str(tmp_path / "r" / "sectors.npy"))
    assert arr.shape == (4, 72)
    assert arr[2][0] == 2.0


def test_raw_scans_are_logged_for_the_bench_measurements(tmp_path):
    # e daqui que saem o arco de oclusao e o zero/sentido do angulo
    lg = RunLogger(str(tmp_path / "r"), meta={})
    lg.log_scan(t=1.0, points=[(0.0, 1.5), (180.0, 0.2)])
    lg.close()
    rows = _read_jsonl(str(tmp_path / "r" / "scans.jsonl"))
    assert len(rows) == 1
    assert rows[0]["points"][1] == [180.0, 0.2]


def test_saves_one_jpeg_every_n_frames(tmp_path):
    lg = RunLogger(str(tmp_path / "r"), meta={}, jpeg_every=2)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    for i in range(5):
        lg.log_frame(t=float(i), sectors=np.ones(72, dtype=np.float32),
                     control=(0.0, 0.0, 0.0), servo_us=1500, dt=0.05, frame_bgr=frame)
    lg.close()
    jpegs = sorted(os.listdir(str(tmp_path / "r" / "frames")))
    assert len(jpegs) == 3          # frames 0, 2 e 4


def test_close_is_safe_to_call_twice(tmp_path):
    # o runtime fecha o log no finally; uma excecao pode fazer isso rodar duas vezes
    lg = RunLogger(str(tmp_path / "r"), meta={})
    lg.log_frame(t=0.0, sectors=np.ones(72, dtype=np.float32),
                 control=(0.0, 0.0, 0.0), servo_us=1500, dt=0.05)
    lg.close()
    lg.close()
    assert np.load(str(tmp_path / "r" / "sectors.npy")).shape == (1, 72)


def test_empty_run_still_writes_a_readable_sectors_file(tmp_path):
    lg = RunLogger(str(tmp_path / "r"), meta={})
    lg.close()
    assert np.load(str(tmp_path / "r" / "sectors.npy")).shape == (0, 72)


def test_refuses_to_overwrite_an_existing_run(tmp_path):
    # Isto roda varias vezes seguidas no Jetson. Sobrescrever um log ja gravado
    # perderia o entregavel do teste; e falhar com um OSError cru, depois de a
    # camera e o LiDAR ja estarem abertos, nao diz ao operador o que fazer.
    d = tmp_path / "r"
    d.mkdir()
    with pytest.raises(IOError) as exc:
        RunLogger(str(d), meta={})
    assert str(d) in str(exc.value)
