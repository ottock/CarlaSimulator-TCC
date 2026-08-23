"""Le o log de uma corrida do carro e devolve as medicoes de bancada (Fase 6b).

Roda no PC, sobre o diretorio que o ai.car.run_log gravou no Jetson. E isto que
transforma o teste com rodas no ar em numero:

  1. arco de oclusao da carroceria  (medicao de bancada #1)
  2. onde estao os retornos mais proximos (ajuda a medicao #2, zero/sentido)
  3. FPS real do laco
  4. ocupacao dos setores, para comparar com o dataset do simulador

Uso:
    python scripts/analyze_car_log.py runs/car_teste1
"""
import argparse
import io
import json
import os

import numpy as np


def load_run(run_dir):
    """Read a run directory back into memory."""
    with io.open(os.path.join(run_dir, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    sectors = np.load(os.path.join(run_dir, "sectors.npy"))
    frames = _read_jsonl(os.path.join(run_dir, "frames.jsonl"))
    scans = _read_jsonl(os.path.join(run_dir, "scans.jsonl"))
    return {"meta": meta, "sectors": sectors, "frames": frames, "scans": scans}


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    with io.open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sector_to_deg(i, n_sectors=72):
    """Centre angle of sector ``i`` -- the same convention ``apply_fov_mask`` uses."""
    return (i + 0.5) * (360.0 / n_sectors)


def occlusion_arc(sectors, max_range, near_frac=0.5, blocked_frac=0.95, spread_frac=0.05):
    """Sectors blocked by the car's own chassis.

    The chassis is the only obstacle that never moves: it reads SHORT in nearly
    every frame AND barely varies. Everything out in the world changes as the car
    turns, so a close-but-varying wall is correctly not reported.

    Args:
        sectors: ``(N, n_sectors)`` of normalized values, as fed to the model.
        max_range: the run's ``max_range``. Kept for call-site symmetry with the
            rest of the analysis; the thresholds below act on the already
            normalized [0, 1] values, so it is not read here.
        near_frac: a reading below this fraction of full range counts as "something
            there".
        blocked_frac: fraction of frames that must be near for the sector to count.
        spread_frac: max (p95 - p5) spread, as a fraction of full range, for the
            sector to count as "not moving".
    """
    if sectors.size == 0:
        return []
    arr = np.asarray(sectors, dtype=np.float64)
    near = (arr < near_frac).mean(axis=0)
    spread = np.percentile(arr, 95, axis=0) - np.percentile(arr, 5, axis=0)
    blocked = (near >= blocked_frac) & (spread <= spread_frac)
    return [int(i) for i in np.nonzero(blocked)[0]]


def fps_stats(frames):
    """Loop timing. The sim ran at 20 Hz; well under that and the car reacts late."""
    dts = [f["dt"] for f in frames if f.get("dt", 0.0) > 0.0]
    if not dts:
        return {"n": len(frames), "fps_median": 0.0, "fps_min": 0.0, "dt_max": 0.0}
    dts = np.asarray(dts, dtype=np.float64)
    return {"n": len(frames), "fps_median": float(1.0 / np.median(dts)),
            "fps_min": float(1.0 / dts.max()), "dt_max": float(dts.max())}


def main():
    p = argparse.ArgumentParser(description="Analisa o log de uma corrida do carro")
    p.add_argument("run_dir")
    a = p.parse_args()

    run = load_run(a.run_dir)
    sectors, meta = run["sectors"], run["meta"]
    n_sectors = int(meta.get("n_sectors", 72))
    max_range = float(meta.get("max_range_m_car", 1.0))
    print("corrida: %s  (%d frames, %d voltas de LiDAR)"
          % (a.run_dir, len(run["frames"]), len(run["scans"])))
    print("meta: fov=%s deg  max_range no carro=%.3f m  crop_frac=%s"
          % (meta.get("fov_deg"), max_range, meta.get("crop_frac")))

    st = fps_stats(run["frames"])
    print("\n--- FPS (o sim rodava a 20 Hz) ---")
    print("mediana %.1f Hz   pior frame %.1f Hz (dt %.3f s)"
          % (st["fps_median"], st["fps_min"], st["dt_max"]))

    print("\n--- MEDICAO #1: arco de oclusao da carroceria ---")
    arc = occlusion_arc(sectors, max_range)
    if not arc:
        print("nenhum setor sempre bloqueado. Se o LiDAR estava montado no carro,")
        print("desconfie: ou o feixe passa por cima da carroceria, ou faltou dado.")
    else:
        graus = [sector_to_deg(i, n_sectors) for i in arc]
        print("%d de %d setores (%.0f%% do circulo) sempre bloqueados"
              % (len(arc), n_sectors, 100.0 * len(arc) / n_sectors))
        print("angulos: %.1f deg a %.1f deg" % (min(graus), max(graus)))
        print("-> fov_deg sugerido para o retreino: %.0f"
              % (360.0 * (n_sectors - len(arc)) / n_sectors))

    print("\n--- MEDICAO #2: onde estao os retornos mais proximos ---")
    if sectors.size:
        medianas = np.median(sectors, axis=0)
        ordem = np.argsort(medianas)[:5]
        print("5 setores mais proximos (compare com onde voce pos o objeto):")
        for i in ordem:
            print("   setor %2d = %6.1f deg   mediana %.3f" % (i, sector_to_deg(i, n_sectors), medianas[i]))

    print("\n--- ocupacao dos setores (quanto a entrada real parece a de treino) ---")
    if sectors.size:
        ocupados = float((sectors < 0.999).mean())
        print("%.1f%% dos valores tem retorno (no dataset_track_v1: frente 45.1%%)"
              % (100.0 * ocupados))


if __name__ == "__main__":
    main()
