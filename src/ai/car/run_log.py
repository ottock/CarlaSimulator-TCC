"""Record what the car saw and did, for analysis back on the PC (Fase 6b).

The log IS the deliverable of the wheels-up test: the chassis occlusion arc and
the LiDAR's angle zero/direction come out of it, with numbers. Formats are
deliberately dull -- jsonl and npy read on the PC with no dependencies.

Named ``run_log`` and not ``logging`` on purpose: never shadow the stdlib module
(see the warning in ``requirements.txt``).

Python 3.6-safe.
"""
import io
import json
import os

import cv2
import numpy as np

N_SECTORS = 72


class RunLogger:
    """One directory per run; ``close()`` flushes the sector matrix."""

    def __init__(self, out_dir, meta, jpeg_every=10, n_sectors=N_SECTORS):
        self.out_dir = out_dir
        self.jpeg_every = jpeg_every
        self.n_sectors = n_sectors
        self._sectors = []
        self._frame_i = 0
        self._closed = False
        self._close_count = 0

        # Recusa sobrescrever: um log ja gravado e o entregavel do teste no carro.
        # Erro claro em vez do OSError cru, porque isso estoura DEPOIS de a camera
        # e o LiDAR ja estarem abertos, e a mensagem e' o que o operador ve.
        if os.path.exists(out_dir):
            raise IOError(
                "run directory already exists: {0} -- refusing to overwrite a "
                "recorded log; pass a new output directory".format(out_dir))
        os.makedirs(os.path.join(out_dir, "frames"))
        with io.open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(meta, indent=2, sort_keys=True))
        self._frames_fh = io.open(os.path.join(out_dir, "frames.jsonl"), "w", encoding="utf-8")
        try:
            self._scans_fh = io.open(os.path.join(out_dir, "scans.jsonl"), "w", encoding="utf-8")
        except:
            # Se a segunda abertura falha, a primeira fica aberta com nenhuma referencia
            # alcancavel: __init__ nao retorna, logo ninguem nunca chama close(). Fechar
            # e relancado para nao vazar o handle.
            self._frames_fh.close()
            raise

    def log_frame(self, t, sectors, control, servo_us, dt, frame_bgr=None):
        """Record one control cycle. ``control`` is ``(steer, throttle, brake)``."""
        steer, throttle, brake = control
        row = {"t": float(t), "steer": float(steer), "throttle": float(throttle),
               "brake": float(brake), "servo_us": int(servo_us), "dt": float(dt)}
        self._frames_fh.write(json.dumps(row) + "\n")
        self._sectors.append(np.asarray(sectors, dtype=np.float32))
        if frame_bgr is not None and self._frame_i % self.jpeg_every == 0:
            cv2.imwrite(os.path.join(self.out_dir, "frames",
                                     "%06d.jpg" % self._frame_i), frame_bgr)
        self._frame_i += 1

    def log_scan(self, t, points):
        """Record one raw LiDAR revolution -- the bench measurements come from these."""
        row = {"t": float(t), "points": [[float(a), float(d)] for a, d in points]}
        self._scans_fh.write(json.dumps(row) + "\n")

    def close(self):
        """Flush and close. Safe to call twice (the runtime closes it in a finally)."""
        if self._closed:
            return
        self._closed = True
        self._close_count += 1
        self._frames_fh.close()
        self._scans_fh.close()
        if self._sectors:
            arr = np.stack(self._sectors)
        else:
            arr = np.zeros((0, self.n_sectors), dtype=np.float32)
        np.save(os.path.join(self.out_dir, "sectors.npy"), arr)
