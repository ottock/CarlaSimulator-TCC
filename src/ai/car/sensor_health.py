"""Camera cega e dado velho -- puro, Python 3.6-safe.

O envelope de seguranca do laco ja cobria sensor AUSENTE (`read()` devolvendo
None). A bancada mostrou que isso nao basta, porque as falhas reais nao se
apresentam como ausencia:

- uma camera TAPADA continua entregando quadros, so que sem informacao;
- uma camera quebrada pode abrir e nunca entregar nada (aconteceu: 3934 quadros
  com steer 0.000 e nenhum aviso);
- um LiDAR que para de completar voltas deixa o ultimo vetor CONGELADO, e o
  carro segue dirigindo por um mapa que nao existe mais.

Em todos os casos a saida segura e a mesma: servo ao centro e ESC a zero.
"""
import numpy as np

# Desvio padrao minimo (em niveis de cinza) para o quadro ter estrutura. Uma
# lente tapada da uma imagem praticamente uniforme; uma sala escura e' escura
# mas ainda tem textura, e confundir as duas pararia o carro a toa.
MIN_STD = 4.0


def frame_is_blind(frame_bgr, min_std=MIN_STD):
    """True when the frame carries no usable image (missing, covered or saturated)."""
    if frame_bgr is None:
        return True
    arr = np.asarray(frame_bgr)
    if arr.size == 0:
        return True
    return bool(float(arr.std()) < float(min_std))


class StaleTracker:
    """Tells whether the last good reading is too old to act on.

    Distinct from the frame watchdog, which measures how long the LOOP took: a
    fast loop can still be running on sensor data frozen minutes ago. Starts
    stale, because "no reading yet" is not "fresh".
    """

    def __init__(self, timeout_s=0.5):
        self.timeout_s = timeout_s
        self._last = None

    def mark(self, now):
        """Record that a good reading just arrived."""
        self._last = now

    def is_stale(self, now):
        if self._last is None:
            return True
        return (now - self._last) > self.timeout_s
