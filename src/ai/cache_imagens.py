"""Cache das imagens ja recortadas e reduzidas, para o treino nao refazer isso (2026-10-10).

Medido no treino do v6: 98 mil quadros, ~6 min por epoca, placa de video a 1-17%
de uso. O tempo ia todo em ler o JPEG de 640x360, descomprimir, recortar e
reduzir para 66x200 -- as MESMAS 98 mil fotos, 25 vezes. A rede em si leva
20-30 s por epoca.

Aqui cada episodio ganha um ``.npy`` com as imagens no ponto em que o
``preprocess`` ja recortou e reduziu (``(N, 66, 200, 3)`` uint8 BGR, ~40 KB por
quadro), lido sob demanda (memmap). O que muda de epoca para epoca continua
sorteado a cada amostra:

  - aumento fotometrico: agora aplicado na imagem PEQUENA. Brilho e contraste
    sao afins e comutam com a media do INTER_AREA (fora o grampo em 0/255); o
    gama nao comuta exatamente -- a diferenca e medida nos testes e e pequena;
  - espelho horizontal: igual.

Sem o aumento o resultado e IDENTICO ao do ``preprocess`` (mesmas operacoes, na
mesma ordem): os testes conferem bit a bit.
"""
import os
from multiprocessing import Pool

import cv2
import numpy as np
import torch

from ai.augment import photometric_jitter
from ai.dataset import DrivingDataset
from ai.mirror import mirror_lidar
from ai.shared.image_pipeline import CROP_BOTTOM, CROP_TOP, OUT_SIZE
from ai.shared.lidar_pipeline import apply_fov_mask, normalize_sectors_m


def nome_cache(crop_top=CROP_TOP, crop_bottom=CROP_BOTTOM, out_size=OUT_SIZE):
    """O nome carrega os parametros: mudar o recorte invalida o cache sozinho."""
    return "cache_t%d_b%d_%dx%d.npy" % (crop_top, crop_bottom, out_size[0], out_size[1])


def recorta_reduz(img_bgr, crop_top=CROP_TOP, crop_bottom=CROP_BOTTOM, out_size=OUT_SIZE):
    """Os dois primeiros passos do ``preprocess``, em uint8 (sem normalizar)."""
    h = img_bgr.shape[0]
    return cv2.resize(img_bgr[crop_top:h - crop_bottom, :, :], out_size,
                      interpolation=cv2.INTER_AREA)


def normaliza(peq_bgr):
    """Os dois ultimos passos do ``preprocess``: [-1, 1] e HWC -> CHW."""
    return np.transpose(peq_bgr.astype(np.float32) / 127.5 - 1.0, (2, 0, 1))


def constroi_cache_episodio(ep_dir):
    """Escreve o cache de um episodio (se faltar). Devolve ``(caminho, n_quadros)``."""
    destino = os.path.join(ep_dir, nome_cache())
    n = len(np.load(os.path.join(ep_dir, "lidar.npy"), mmap_mode="r"))
    if os.path.exists(destino):
        try:
            if len(np.load(destino, mmap_mode="r")) == n:
                return destino, n
        except (ValueError, OSError):
            pass                                  # cache corrompido: refaz
    w, h = OUT_SIZE
    out = np.empty((n, h, w, 3), dtype=np.uint8)
    for i in range(n):
        img = cv2.imread(os.path.join(ep_dir, "frames", "%06d.jpg" % i), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(os.path.join(ep_dir, "frames", "%06d.jpg" % i))
        out[i] = recorta_reduz(img)
    # Escrita atomica: um treino lendo junto nunca ve um arquivo pela metade.
    tmp = destino + ".tmp.npy"
    np.save(tmp, out)
    os.replace(tmp, destino)
    return destino, n


def constroi_cache(episodios, workers=8):
    """Cache de todos os episodios, em paralelo. Devolve quantos quadros cobre."""
    if workers and workers > 1:
        with Pool(int(workers)) as pool:
            res = pool.map(constroi_cache_episodio, list(episodios), chunksize=1)
    else:
        res = [constroi_cache_episodio(e) for e in episodios]
    return sum(n for _, n in res)


class DrivingDatasetCache(DrivingDataset):
    """O ``DrivingDataset``, lendo do cache em vez do JPEG. Mesmos alvos e LiDAR."""

    def __init__(self, index, **kw):
        DrivingDataset.__init__(self, index, **kw)
        self._img_cache = {}

    def _imagens(self, ep_dir):
        arr = self._img_cache.get(ep_dir)
        if arr is None:
            arr = np.load(os.path.join(ep_dir, nome_cache()), mmap_mode="r")
            self._img_cache[ep_dir] = arr
        return arr

    def __getitem__(self, i):
        rec = self.index[i]
        ep_dir = os.path.dirname(os.path.dirname(rec["image"]))
        peq = np.asarray(self._imagens(ep_dir)[rec["row"]])
        if self.photometric:
            peq = photometric_jitter(peq, np.random.default_rng(), **self.faixas)
        x = normaliza(peq)
        sectors_m = np.asarray(self._lidar_array(rec["lidar"])[rec["row"]], dtype=np.float32)
        if self.fov_deg is not None:
            sectors_m = apply_fov_mask(sectors_m, self.fov_deg, self.max_range)
        lidar = normalize_sectors_m(sectors_m, self.max_range)
        steer = rec["steer"]
        if self.mirror and np.random.default_rng().random() < 0.5:
            x = x[:, :, ::-1].copy()
            lidar = mirror_lidar(lidar)
            steer = -steer
        target = np.array([steer, rec["throttle"], rec["brake"]], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(lidar), torch.from_numpy(target)
