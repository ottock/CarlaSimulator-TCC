"""Cache das imagens recortadas (2026-10-10): tem de entregar a MESMA entrada."""
import numpy as np
import torch

from ai.augment import FAIXAS
from ai.cache_imagens import (DrivingDatasetCache, constroi_cache, constroi_cache_episodio,
                              nome_cache, normaliza, recorta_reduz)
from ai.dataset import DrivingDataset
from ai.dataset_index import build_index, list_episodes
from ai.dataset_writer import EpisodeWriter
from ai.shared.image_pipeline import preprocess


def _cena(i):
    # Imagem com estrutura (gradientes + faixas), nao ruido: o INTER_AREA tira
    # media de blocos, e e assim que a camera ve a pista.
    y, x = np.mgrid[0:360, 0:640]
    b = (x * 0.3 + i * 7) % 256
    g = (y * 0.6 + 40) % 256
    r = np.where(((x // 40) + (y // 30)) % 2 == 0, 200, 60)
    return np.stack([b, g, r], -1).astype(np.uint8)


def _episodio(path, n):
    w = EpisodeWriter(str(path))
    for i in range(n):
        w.add(_cena(i), np.full(72, float(1 + i % 10), dtype=np.float32),
              {"steer": 0.1 * i, "throttle": 0.5, "brake": 0.0, "v": 1.0,
               "x": 0, "y": 0, "yaw": 0, "noise_active": False})
    w.close()


def test_without_augmentation_the_cache_is_bit_identical(tmp_path):
    _episodio(tmp_path / "ep_0000", 6)
    eps = list_episodes(str(tmp_path))
    assert constroi_cache(eps, workers=1) == 6
    idx = build_index(eps)
    a = DrivingDataset(idx, fov_deg=180.0)
    b = DrivingDatasetCache(idx, fov_deg=180.0)
    for i in range(6):
        for ta, tb in zip(a[i], b[i]):
            assert torch.equal(ta, tb)


def test_the_cache_is_the_first_half_of_preprocess():
    img = _cena(3)
    assert np.array_equal(normaliza(recorta_reduz(img)), preprocess(img))


def test_augmenting_the_small_image_stays_close_to_the_big_one():
    # Brilho e contraste comutam com a media; o gama nao, exatamente. Medido aqui.
    from ai.augment import photometric_jitter
    img = _cena(5)
    difs = []
    for s in range(20):
        grande = preprocess(photometric_jitter(img, np.random.default_rng(s), **FAIXAS["v2"]))
        peq = normaliza(photometric_jitter(recorta_reduz(img), np.random.default_rng(s),
                                           **FAIXAS["v2"]))
        difs.append(np.abs(grande - peq).mean())
    assert max(difs) < 0.02                        # em [-1, 1]: < 1% da escala


def test_a_built_cache_is_reused_and_a_short_one_is_rebuilt(tmp_path):
    _episodio(tmp_path / "ep_0000", 4)
    ep = str(tmp_path / "ep_0000")
    caminho, n = constroi_cache_episodio(ep)
    assert n == 4 and caminho.endswith(nome_cache())
    antes = (tmp_path / "ep_0000" / nome_cache()).stat().st_mtime_ns
    constroi_cache_episodio(ep)
    assert (tmp_path / "ep_0000" / nome_cache()).stat().st_mtime_ns == antes
    np.save(str(tmp_path / "ep_0000" / nome_cache()), np.zeros((2, 66, 200, 3), np.uint8))
    _, n = constroi_cache_episodio(ep)
    assert n == 4
    assert np.load(str(tmp_path / "ep_0000" / nome_cache())).shape == (4, 66, 200, 3)
