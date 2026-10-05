"""Saude dos sensores: camera cega e LiDAR velho (Fase 6b).

Duas falhas que o carro exibiu na bancada e que o codigo NAO detectava:

1. Tapar a camera nao para o carro. `cap.read()` continua entregando quadros --
   apenas escuros. Para o laco isso e um quadro valido, e o modelo opina sobre
   uma imagem preta com toda a confianca.
2. Tapar o LiDAR tambem nao para. Se ele deixa de completar voltas, o laco
   mantem o ULTIMO vetor bom para sempre: o carro segue dirigindo por um mapa
   congelado.

E houve uma terceira, cara: no `runs/verifica2` a camera abriu mas nunca
entregou quadro. O programa rodou 15 s, gravou 3934 quadros com steer
exatamente 0.000 e nao avisou nada.
"""
import numpy as np
import pytest

from ai.car.sensor_health import StaleTracker, frame_is_blind


def _frame(value=0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    f = np.full((720, 1280, 3), value, dtype=np.uint8)
    if noise:
        f = np.clip(f + rng.normal(0, noise, f.shape), 0, 255).astype(np.uint8)
    return f


def test_a_covered_lens_is_blind():
    assert frame_is_blind(_frame(0)) is True          # tapada: preto uniforme
    assert frame_is_blind(_frame(255)) is True        # estourada: branco uniforme


def test_a_real_scene_is_not_blind():
    rng = np.random.default_rng(1)
    cena = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    assert frame_is_blind(cena) is False


def test_a_dark_but_real_scene_is_not_blind():
    # Uma sala mal iluminada e escura MAS tem estrutura. Confundir "escuro" com
    # "tapado" pararia o carro a toa num ambiente legitimo.
    assert frame_is_blind(_frame(12, noise=9.0, seed=2)) is False


def test_a_missing_frame_counts_as_blind():
    assert frame_is_blind(None) is True


def test_stale_tracker_is_fresh_right_after_an_update():
    st = StaleTracker(timeout_s=0.5)
    st.mark(100.0)
    assert st.is_stale(100.2) is False


def test_stale_tracker_trips_after_the_timeout():
    st = StaleTracker(timeout_s=0.5)
    st.mark(100.0)
    assert st.is_stale(100.6) is True


def test_stale_tracker_starts_stale_until_the_first_update():
    # Antes da primeira volta completa nao ha mapa nenhum -- e isso e' "velho",
    # nao "novo".
    assert StaleTracker(timeout_s=0.5).is_stale(0.0) is True


# ---------------------------------------------------------------------------
# Custo por quadro (2026-10-04)
# ---------------------------------------------------------------------------

def test_the_blind_check_subsamples_instead_of_reading_every_pixel():
    # Media o quadro INTEIRO custava 2,76 milhoes de bytes por quadro num Nano
    # que so entrega 16 Hz. O desvio padrao de uma amostra regular responde a
    # mesma pergunta -- "ha textura aqui?" -- por 1/64 do trabalho.
    import time
    grande = _frame(0, noise=40, seed=5)
    t0 = time.perf_counter()
    for _ in range(20):
        frame_is_blind(grande)
    rapido = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(20):
        float(np.asarray(grande).std())
    completo = time.perf_counter() - t0
    assert rapido < completo / 4, (
        "a checagem nao esta subamostrando: %.1f ms contra %.1f ms do quadro inteiro"
        % (1000 * rapido, 1000 * completo))


def test_subsampling_does_not_change_the_verdict():
    # O ganho nao vale nada se mudar a resposta nos casos que importam.
    assert frame_is_blind(_frame(0)) is True
    assert frame_is_blind(_frame(255)) is True
    rng = np.random.default_rng(11)
    assert frame_is_blind(rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)) is False
