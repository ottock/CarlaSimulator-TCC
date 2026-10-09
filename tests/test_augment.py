"""Aumento fotometrico para fechar a lacuna sim->real (Fase 6c).

Medido nos logs de pista de 2026-09-12, na entrada do modelo (pixels 0-255):

    treino (simulador) : media 120   desvio 75
    real   (carro)     : media  79   desvio 36

O real e mais escuro e tem METADE do contraste. Faz sentido: no simulador o chao
e preto e a parede branca sob luz uniforme; na pista real o chao e cinza medio, o
isopor e esbranquicado e ha contraluz de janela. O modelo aprendeu a achar a
borda parede/chao num contraste que nao existe la.

As faixas abaixo tem de COBRIR essa diferenca, senao o aumento nao resolve nada.
"""
import numpy as np
import pytest

from ai.augment import photometric_jitter


def _treino_like(seed=0):
    """Imagem com as estatisticas do simulador: media ~120, desvio ~75."""
    rng = np.random.default_rng(seed)
    x = rng.normal(120, 75, (66, 200, 3))
    return np.clip(x, 0, 255).astype(np.uint8)


def test_identity_ranges_change_nothing():
    # noise_std tambem precisa ser zerado: o granulado e uma quarta distorcao,
    # e sem desliga-lo "identidade" deixou de ser identidade quando ele entrou.
    img = _treino_like()
    out = photometric_jitter(img, np.random.default_rng(0),
                             contrast=(1.0, 1.0), brightness=(0.0, 0.0), gamma=(1.0, 1.0),
                             noise_std=(0.0, 0.0))
    assert np.array_equal(out, img)


def test_shape_and_dtype_survive():
    img = _treino_like()
    out = photometric_jitter(img, np.random.default_rng(1))
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_output_never_leaves_the_valid_range():
    # Um estouro virando wrap-around (255 -> 0) inventaria bordas que nao existem.
    img = _treino_like()
    out = photometric_jitter(img, np.random.default_rng(2),
                             contrast=(3.0, 3.0), brightness=(200.0, 200.0), gamma=(1.0, 1.0))
    assert out.min() >= 0 and out.max() <= 255


def test_lower_contrast_reduces_the_spread():
    img = _treino_like()
    out = photometric_jitter(img, np.random.default_rng(3),
                             contrast=(0.5, 0.5), brightness=(0.0, 0.0), gamma=(1.0, 1.0))
    assert out.std() < img.std() * 0.7


def test_negative_brightness_darkens():
    img = _treino_like()
    out = photometric_jitter(img, np.random.default_rng(4),
                             contrast=(1.0, 1.0), brightness=(-40.0, -40.0), gamma=(1.0, 1.0))
    assert out.mean() < img.mean() - 30


def test_the_range_covers_the_measured_real_camera():
    """O teste que amarra o aumento a medicao: as faixas padrao TEM de alcancar
    media ~79 e desvio ~36, senao o modelo nunca vera algo parecido com a pista."""
    img = _treino_like()
    medias, desvios = [], []
    rng = np.random.default_rng(5)
    for _ in range(200):
        out = photometric_jitter(img, rng)
        medias.append(out.mean())
        desvios.append(out.std())
    assert min(medias) <= 79, "o aumento nunca chega ao brilho real (%.0f)" % min(medias)
    assert min(desvios) <= 36, "o aumento nunca chega ao contraste real (%.0f)" % min(desvios)
    # e nao pode ser SO escuro: o modelo ainda precisa ver o caso do simulador
    assert max(medias) >= 115


# ---------------------------------------------------------------------------
# Cobertura do contraste real (2026-10-05)
# ---------------------------------------------------------------------------

def test_the_ranges_now_reach_the_contrast_the_car_really_produces():
    # Medido: contraste por quadro 17.3 no carro contra 58.2 no treino. Com as
    # faixas antigas (0.45 minimo) so 3% das amostras chegavam la.
    from ai.augment import CONTRAST
    assert CONTRAST[0] <= 17.3 / 58.2, (
        "contraste minimo %.2f nao alcanca o %.2f que o carro entrega"
        % (CONTRAST[0], 17.3 / 58.2))


# ---------------------------------------------------------------------------
# Faixas por nome (2026-10-09): retreinar mudando UMA coisa so
# ---------------------------------------------------------------------------

def test_the_v2_ranges_are_the_ones_driving_oval_v2_was_trained_with():
    # Lidos de `git show 2dd78f2:src/ai/augment.py`. Se mudarem, o v4 deixa de
    # repetir a receita do v2 e o teste da margem passa a medir duas coisas.
    from ai.augment import FAIXAS
    assert FAIXAS["v2"] == {"contrast": (0.45, 1.15), "brightness": (-50.0, 25.0),
                            "gamma": (0.7, 1.6), "noise_std": (0.0, 0.0)}


def test_the_v3_ranges_are_the_current_defaults():
    from ai.augment import BRIGHTNESS, CONTRAST, FAIXAS, GAMMA, NOISE_STD
    assert FAIXAS["v3"] == {"contrast": CONTRAST, "brightness": BRIGHTNESS,
                            "gamma": GAMMA, "noise_std": NOISE_STD}
