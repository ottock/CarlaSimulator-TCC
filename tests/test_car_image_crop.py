"""Recorte central da imagem do carro (Fase 6b).

A camera real e de 130 graus; o modelo treinou com 62.2. Recortar o centro
aproxima o enquadramento. A fracao certa depende da projecao da lente e AINDA NAO
foi medida -- por isso e parametro, e 1.0 significa "sem recorte" (para gravar um
log cru durante a calibracao).
"""
import numpy as np
import pytest

from ai.car.image_crop import center_crop, prepare_frame


def _frame(h, w):
    """Frame com um valor unico por pixel, para dar para rastrear o recorte."""
    return np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)


def test_frac_one_is_a_noop():
    f = _frame(360, 640)
    assert np.array_equal(center_crop(f, 1.0), f)


def test_half_crop_halves_both_dimensions():
    out = center_crop(_frame(360, 640), 0.5)
    assert out.shape == (180, 320, 3)


def test_crop_preserves_the_aspect_ratio():
    # mesma fracao nos dois eixos: 16:9 entra, 16:9 sai. Esticar mudaria a
    # geometria da cena e a rede nunca viu o mundo esticado.
    out = center_crop(_frame(720, 1280), 0.48)
    assert out.shape[1] / float(out.shape[0]) == pytest.approx(1280 / 720.0, rel=1e-2)


def test_crop_takes_the_centre_not_a_corner():
    f = np.zeros((10, 10, 3), dtype=np.uint8)
    f[4:6, 4:6, :] = 255                      # marca so o centro
    out = center_crop(f, 0.4)                 # 4x4 central
    assert out.shape == (4, 4, 3)
    assert out.max() == 255


def test_invalid_fraction_is_rejected_loudly():
    # um 0.0 silencioso viraria um frame vazio e a rede receberia lixo
    for bad in (0.0, -0.5):
        with pytest.raises(ValueError):
            center_crop(_frame(10, 10), bad)


def test_prepare_frame_crops_then_resizes_to_the_model_size():
    out = prepare_frame(_frame(720, 1280), crop_frac=0.5, out_size=(640, 360))
    assert out.shape == (360, 640, 3)


def test_prepare_frame_without_crop_still_resizes():
    out = prepare_frame(_frame(720, 1280), crop_frac=1.0, out_size=(640, 360))
    assert out.shape == (360, 640, 3)
