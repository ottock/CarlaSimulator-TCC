"""Espelhamento horizontal para equilibrar o dataset (Fase 6c).

O oval tem as quatro curvas para o MESMO lado: medido no dataset_oval_v1, sao
14.364 quadros de esterco negativo contra 36 positivos em 14.400. Treinado
assim, o modelo aprende "sempre vire para a esquerda" e nao sabe corrigir para
o outro lado nem quando o ruido o joga para la.

Coletar no sentido contrario seria o ideal, mas a pista espelhada nao existe e
dirigir o oval ao contrario quebrou (o carro trava na partida). O espelho
resolve de graça e com simetria EXATA: a camera e central, entao inverter a
imagem e um estado valido da mesma pista vista do outro lado.
"""
import numpy as np
import pytest

from ai.mirror import mirror_lidar, mirror_sample


def test_the_image_is_flipped_left_to_right():
    img = np.zeros((4, 6, 3), dtype=np.uint8)
    img[:, 0] = 255                                  # marca a esquerda
    out_img, _, _ = mirror_sample(img, 0.0, np.ones(8, dtype=np.float32))
    assert (out_img[:, -1] == 255).all()
    assert (out_img[:, 0] == 0).all()


def test_the_steering_changes_sign():
    _, steer, _ = mirror_sample(np.zeros((2, 2, 3), np.uint8), 0.73, np.ones(8, np.float32))
    assert steer == pytest.approx(-0.73)


def test_mirroring_twice_returns_the_original():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 255, (4, 6, 3), dtype=np.uint8)
    lid = rng.random(12).astype(np.float32)
    a = mirror_sample(img, 0.4, lid)
    b = mirror_sample(*a)
    assert np.array_equal(b[0], img)
    assert b[1] == pytest.approx(0.4)
    assert np.allclose(b[2], lid)


# ---------------------------------------------------------------------------
# O LiDAR e a parte que erra fácil
# ---------------------------------------------------------------------------

def test_the_lidar_mirrors_about_the_FORWARD_axis():
    # Setor k cobre [k*w, (k+1)*w) e seu centro e (k+0.5)*w. Espelhar e theta ->
    # -theta, o que manda o setor k para o -k-1 (modulo n). Um simples
    # vetor[::-1] mandaria k para n-1-k, que e a mesma coisa -- mas so porque
    # os centros sao simetricos; o teste fixa o comportamento correto.
    n = 8
    lid = np.zeros(n, dtype=np.float32)
    lid[0] = 0.5          # setor logo a ESQUERDA da frente (centro 22.5 graus)
    out = mirror_lidar(lid)
    assert out[n - 1] == pytest.approx(0.5)
    assert out[0] == 0.0


def test_the_sector_straight_ahead_has_no_single_fixed_point():
    # Com n par nao existe setor centrado em 0: a frente cai na fronteira entre
    # o setor 0 e o n-1, e eles trocam. Se alguem "corrigir" isso para manter o
    # setor 0 no lugar, o vetor fica girado meio setor.
    n = 72
    lid = np.arange(n, dtype=np.float32)
    out = mirror_lidar(lid)
    assert out[0] == pytest.approx(n - 1)
    assert out[n - 1] == pytest.approx(0)
    assert out[35] == pytest.approx(36)


def test_mirroring_the_lidar_twice_is_the_identity():
    rng = np.random.default_rng(7)
    lid = rng.random(72).astype(np.float32)
    assert np.allclose(mirror_lidar(mirror_lidar(lid)), lid)


def test_a_symmetric_scan_is_unchanged():
    lid = np.array([1, 2, 3, 3, 2, 1], dtype=np.float32)
    assert np.allclose(mirror_lidar(lid), lid)
