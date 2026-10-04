"""Espelhamento horizontal de amostras (Fase 6c).

O `oval_tcc` tem as quatro curvas para o MESMO lado. Medido no
`dataset_oval_v1`: 14.364 quadros de esterco negativo contra 36 positivos em
14.400. Treinado assim, o modelo aprende "sempre vire para a esquerda" e nao
sabe voltar quando algo o joga para o outro lado.

Coletar no sentido contrario seria o ideal, mas dirigir o oval ao contrario nao
funcionou (o carro trava na partida) e a pista espelhada nao existe em props.
O espelho resolve com simetria EXATA e sem simulacao nenhuma: a camera e
central, entao a imagem invertida e um estado valido da mesma pista vista do
outro lado, com o esterco trocado de sinal.

Puro: nao importa torch nem carla.
"""
import numpy as np


def mirror_lidar(sectors):
    """Espelha o vetor de setores em torno do eixo FRONTAL.

    Setor ``k`` cobre ``[k*w, (k+1)*w)`` e tem centro em ``(k+0.5)*w``.
    Espelhar e ``theta -> -theta``, o que leva o centro ``(k+0.5)*w`` para
    ``-(k+0.5)*w``, isto e, ao setor ``-k-1`` (modulo n) -- exatamente o
    inverso da ordem. Com ``n`` par nao ha setor fixo: a frente cai na fronteira
    entre o setor 0 e o ``n-1``, e os dois trocam de lugar.
    """
    return np.asarray(sectors, dtype=np.float32)[::-1].copy()


def mirror_sample(image, steer, sectors):
    """Espelha imagem, esterco e LiDAR de uma amostra.

    Returns:
        ``(imagem_espelhada, -steer, setores_espelhados)``.
    """
    img = np.asarray(image)[:, ::-1].copy()
    return img, -float(steer), mirror_lidar(sectors)
