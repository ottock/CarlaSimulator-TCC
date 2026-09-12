"""Atribuicao do esterco a cada sensor, a partir de uma corrida real (Fase 6c).

Ate agora a pergunta "foi a camera ou o LiDAR?" so tinha resposta por inferencia
(correlacao entre esterco e posicao da parede). Com a entrada exata gravada, da
para RE-EXECUTAR o modelo e neutralizar um sensor de cada vez: o que muda mais o
esterco e o que estava mandando.
"""
import numpy as np
import pytest

from replay_attrib import attribution


def test_a_model_driven_by_the_camera_reacts_to_blinding_it():
    full = np.array([0.5, -0.3, 0.2])
    sem_lidar = np.array([0.5, -0.3, 0.2])        # tirar o LiDAR nao muda nada
    sem_camera = np.array([0.0, 0.0, 0.0])        # tirar a camera muda tudo
    a = attribution(full, sem_lidar, sem_camera)
    assert a["delta_lidar"] == pytest.approx(0.0)
    assert a["delta_camera"] > 0.3
    assert a["dominante"] == "camera"


def test_a_model_driven_by_the_lidar_reacts_to_blinding_it():
    full = np.array([0.5, -0.3, 0.2])
    sem_lidar = np.array([0.0, 0.0, 0.0])
    sem_camera = np.array([0.5, -0.3, 0.2])
    a = attribution(full, sem_lidar, sem_camera)
    assert a["dominante"] == "lidar"


def test_a_balanced_model_reports_neither_as_dominant():
    full = np.array([0.5, -0.3])
    sem_lidar = np.array([0.3, -0.1])
    sem_camera = np.array([0.3, -0.1])
    a = attribution(full, sem_lidar, sem_camera)
    assert a["dominante"] == "equilibrado"


def test_the_replay_fidelity_is_reported():
    """Se o replay nao reproduz o que o carro fez, nada acima vale.

    Comparar com o steer gravado e o unico jeito de saber que o replay usa as
    MESMAS entradas -- diferencas de TensorRT vs PyTorch ou um erro de recorte
    apareceriam aqui antes de contaminarem a atribuicao.
    """
    full = np.array([0.5, -0.3, 0.2])
    a = attribution(full, full, full, gravado=np.array([0.5, -0.3, 0.2]))
    assert a["fidelidade"] == pytest.approx(0.0)
    b = attribution(full, full, full, gravado=np.array([0.1, 0.1, 0.1]))
    assert b["fidelidade"] > 0.2
