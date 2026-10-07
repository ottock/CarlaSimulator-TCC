"""O teste de ponto cego, rodado de ponta a ponta com um LiDAR falso.

A primeira versao de `hardware/testa_ponto_cego.py` tomava o minimo sobre TODOS
os pontos. Como a carroceria fica a ~0,20 m do sensor, ela teria anunciado
"ponto cego confirmado" com qualquer coisa que o operador fizesse. Nenhum teste
existia para pegar isso. Agora roda o `main()` inteiro contra um sensor falso
com a geometria real do carro (offset 126.8, invertido, carroceria em 0:120).
"""
import importlib.util
import io
import os
import sys
import types

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_REPO, "hardware", "testa_ponto_cego.py")

OFFSET = 126.8
# Angulo do SENSOR que aponta para a frente do carro: com invert,
# carro = (-a - 126.8) % 360 = 0  =>  a = 233.2.
FRENTE_SENSOR = 233.2


def _carrega():
    spec = importlib.util.spec_from_file_location("testa_ponto_cego", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _volta(parede_frente_m, chao_m=None):
    """Uma revolucao do sensor, em ordem crescente de angulo.

    - carroceria (0..120 graus do sensor) a 0,202 m, como nas rodadas;
    - fundo a 1,0 m;
    - uma parede de 15 graus centrada na FRENTE do carro, a `parede_frente_m`.
    `chao_m` simula um ponto cego: leituras abaixo dele somem.
    """
    pts = []
    a = 0.0
    while a < 360.0:
        if a <= 120.0:
            d = 0.202
        elif abs(a - FRENTE_SENSOR) <= 7.5:
            d = parede_frente_m
        else:
            d = 1.0
        if chao_m is None or d >= chao_m:
            pts.append((a, d))
        a += 1.5
    return pts


class _LidarFalso(object):
    """Entrega uma volta por leitura; a parede da frente se aproxima com o tempo."""

    def __init__(self, distancias, chao_m=None):
        self.distancias = list(distancias)
        self.chao_m = chao_m
        self.i = 0

    def read_points(self):
        d = self.distancias[min(self.i, len(self.distancias) - 1)]
        self.i += 1
        return _volta(d, self.chao_m)

    def close(self):
        pass


def _roda(monkeypatch, tmp_path, lidar):
    jr = types.ModuleType("jetson_runtime")
    jr.SELF_OCCLUSION_ARCS = "0:120,210:215"
    jr.LIDAR_OFFSET_DEG = OFFSET
    jr.LIDAR_INVERT = True
    jr.SerialLidar = lambda *a, **k: lidar
    monkeypatch.setitem(sys.modules, "jetson_runtime", jr)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["testa_ponto_cego.py", "--segundos", "0.6",
                                      "--sem-janela"])
    _carrega().main()
    with io.open(os.path.join(str(tmp_path), "ponto_cego.txt"), encoding="utf-8") as fh:
        return fh.read()


def test_the_car_body_alone_never_reads_as_a_blind_zone(monkeypatch, tmp_path):
    """O bug da primeira versao: so a carroceria perto, parede longe.
    Tem de dizer que o teste nao decide -- nunca 'ponto cego'."""
    texto = _roda(monkeypatch, tmp_path, _LidarFalso([0.8]))
    assert "MENOR distancia de PAREDE: 0.800 m" in texto
    assert "Nenhuma parede chegou a menos de 40 cm" in texto
    assert "IGNORADA" in texto                     # a carroceria foi vista e descartada
    assert "ponto cego de" not in texto


def test_a_wall_brought_to_8_cm_is_measured_at_8_cm(monkeypatch, tmp_path):
    rampa = [0.40 - 0.004 * k for k in range(81)]          # 40 cm ate 8 cm
    texto = _roda(monkeypatch, tmp_path, _LidarFalso(rampa))
    assert "MENOR distancia de PAREDE: 0.080 m" in texto
    assert "MEDE abaixo de 10 cm" in texto


def test_a_real_blind_zone_shows_up_as_a_floor(monkeypatch, tmp_path):
    """Sensor que nao le abaixo de 0,20: a parede chega a 8 cm mas some antes."""
    rampa = [0.40 - 0.004 * k for k in range(81)]
    texto = _roda(monkeypatch, tmp_path, _LidarFalso(rampa, chao_m=0.20))
    menor = [l for l in texto.splitlines() if l.startswith("MENOR distancia de PAREDE")][0]
    valor = float(menor.split(":")[1].split()[0])
    assert 0.20 <= valor < 0.21
    assert "Nada de parede abaixo de" in texto


def test_the_png_is_written_even_without_a_screen(monkeypatch, tmp_path):
    _roda(monkeypatch, tmp_path, _LidarFalso([0.3]))
    assert os.path.getsize(os.path.join(str(tmp_path), "ponto_cego.png")) > 1000


@pytest.mark.parametrize("menor,trecho", [
    (None, "Nenhuma parede"),
    (0.55, "Nenhuma parede"),
    (0.06, "MEDE abaixo de 10 cm"),
    (0.15, "Ve de perto o suficiente"),
    (0.21, "SO conta como ponto cego se voce de fato ENCOSTOU"),
])
def test_the_verdict_never_claims_more_than_the_data(menor, trecho):
    assert trecho in "\n".join(_carrega().veredito(menor))
