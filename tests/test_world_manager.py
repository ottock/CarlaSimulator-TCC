"""O mundo so e recarregado quando o mapa e outro (2026-10-09).

Recarregar o Town10HD por cima dele mesmo derrubou o CARLA duas vezes com "Out
of video memory" numa RTX 3070 de 8 GB: durante a troca a GPU segura os dois
mapas. Estes testes usam dubles, sem servidor.
"""
import pytest

pytest.importorskip("carla")

from core.carlaClient.world_manager import _mesmo_mapa, _preparar_mundo


class Ator(object):
    def __init__(self, tipo):
        self.type_id = tipo
        self.destruido = False

    def destroy(self):
        self.destruido = True


class Atores(list):
    def filter(self, padrao):
        prefixo = padrao.rstrip("*")
        return [a for a in self if a.type_id.startswith(prefixo)]


class Mapa(object):
    def __init__(self, nome):
        self.name = nome


class Mundo(object):
    def __init__(self, nome, atores=()):
        self._mapa = Mapa(nome)
        self._atores = Atores(atores)

    def get_map(self):
        return self._mapa

    def get_actors(self):
        return self._atores


class Cliente(object):
    def __init__(self, mundo):
        self.mundo = mundo
        self.recarregamentos = []

    def get_world(self):
        return self.mundo

    def load_world(self, nome):
        self.recarregamentos.append(nome)
        self.mundo = Mundo("Carla/Maps/" + nome)
        return self.mundo


@pytest.mark.parametrize("atual,pedido,igual", [
    ("Carla/Maps/Town10HD_Opt", "Town10HD_Opt", True),
    ("Carla/Maps/Town10HD_Opt", "Carla/Maps/Town10HD_Opt", True),
    ("Carla/Maps/Town10HD_Opt", "Town10HD", False),
    ("Carla/Maps/Town01", "Town10HD_Opt", False),
])
def test_same_map_ignores_the_path(atual, pedido, igual):
    assert _mesmo_mapa(atual, pedido) is igual


def test_the_map_already_loaded_is_reused_not_reloaded():
    cliente = Cliente(Mundo("Carla/Maps/Town10HD_Opt"))
    mundo = _preparar_mundo(cliente, "Town10HD_Opt")
    assert cliente.recarregamentos == []
    assert mundo is cliente.mundo


def test_another_map_is_loaded():
    cliente = Cliente(Mundo("Carla/Maps/Town01"))
    mundo = _preparar_mundo(cliente, "Town10HD_Opt")
    assert cliente.recarregamentos == ["Town10HD_Opt"]
    assert mundo.get_map().name.endswith("Town10HD_Opt")


def test_leftovers_of_a_crashed_run_are_removed_but_the_map_stays():
    # Paredes duplicadas fariam o expert colidir com fantasmas.
    restos = [Ator("static.prop.tcc_mureta"), Ator("vehicle.tesla.model3"),
              Ator("sensor.camera.rgb"), Ator("walker.pedestrian.0001")]
    do_mapa = [Ator("traffic.traffic_light"), Ator("traffic.stop")]
    cliente = Cliente(Mundo("Carla/Maps/Town10HD_Opt", restos + do_mapa))
    _preparar_mundo(cliente, "Town10HD_Opt")
    assert all(a.destruido for a in restos)
    assert not any(a.destruido for a in do_mapa)
