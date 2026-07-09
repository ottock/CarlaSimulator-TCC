"""ExpertPolicy — Estagio A (Towns padrao).

Envolve o `BasicAgent` do CARLA numa interface de POLITICA identica a que o modelo
de rede vai usar depois:

    steer, a = policy.run_step(obs)

onde `steer` e `a` sao normalizados em [-1, 1] (o mesmo espaco de acao do modelo),
com `a` = eixo longitudinal assinado (`a = throttle - brake`; positivo acelera,
negativo freia). Ver `docs/PLANO_IA.md` secao 6 (design das saidas).

O BasicAgent dirige seguindo a malha OpenDRIVE do mapa (por isso so funciona nos
Towns padrao — Estagio A). Quando chega ao destino, um novo destino aleatorio e
sorteado para a direcao continuar indefinidamente.
"""

# imports
import random
import logging

# Injeta o pacote `agents` do CARLA no sys.path antes de importa-lo.
from ai.paths import ensure_carla_agents_on_path

ensure_carla_agents_on_path()

import carla  # noqa: E402  (import apos ajuste de sys.path)
from agents.navigation.basic_agent import BasicAgent  # noqa: E402

# constants
logger = logging.getLogger(__name__)


# classes
class ExpertPolicy:
    """Politica de direcao baseada no BasicAgent do CARLA.

    Args:
        vehicle: Veiculo ego ja spawnado.
        world: Mundo CARLA (para obter o mapa e os spawn points).
        target_speed_kmh: Velocidade alvo do agente, em km/h.
        seed: Semente para o sorteio de destinos (reprodutibilidade).
    """

    def __init__(
        self,
        vehicle: carla.Vehicle,
        world: carla.World,
        target_speed_kmh: float = 25.0,
        seed: int = 42,
    ):
        self._vehicle = vehicle
        self._world = world
        self._map = world.get_map()
        self._spawn_points = self._map.get_spawn_points()
        self._rng = random.Random(seed)

        if not self._spawn_points:
            raise RuntimeError("Mapa sem spawn points — BasicAgent nao tem para onde ir.")

        self._agent = BasicAgent(vehicle, target_speed=target_speed_kmh)
        self._set_random_destination()
        logger.info("ExpertPolicy (BasicAgent) pronto — alvo %.0f km/h", target_speed_kmh)

    def _set_random_destination(self) -> None:
        """Sorteia e define um novo destino a partir dos spawn points do mapa."""
        destino = self._rng.choice(self._spawn_points).location
        self._agent.set_destination(destino)
        logger.debug("Novo destino: (%.1f, %.1f, %.1f)", destino.x, destino.y, destino.z)

    def run_step(self, obs=None) -> tuple:
        """Calcula a acao do expert para o tick atual.

        Args:
            obs: Ignorado (o BasicAgent usa a malha do mapa, nao os sensores).
                 Existe na assinatura para casar com a interface do modelo.

        Returns:
            Tupla (steer, a) com steer em [-1, 1] e a = throttle - brake em [-1, 1].
        """
        if self._agent.done():
            self._set_random_destination()

        control = self._agent.run_step()
        steer = float(control.steer)
        a = float(control.throttle) - float(control.brake)
        return steer, a
