#!/usr/bin/env python3
"""Harness de avaliacao em MALHA FECHADA (Fase 0 do docs/PLANO_IA.md).

Este e o "esqueleto" do laco de controle e o PRIMEIRO entregavel do TCC:

    obs = ler_sensores()          # (por enquanto nao usado pela politica)
    steer, a = policy(obs)        # decidir
    ego.apply_control(...)        # atuar
    world.tick()                  # avancar a simulacao (modo sincrono)
    medir()                       # desvio de faixa, velocidade, etc.

A `policy` e injetavel: agora e o ExpertPolicy (BasicAgent); nas fases seguintes,
o MESMO ponto de plugagem recebe o modelo de rede, sem mudar a estrutura do laco.

Uso (com o servidor CARLA JA rodando):
    python src/ai/eval_closedloop.py settings/baseSettings.json
    python src/ai/eval_closedloop.py settings/baseSettings.json --seconds 120

Ou deixando o harness subir o servidor:
    python src/ai/eval_closedloop.py settings/baseSettings.json --launch
"""

# imports
import os
import sys
import time
import math
import argparse
import subprocess
from datetime import datetime

# --- bootstrap de sys.path: garante 'src' e o pacote 'agents' do CARLA ---
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from ai.paths import ensure_carla_agents_on_path, PROJECT_ROOT  # noqa: E402

ensure_carla_agents_on_path()

import carla  # noqa: E402
from core.logger.logger import setup_logger  # noqa: E402
from utils.config import load_settings  # noqa: E402
from core.carlaClient.world_manager import (  # noqa: E402
    connect_to_carla,
    simulation_context,
    spawn_actor_vehicle,
    setup_spectator_follow_vehicle,
    update_spectator_position,
)
from ai.expert.basic_agent_expert import ExpertPolicy  # noqa: E402

# constants
logger = setup_logger(
    logger_name=__name__,
    log_filename=f"log_{datetime.now().strftime('%Y-%m-%d')}.log",
)

OFF_LANE_THRESHOLD_M = 1.5  # desvio ao centro da faixa que conta como "fora"
SETTLE_TICKS = 20           # ticks de assentamento apos o spawn (spawn vem ~0.3m no ar)


# functions
def _speed_kmh(vehicle: carla.Vehicle) -> float:
    """Velocidade escalar do veiculo em km/h."""
    v = vehicle.get_velocity()
    return 3.6 * math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def _launch_carla_server(carla_config: dict, quality: str = "Epic"):
    """Sobe o servidor CarlaUE4.exe.

    Replica o launcher do `main.py` localmente de proposito: importar `main.py`
    dispara o `setup_logger` dele no nivel de modulo, que tenta apagar o log que
    ESTE processo ja mantem aberto (PermissionError no Windows).

    Args:
        carla_config: Secao `carla_client` do settings.json.
        quality: Nivel grafico ('Epic' ou 'Low'). 'Low' e mais leve/estavel.

    Returns:
        subprocess.Popen do servidor, ou None se o executavel nao existir/falhar.
    """
    carla_root = os.path.join(PROJECT_ROOT, "CARLA_0.9.16")
    exe_path = os.path.join(carla_root, "CarlaUE4.exe")
    if not os.path.exists(exe_path):
        logger.warning("CarlaUE4.exe nao encontrado em %s", exe_path)
        return None

    port = carla_config.get("port", 2000)
    cmd = [exe_path, "-carla-port=%d" % port, "-quality-level=%s" % quality, "-nosound"]
    logger.info("Subindo servidor CARLA: %s (porta %d)", exe_path, port)
    try:
        return subprocess.Popen(cmd, cwd=carla_root)
    except Exception:
        logger.exception("Falha ao subir o servidor CARLA")
        return None


def _lane_deviation_m(world_map: carla.Map, vehicle: carla.Vehicle) -> float:
    """Distancia do veiculo ao centro da faixa de rolamento mais proxima (m).

    Usa a malha OpenDRIVE do mapa (`get_waypoint`) — valido nos Towns padrao.
    """
    loc = vehicle.get_location()
    wp = world_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp is None:
        return float("nan")
    return loc.distance(wp.transform.location)


def run_closed_loop(settings: dict, max_seconds: int, target_speed_kmh: float) -> None:
    """Roda o laco de malha fechada com o expert dirigindo.

    Args:
        settings: Dicionario de settings ja carregado (mapa, fisica, conexao...).
        max_seconds: Duracao maxima; 0 = roda ate Ctrl+C.
        target_speed_kmh: Velocidade alvo do expert.
    """
    carla_config = settings["carla_client"]
    world_config = settings["world"]

    # Esqueleto minimo: ego SEM sensores e SEM trafego (adicionados nas proximas fases).
    actor_cfg = dict(world_config.get("actor_vehicle", {}))
    actor_cfg["use_lidar"] = False
    actor_cfg["use_camera"] = False

    client = connect_to_carla(carla_config)

    with simulation_context(client, world_config) as (world, actor_list):
        ego, _ = spawn_actor_vehicle(world, actor_list, actor_cfg, traffic_manager=None)

        # Assenta o veiculo no chao ANTES de dirigir. Spawn points do CARLA ficam
        # ~0.3 m no ar; se o controlador do agente rodar enquanto o carro ainda
        # cai, o PID de estercamento satura (-0.8) e "enrola" o integral, e o carro
        # sai da pista sem se recuperar (diagnosticado em Fase 0). O freio de mao
        # segura o carro parado enquanto ele assenta.
        for _ in range(SETTLE_TICKS):
            ego.apply_control(carla.VehicleControl(hand_brake=True))
            world.tick()
        ego.apply_control(carla.VehicleControl(hand_brake=False))
        world.tick()
        logger.info("Veiculo assentado (%d ticks); iniciando controle.", SETTLE_TICKS)

        world_map = world.get_map()

        policy = ExpertPolicy(
            ego, world,
            target_speed_kmh=target_speed_kmh,
            seed=world_config.get("seed", 42),
        )
        spectator_state = setup_spectator_follow_vehicle(world, ego, mode="behind")

        fixed_delta = world.get_settings().fixed_delta_seconds or 0.05
        logger.info(
            "Malha fechada iniciada (dt=%.3fs, ~%.0f Hz). Ctrl+C para sair.",
            fixed_delta, 1.0 / fixed_delta,
        )

        t_start = time.perf_counter()
        last_report = t_start
        n_ticks = 0
        dev_sum = 0.0
        dev_max = 0.0
        off_lane_ticks = 0

        try:
            while True:
                # 1. decidir (a politica ignora obs por enquanto)
                steer, a = policy.run_step(obs=None)

                # 2. atuar — eixo longitudinal assinado -> throttle/brake
                control = carla.VehicleControl(
                    throttle=max(a, 0.0),
                    brake=max(-a, 0.0),
                    steer=steer,
                )
                ego.apply_control(control)

                # 3. avancar a simulacao
                world.tick()
                update_spectator_position(world, spectator_state)

                # 4. medir
                dev = _lane_deviation_m(world_map, ego)
                if not math.isnan(dev):
                    dev_sum += dev
                    dev_max = max(dev_max, dev)
                    if dev > OFF_LANE_THRESHOLD_M:
                        off_lane_ticks += 1
                n_ticks += 1

                now = time.perf_counter()
                if now - last_report >= 1.0:
                    logger.info(
                        "v=%5.1f km/h | steer=%+.2f a=%+.2f | desvio=%.2fm (med=%.2f max=%.2f) | ticks=%d",
                        _speed_kmh(ego), steer, a, dev,
                        dev_sum / max(n_ticks, 1), dev_max, n_ticks,
                    )
                    last_report = now

                if max_seconds and (now - t_start) >= max_seconds:
                    logger.info("Tempo limite (%ds) atingido.", max_seconds)
                    break

        except KeyboardInterrupt:
            logger.info("Interrompido pelo usuario.")

        logger.info(
            "=== Resumo: %d ticks | desvio medio=%.2fm max=%.2fm | "
            "fora de faixa (>%.1fm)=%d ticks (%.1f%%) ===",
            n_ticks, dev_sum / max(n_ticks, 1), dev_max, OFF_LANE_THRESHOLD_M,
            off_lane_ticks, 100.0 * off_lane_ticks / max(n_ticks, 1),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harness de malha fechada (Fase 0) — expert dirige o ego no CARLA."
    )
    parser.add_argument(
        "settings_json", nargs="?", default="settings/baseSettings.json",
        help="Caminho do settings.json (padrao: settings/baseSettings.json).",
    )
    parser.add_argument("--seconds", type=int, default=0, help="Duracao; 0 = ate Ctrl+C.")
    parser.add_argument("--target-speed", type=float, default=25.0, help="Velocidade alvo (km/h).")
    parser.add_argument(
        "--launch", action="store_true",
        help="Sobe o servidor CARLA antes de conectar (senao, espera um ja rodando).",
    )
    parser.add_argument(
        "--quality", default="Epic", choices=["Epic", "Low"],
        help="Nivel grafico do servidor lancado com --launch (Low = mais estavel).",
    )
    args = parser.parse_args()

    settings = load_settings(args.settings_json)

    server = None
    try:
        if args.launch:
            server = _launch_carla_server(settings["carla_client"], quality=args.quality)
            if server is None:
                raise RuntimeError("Falha ao subir o servidor CARLA.")

            # Boot a frio do CARLA passa facil de 1 min (compila shaders + carrega
            # o mapa). Em vez de chutar um tempo fixo, damos um head start curto e
            # deixamos a conexao MUITO paciente: ela tenta ate o RPC do servidor subir.
            cc = dict(settings["carla_client"])
            cc["max_retries"] = max(cc.get("max_retries", 3), 40)
            cc["retry_delay"] = 3
            settings["carla_client"] = cc

            boot = cc.get("server_boot_time", 20)
            logger.info(
                "Servidor subindo; head start de %ds e depois conexao paciente "
                "(ate ~%d tentativas)...", boot, cc["max_retries"],
            )
            time.sleep(boot)

        run_closed_loop(settings, args.seconds, args.target_speed)

    except Exception:
        logger.exception("Erro no harness de malha fechada")
    finally:
        if server is not None:
            logger.info("Encerrando o servidor CARLA.")
            server.terminate()


if __name__ == "__main__":
    main()
