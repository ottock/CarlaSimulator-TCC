# imports
import time
import argparse
import threading
import subprocess
from pathlib import Path
from datetime import datetime

# project imports
from core.logger.logger import setup_logger
from utils.config import load_settings
from presentation.visualization import LidarVisualization
from core.carlaClient.world_manager import (
    connect_to_carla,
    simulation_context,
    setup_scenario,
    run_simulation,
)

# constants
logger = setup_logger(
    logger_name=__name__,
    log_filename=f"log_{datetime.now().strftime('%Y-%m-%d')}.log"
)


# functions
def launch_carla_server(carla_config: dict) -> subprocess.Popen | None:
    """Launch CARLA server process if executable is available.

    Args:
        carla_config: CARLA connection configuration.

    Returns:
        subprocess.Popen for the server process, or None if launch failed.
    """
    project_root = Path(__file__).resolve().parents[1]
    carla_root = project_root / "CARLA_0.9.16"
    exe_path = carla_root / "CarlaUE4.exe"

    if not exe_path.exists():
        logger.warning(f"CARLA executable not found at {exe_path}, cannot auto-launch server")
        return None

    port = carla_config.get("port", 2000)
    cmd = [str(exe_path), f"-carla-port={port}", "-quality-level=Epic", "-nosound"]

    logger.info(f"Starting CARLA server from {exe_path} on port {port}")
    try:
        process = subprocess.Popen(cmd, cwd=str(carla_root))
        return process
    except Exception:
        logger.exception("Failed to launch CARLA server process")
        return None


def main(settings_path: str) -> None:
    """Run CARLA simulation.

    Args:
        settings_path: Path to settings.json.
    """
    settings = load_settings(settings_path)
    carla_config = settings.get("carla_client", {})
    world_config = settings.get("world", {})

    server_process: subprocess.Popen | None = None
    visualization: LidarVisualization | None = None
    visualization_thread: threading.Thread | None = None

    try:
        server_process = launch_carla_server(carla_config)
        if not server_process:
            raise RuntimeError("Failed to auto-launch CARLA server")

        boot_wait = carla_config.get("server_boot_time", 30)
        logger.info(f"Waiting {boot_wait}s for CARLA server to start")
        time.sleep(boot_wait)

        client = connect_to_carla(carla_config)

        with simulation_context(client, world_config) as (world, actor_list):
            logger.info("Simulation started")

            # --- Modo pista custom (opt-in via settings["world"]["track"]) ---
            # Monta a pista com as pecas do TCC (props tcc_*) e apenas exibe.
            # NAO altera o fluxo padrao: sem "track" habilitado, segue igual
            # (Town01 + ego + trafego + visualizacao).
            track_cfg = world_config.get("track", {})
            if track_cfg.get("enabled"):
                from core.carlaClient.track_builder import build_track
                build_track(world, track_cfg, actor_list)

                # --- Modo PROFESSOR (behavior cloning) ---
                # Se track.professor.enabled: spawna o ego, dirige seguindo os
                # waypoints da linha de centro e grava o dataset (imagem+lidar+comando).
                if track_cfg.get("professor", {}).get("enabled"):
                    from core.carlaClient.professor import run_professor
                    logger.info("Pista montada. Iniciando o PROFESSOR (coleta de dados).")
                    run_professor(world, track_cfg, actor_list)
                    return

                logger.info("Pista custom montada. Modo visualizacao (Ctrl+C para sair).")
                run_simulation(world)
                return

            actor_vehicle, sensors = setup_scenario(client, world, world_config, actor_list)
            if actor_vehicle and sensors:
                logger.info("Starting LIDAR visualization window")
                visualization = LidarVisualization(
                    width=1400,
                    height=820,
                    title="CARLA – LIDAR Obstacle Monitor",
                )

                # LIDAR Sensor setup
                if "lidar" in sensors:
                    lidar_sensor = sensors["lidar"]
                    lidar_sensor.on_data_callback = visualization.update_lidar_data
                    logger.info("LIDAR callback connected to visualization")

                # Camera RGB Sensor setup
                if "camera" in sensors:
                    camera_sensor = sensors["camera"]
                    camera_sensor.on_data_callback = visualization.update_camera_data
                    logger.info("Camera callback connected to visualization")

                # Start visualization in separate thread
                visualization_thread = visualization.start()
                logger.info("Visualization window started")
            else:
                logger.warning("No actor vehicle or sensors available - running without visualization")

            # Run simulation
            run_simulation(world, actor_vehicle, sensors)

    finally:
        if visualization:
            visualization.stop()
            if visualization_thread:
                visualization_thread.join(timeout=5)

        if server_process is not None:
            logger.info("Terminating CARLA server process")
            server_process.terminate()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CARLA Simulator")
    parser.add_argument("settings_json", help="Path to settings.json")
    args = parser.parse_args()

    logger.info("Starting CARLA Simulator")
    try:
        main(args.settings_json)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception:
        logger.exception("Error")
    finally:
        logger.info("CARLA Simulator Finished")