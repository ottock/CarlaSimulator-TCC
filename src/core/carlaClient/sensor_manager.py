# imports
import carla
import logging
from typing import Callable, Optional

# project imports
from core.carlaClient.lidar_manager import LidarSensor
from core.carlaClient.camera_manager import CameraRGBSensor

# constants
logger = logging.getLogger(__name__)


# functions
def setup_sensors(
    vehicle: carla.Actor,
    world: carla.World,
    actor_list: list,
    use_lidar: bool = True,
    use_camera: bool = True,
    lidar_callback: Optional[Callable] = None,
    camera_callback: Optional[Callable] = None,
    lidar_config: Optional[dict] = None,
    camera_config: Optional[dict] = None,
) -> dict:
    """Setup sensors for a vehicle.

    Args:
        vehicle: Vehicle actor.
        world: CARLA world.
        actor_list: List to track actors.
        use_lidar: Whether to attach LIDAR.
        use_camera: Whether to attach camera.
        lidar_callback: Optional LIDAR callback.
        camera_callback: Optional camera callback.
        lidar_config: Optional LIDAR configuration dict.
        camera_config: Optional camera configuration dict.

    Returns:
        Dictionary with sensor objects.
    """
    sensors = {}

    try:
        if use_lidar:
            sensors["lidar"] = LidarSensor(
                vehicle, world, actor_list, on_data_callback=lidar_callback,
                lidar_config=lidar_config,
            )

        if use_camera:
            sensors["camera"] = CameraRGBSensor(
                vehicle, world, actor_list, on_data_callback=camera_callback,
                camera_config=camera_config,
            )

        logger.info(f"Sensors setup complete: {list(sensors.keys())}")
        return sensors

    except Exception as e:
        logger.error(f"Error setting up sensors: {e}")
        raise