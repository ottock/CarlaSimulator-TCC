# imports
import carla
import logging
import numpy as np
from typing import Callable, Optional

# constants
logger = logging.getLogger(__name__)


# classes
class CameraRGBSensor:
    """RGB camera sensor wrapper for CARLA.

    Captures RGB images from a camera sensor.
    """

    def __init__(
        self,
        vehicle: carla.Actor,
        world: carla.World,
        actor_list: list,
        on_data_callback: Optional[Callable] = None,
        width: int = 800,
        height: int = 600,
    ):
        """Initialize RGB camera sensor.

        Args:
            vehicle: Vehicle actor to attach sensor to.
            world: CARLA world.
            actor_list: List to track sensor actors.
            on_data_callback: Optional callback function for image data.
            width: Image width.
            height: Image height.
        """
        self.vehicle = vehicle
        self.world = world
        self.actor_list = actor_list
        self.on_data_callback = on_data_callback
        self.width = width
        self.height = height
        self.latest_frame = None

        # Create camera blueprint
        blueprint_library = world.get_blueprint_library()
        camera_bp = blueprint_library.find("sensor.camera.rgb")

        # Configure camera
        camera_bp.set_attribute("image_size_x", str(width))
        camera_bp.set_attribute("image_size_y", str(height))
        camera_bp.set_attribute("fov", "90")

        # Spawn sensor – forward-facing camera at windshield level (real-car mounting)
        sensor_transform = carla.Transform(
            carla.Location(x=1.5, y=0.0, z=1.4),
            carla.Rotation(pitch=-5, yaw=0, roll=0)
        )

        self.sensor = world.spawn_actor(camera_bp, sensor_transform, attach_to=vehicle)
        self.actor_list.append(self.sensor)

        # Listen to sensor data
        self.sensor.listen(self._parse_camera_data)
        logger.info("RGB camera sensor attached to vehicle")


    def _parse_camera_data(self, data: carla.Image) -> None:
        """Parse camera image data.

        Args:
            data: Image data from CARLA.
        """
        try:
            # Convert to numpy array
            array = np.copy(np.frombuffer(data.raw_data, dtype=np.uint8))
            array = array.reshape((data.height, data.width, 4))
            array = array[:, :, :3]  # Remove alpha channel

            self.latest_frame = {
                "image": array,
                "timestamp": data.timestamp,
                "frame": data.frame,
            }

            # Call user callback if provided
            if self.on_data_callback:
                self.on_data_callback(self.latest_frame)

        except Exception as e:
            logger.error(f"Error parsing camera data: {e}")


    def get_latest_frame(self) -> Optional[dict]:
        """Get latest camera frame.

        Returns:
            Dictionary with image data or None if no frame yet.
        """
        return self.latest_frame