# imports
import math
import carla
import logging
import numpy as np
from collections import deque
from typing import Callable, Optional

# constants
logger = logging.getLogger(__name__)


# classes
class LidarSensor:
    """LIDAR sensor wrapper for CARLA.

    Captures point cloud data from the LIDAR sensor and provides callbacks for
    processing sensor data.
    """

    def __init__(
        self,
        vehicle: carla.Actor,
        world: carla.World,
        actor_list: list,
        on_data_callback: Optional[Callable] = None,
        max_points: int = 100000,
        lidar_config: Optional[dict] = None,
    ):
        """Initialize LIDAR sensor.

        Args:
            vehicle: Vehicle actor to attach sensor to.
            world: CARLA world.
            actor_list: List to track sensor actors.
            on_data_callback: Optional callback function for sensor data.
            max_points: Maximum number of points to keep in buffer.
            lidar_config: Optional LIDAR configuration dict (channels, points_per_second,
                          rotation_frequency, range, upper_fov, lower_fov).
        """
        self.vehicle = vehicle
        self.world = world
        self.actor_list = actor_list
        self.on_data_callback = on_data_callback
        self.max_points = max_points

        if lidar_config is None:
            lidar_config = {}

        # Point cloud buffer
        self.point_cloud = deque(maxlen=max_points)
        self.latest_data = None

        # Create LIDAR blueprint
        blueprint_library = world.get_blueprint_library()
        lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")

        # Configure LIDAR from config (with sensible defaults)
        lidar_bp.set_attribute("channels", str(lidar_config.get("channels", 32)))
        lidar_bp.set_attribute("points_per_second", str(lidar_config.get("points_per_second", 100000)))
        lidar_bp.set_attribute("rotation_frequency", str(lidar_config.get("rotation_frequency", 10)))
        lidar_bp.set_attribute("range", str(lidar_config.get("range", 100)))
        lidar_bp.set_attribute("upper_fov", str(lidar_config.get("upper_fov", 15)))
        lidar_bp.set_attribute("lower_fov", str(lidar_config.get("lower_fov", -25)))

        # Spawn sensor with offset from vehicle center
        sensor_transform = carla.Transform(
            carla.Location(x=0, y=0, z=2.0),
            carla.Rotation(pitch=0, yaw=0, roll=0)
        )

        self.sensor = world.spawn_actor(lidar_bp, sensor_transform, attach_to=vehicle)
        self.actor_list.append(self.sensor)

        # Listen to sensor data
        self.sensor.listen(self._parse_lidar_data)
        logger.info("LIDAR sensor attached to vehicle")


    def _detect_obstacles_in_points(self, points: np.ndarray) -> tuple:
        """Detect obstacles in the LIDAR point cloud using grid-based clustering.

        Does not rely on the world API. Steps:
          1. Remove ground points and ego-vehicle artifacts.
          2. Project surviving points onto a 2D XY grid.
          3. BFS flood-fill over occupied grid cells to form clusters.
          4. Return a mask of obstacle points and a list of detected objects
             with distance and local position for each cluster.

        Args:
            points: Point cloud in format (N, 3) with X, Y, Z (sensor-local frame).

        Returns:
            Tuple of (obstacle_mask, detected_objects).
        """
        obstacle_mask = np.zeros(len(points), dtype=bool)
        detected_objects = []

        if len(points) < 10:
            return obstacle_mask, detected_objects

        try:
            z = points[:, 2]
            r2 = points[:, 0] ** 2 + points[:, 1] ** 2

            # --- 1. Filter ---
            # Sensor sits at z=2.0 m; ground appears at z ≈ -2.0 in local frame.
            # Keep points above -1.7 m (above road surface) and below 8 m,
            # farther than 2 m from origin (removes ego-vehicle returns).
            keep = (z > -1.7) & (z < 8.0) & (r2 > 4.0)
            valid_idx = np.where(keep)[0]

            if len(valid_idx) < 10:
                return obstacle_mask, detected_objects

            vp = points[valid_idx]  # shape (M, 3)

            # --- 2. Project to XY grid ---
            cell_size = 0.5          # metres per cell
            grid_range = 100         # ± 100 m
            grid_size = int(2 * grid_range / cell_size)  # 400 cells per axis

            gx = np.clip(
                ((vp[:, 0] + grid_range) / cell_size).astype(np.int32), 0, grid_size - 1
            )
            gy = np.clip(
                ((vp[:, 1] + grid_range) / cell_size).astype(np.int32), 0, grid_size - 1
            )
            cell_keys = (gx * grid_size + gy).astype(np.int32)  # 1-D key per point

            # Sort points by cell key so we can slice contiguous runs per cell
            order = np.argsort(cell_keys, kind="stable")
            sorted_keys = cell_keys[order]

            unique_keys, starts, ucounts = np.unique(
                sorted_keys, return_index=True, return_counts=True
            )

            # Keep only cells that have at least 2 points (noise rejection)
            dense_mask = ucounts >= 2
            dense_keys = unique_keys[dense_mask]
            dense_starts = starts[dense_mask]
            dense_counts = ucounts[dense_mask]

            if len(dense_keys) == 0:
                return obstacle_mask, detected_objects

            # Build cell-key → (start_in_order, count) lookup
            key_info = {
                int(k): (int(s), int(c))
                for k, s, c in zip(dense_keys, dense_starts, dense_counts)
            }
            dense_set = set(key_info.keys())

            # --- 3. BFS connected components ---
            def _encode(cx: int, cy: int) -> int:
                return cx * grid_size + cy

            def _decode(k: int):
                return k // grid_size, k % grid_size

            visited: set = set()
            min_cluster_pts = 5

            for seed in dense_set:
                if seed in visited:
                    continue

                stack = [seed]
                cluster_keys: list = []
                cluster_pt_count = 0

                while stack:
                    k = stack.pop()
                    if k in visited:
                        continue
                    visited.add(k)
                    cluster_keys.append(k)
                    cluster_pt_count += key_info[k][1]

                    cx, cy = _decode(k)
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            if dx == 0 and dy == 0:
                                continue
                            nk = _encode(cx + dx, cy + dy)
                            if nk not in visited and nk in dense_set:
                                stack.append(nk)

                if cluster_pt_count < min_cluster_pts:
                    continue

                # --- 4. Gather cluster points and compute centroid ---
                local_idxs: list = []
                for k in cluster_keys:
                    s, c = key_info[k]
                    local_idxs.extend(order[s: s + c].tolist())

                cluster_pts = vp[local_idxs]
                centroid_x = float(np.mean(cluster_pts[:, 0]))
                centroid_y = float(np.mean(cluster_pts[:, 1]))
                distance = math.sqrt(centroid_x ** 2 + centroid_y ** 2)

                # Mark original points as obstacles
                orig_idxs = valid_idx[local_idxs]
                obstacle_mask[orig_idxs] = True

                detected_objects.append({
                    "distance": distance,
                    "local_x": centroid_x,
                    "local_y": centroid_y,
                    "point_count": cluster_pt_count,
                })

            detected_objects.sort(key=lambda o: o["distance"])
            return obstacle_mask, detected_objects

        except Exception as e:
            logger.debug(f"Error detecting obstacles in LIDAR points: {e}")
            return np.zeros(len(points), dtype=bool), []


    def _parse_lidar_data(self, data: carla.LidarMeasurement) -> None:
        """Parse LIDAR measurement data.

        Args:
            data: LIDAR measurement from CARLA.
        """
        try:
            # Convert to numpy array
            points = np.copy(np.frombuffer(data.raw_data, dtype=np.float32))
            points = points.reshape((-1, 4))

            # Extract XYZ coordinates and intensity
            xyz = points[:, :3]
            intensity = points[:, 3]

            # Detect obstacles in the point cloud
            vehicle_mask, detected_objects = self._detect_obstacles_in_points(xyz)

            # Store in circular buffer
            self.point_cloud.extend(xyz)
            self.latest_data = {
                "points": xyz,
                "intensity": intensity,
                "vehicle_mask": vehicle_mask,
                "detected_objects": detected_objects,
                "timestamp": data.timestamp,
                "frame": data.frame,
            }

            # Call user callback if provided
            if self.on_data_callback:
                self.on_data_callback(self.latest_data)

        except Exception as e:
            logger.error(f"Error parsing LIDAR data: {e}")


    def get_latest_point_cloud(self) -> Optional[dict]:
        """Get latest point cloud data.

        Returns:
            Dictionary with point cloud data or None if no data yet.
        """
        return self.latest_data


    def get_buffered_points(self) -> np.ndarray:
        """Get all buffered points.

        Returns:
            Numpy array of all buffered points (N, 3).
        """
        if self.point_cloud:
            return np.array(list(self.point_cloud))
        return np.array([]).reshape(0, 3)
