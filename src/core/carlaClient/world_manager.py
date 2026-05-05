# imports
import time
import carla
import random
import logging
from typing import Generator, Optional
from contextlib import contextmanager

# project imports
from core.carlaClient.sensor_manager import setup_sensors

# constants
logger = logging.getLogger(__name__)


# context manager
@contextmanager
def simulation_context(
    client: carla.Client, world_config: dict
) -> Generator[tuple[carla.World, list], None, None]:
    """Context manager for simulation lifecycle.

    Args:
        client: CARLA client.
        world_config: World configuration from settings.json.

    Yields:
        Tuple of (world, actor_list).
    """
    # Configure determinism
    seed = world_config.get("seed", 42)
    random.seed(seed)

    # Load desired map (if provided)
    map_name = world_config.get("map_name")
    if map_name:
        logger.info(f"Loading map: {map_name}")
        max_load_retries = 3
        for attempt in range(1, max_load_retries + 1):
            try:
                world = client.load_world(map_name)
                logger.info(f"Successfully loaded map: {map_name}")
                break
            except RuntimeError as e:
                logger.warning(f"Failed to load map attempt {attempt}/{max_load_retries}: {e}")
                if attempt < max_load_retries:
                    logger.info("Retrying in 3 seconds...")
                    time.sleep(3)
                else:
                    raise RuntimeError(f"Failed to load map '{map_name}' after {max_load_retries} attempts: {e}") from e
    else:
        world = client.get_world()

    original_settings = world.get_settings()
    actor_list: list = []

    try:
        _apply_settings(world, world_config)

        # Let world settle with new settings
        if world.get_settings().synchronous_mode:
            world.tick()
            logger.debug("World initialized with first tick")

        logger.info("Simulation initialized")
        yield world, actor_list

    finally:
        world.apply_settings(original_settings)
        _destroy_actors(client, actor_list)
        logger.info("Simulation cleaned up")


def connect_to_carla(carla_config: dict) -> carla.Client:
    """Connect to CARLA server with retry logic.

    Args:
        carla_config: Dict with 'host', 'port', 'timeout', 'max_retries', 'retry_delay'.

    Returns:
        CARLA client instance.

    Raises:
        RuntimeError: If connection fails after max retries.
    """
    host = carla_config.get("host", "localhost")
    port = carla_config.get("port", 2000)
    timeout = carla_config.get("timeout", 10)
    max_retries = carla_config.get("max_retries", 3)
    retry_delay = carla_config.get("retry_delay", 2)

    if not host or not isinstance(port, int) or port <= 0:
        raise ValueError(f"Invalid host/port: {host}:{port}")

    logger.info(f"Attempting to connect to CARLA at {host}:{port}")

    for attempt in range(1, max_retries + 1):
        try:
            client = carla.Client(host, port)
            client.set_timeout(timeout)

            client.get_world()
            logger.info(f"Successfully connected to CARLA at {host}:{port}")
            return client

        except RuntimeError as e:
            logger.warning(f"Connection attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise RuntimeError(
                    f"Failed to connect to CARLA after {max_retries} attempts at {host}:{port}"
                ) from e
        except Exception as e:
            logger.error(f"Unexpected error during connection: {e}")
            raise


def spawn_random_vehicles(world: carla.World, actor_list: list, count: int = 5, traffic_manager: carla.TrafficManager = None) -> list:
    """Spawn random vehicles in the world.

    Args:
        world: CARLA world.
        actor_list: List to track spawned actors.
        count: Number of vehicles to spawn.
        traffic_manager: Optional traffic manager for autopilot.

    Returns:
        List of spawned vehicles.
    """
    try:
        vehicle_bps = world.get_blueprint_library().filter("vehicle.*")
        spawn_points = world.get_map().get_spawn_points()

        if not spawn_points:
            logger.warning("No spawn points available")
            return []

        vehicles = []
        for _ in range(count):
            bp = random.choice(vehicle_bps)
            spawn_point = random.choice(spawn_points)
            vehicle = world.try_spawn_actor(bp, spawn_point)

            if vehicle:
                if traffic_manager:
                    vehicle.set_autopilot(True, traffic_manager.get_port())
                else:
                    vehicle.set_autopilot(True)
                actor_list.append(vehicle)
                vehicles.append(vehicle)
                logger.info(f"Spawned: {vehicle.type_id}")
            else:
                logger.warning("Failed to spawn vehicle")

        # Let vehicles settle before next spawns
        if vehicles and world.get_settings().synchronous_mode:
            time.sleep(0.5)

        logger.info(f"Spawned {len(vehicles)} vehicles successfully")
        return vehicles
    except Exception as e:
        logger.error(f"Error spawning vehicles: {e}")
        raise


def spawn_random_pedestrians(world: carla.World, actor_list: list, count: int = 5) -> list:
    """Spawn random pedestrians in the world.

    Args:
        world: CARLA world.
        actor_list: List to track spawned actors.
        count: Number of pedestrians to spawn.

    Returns:
        List of spawned pedestrians.
    """
    try:
        pedestrian_bps = world.get_blueprint_library().filter("walker.pedestrian.*")
        pedestrians = []

        for _ in range(count):
            bp = random.choice(pedestrian_bps)
            location = world.get_random_location_from_navigation()

            if location is None:
                logger.warning("No valid navigation location")
                continue

            transform = carla.Transform(location)
            pedestrian = world.try_spawn_actor(bp, transform)

            if pedestrian:
                # Create and attach AI controller
                controller_bp = world.get_blueprint_library().find("controller.ai.walker")
                if not controller_bp:
                    logger.warning("Controller blueprint not found")
                    pedestrian.destroy()
                    continue

                controller = world.spawn_actor(controller_bp, carla.Transform(), pedestrian)
                controller.start()

                # Set random destination
                destination = world.get_random_location_from_navigation()
                if destination:
                    controller.go_to_location(destination)

                actor_list.append(pedestrian)
                actor_list.append(controller)
                pedestrians.append(pedestrian)
                logger.info(f"Spawned: {pedestrian.type_id} with AI controller")

                # Brief delay between spawns to avoid overwhelming the world
                if world.get_settings().synchronous_mode:
                    time.sleep(0.1)
            else:
                logger.warning("Failed to spawn pedestrian")

        logger.info(f"Spawned {len(pedestrians)} pedestrians successfully")
        return pedestrians
    except Exception as e:
        logger.error(f"Error spawning pedestrians: {e}")
        raise


def spawn_actor_vehicle(
    world: carla.World,
    actor_list: list,
    actor_config: Optional[dict] = None,
    traffic_manager: Optional[object] = None,
) -> tuple[carla.Vehicle, dict]:
    """Spawn the main actor vehicle with sensors.

    Args:
        world: CARLA world.
        actor_list: List to track spawned actors.
        actor_config: Actor configuration from settings.json.
        traffic_manager: Traffic manager for autopilot.

    Returns:
        Tuple of (vehicle, sensors_dict).
    """
    if actor_config is None:
        actor_config = {}

    try:
        blueprint_library = world.get_blueprint_library()

        # Get vehicle blueprint
        vehicle_filter = actor_config.get("vehicle_filter", "vehicle.tesla.model3")
        vehicle_bps = blueprint_library.filter(vehicle_filter)

        if not vehicle_bps:
            logger.warning(f"No vehicles matching filter '{vehicle_filter}', using random vehicle")
            vehicle_bps = blueprint_library.filter("vehicle.*")

        vehicle_bp = vehicle_bps[0]

        # Get spawn point
        spawn_points = world.get_map().get_spawn_points()
        spawn_index = actor_config.get("spawn_index", 0)
        spawn_point = spawn_points[spawn_index % len(spawn_points)]

        logger.info(f"Spawning actor vehicle: {vehicle_bp.id} at spawn point {spawn_index}")

        # Spawn vehicle
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)

        # Setup sensors
        use_lidar = actor_config.get("use_lidar", True)
        use_camera = actor_config.get("use_camera", True)
        lidar_config = actor_config.get("lidar", {})

        sensors = setup_sensors(
            vehicle,
            world,
            actor_list,
            use_lidar=use_lidar,
            use_camera=use_camera,
            lidar_config=lidar_config,
        )

        # Enable autopilot for actor vehicle (if traffic manager provided)
        if traffic_manager is not None:
            try:
                vehicle.set_autopilot(True, traffic_manager.get_port())
                logger.info(f"Actor vehicle autopilot enabled")
            except Exception as e:
                logger.warning(f"Could not enable autopilot for actor vehicle: {e}")

        logger.info(f"Actor vehicle spawned with ID {vehicle.id}")
        return vehicle, sensors

    except Exception as e:
        logger.error(f"Error spawning actor vehicle: {e}")
        raise


def setup_scenario(client: carla.Client, world: carla.World, world_config: dict, actor_list: list) -> tuple[Optional[carla.Vehicle], dict]:
    """Setup traffic manager and spawn actors based on world config.

    Args:
        client: CARLA client.
        world: CARLA world.
        world_config: World configuration from settings.json.
        actor_list: List to track spawned actors.

    Returns:
        Tuple of (actor_vehicle, sensors_dict).
    """
    # Configure traffic manager FIRST
    tm_cfg = world_config.get("traffic_manager", {})
    traffic_manager = client.get_trafficmanager(tm_cfg.get("port", 8000))
    traffic_manager.set_global_distance_to_leading_vehicle(
        tm_cfg.get("global_distance", 2.0)
    )
    traffic_manager.set_synchronous_mode(tm_cfg.get("synchronous_mode", True))
    traffic_manager.set_hybrid_physics_mode(tm_cfg.get("hybrid_physics_mode", True))
    traffic_manager.set_hybrid_physics_radius(tm_cfg.get("hybrid_physics_radius", 50.0))
    traffic_manager.global_percentage_speed_difference(tm_cfg.get("speed_difference", 0))

    logger.info("Traffic manager configured")

    # Spawn actor vehicle with configured traffic manager
    actor_vehicle = None
    sensors = {}
    actor_cfg = world_config.get("actor_vehicle", {})
    if actor_cfg.get("enabled", True):
        actor_vehicle, sensors = spawn_actor_vehicle(world, actor_list, actor_cfg, traffic_manager)

    # Spawn random vehicles
    traffic_cfg = world_config.get("traffic", {})
    vehicle_count = traffic_cfg.get("vehicle_count", 5)
    auto_lane_change = traffic_cfg.get("auto_lane_change", True)

    vehicles = spawn_random_vehicles(
        world, actor_list, count=vehicle_count, traffic_manager=traffic_manager
    )

    for v in vehicles:
        traffic_manager.auto_lane_change(v, auto_lane_change)

    # Spawn pedestrians
    ped_cfg = world_config.get("pedestrians", {})
    spawn_random_pedestrians(world, actor_list, count=ped_cfg.get("count", 5))

    logger.info("Scenario setup complete")
    return actor_vehicle, sensors


def setup_spectator_follow_vehicle(
    world: carla.World,
    vehicle: carla.Vehicle,
    mode: str = "behind"
) -> dict:
    """Setup spectator to follow a vehicle.

    Args:
        world: CARLA world.
        vehicle: Vehicle to follow.
        mode: Camera mode - 'behind', 'chase', 'orbit', 'top', 'first_person'.

    Returns:
        Dictionary with spectator state.
    """
    try:
        spectator = world.get_spectator()
        if spectator is None:
            logger.warning("Spectator not available")
            return {}

        logger.info(f"Spectator following vehicle in '{mode}' mode")

        # Return state dict to track camera
        return {
            "spectator": spectator,
            "vehicle": vehicle,
            "mode": mode
        }

    except Exception as e:
        logger.error(f"Error setting up spectator: {e}")
        return {}


def update_spectator_position(world: carla.World, spectator_state: dict) -> None:
    """Update spectator position based on followed vehicle.

    Args:
        world: CARLA world.
        spectator_state: Dictionary with spectator state from setup_spectator_follow_vehicle.
    """
    if not spectator_state or "vehicle" not in spectator_state:
        return

    try:
        spectator = spectator_state.get("spectator")
        vehicle = spectator_state.get("vehicle")
        mode = spectator_state.get("mode", "behind")

        if spectator is None or vehicle is None:
            return

        if not vehicle.is_alive:
            logger.warning("Actor vehicle is not alive")
            return

        vehicle_transform = vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_rotation = vehicle_transform.rotation

        # Calculate spectator position based on mode
        if mode == "behind":
            # Third-person behind view
            behind_distance = 8.0
            above_height = 3.0
            forward_vector = vehicle_rotation.get_forward_vector()
            spectator_location = vehicle_location - (forward_vector * behind_distance)
            spectator_location.z += above_height
            pitch = -20
            yaw = vehicle_rotation.yaw

        elif mode == "chase":
            # Close third-person chase cam
            behind_distance = 4.0
            above_height = 2.0
            forward_vector = vehicle_rotation.get_forward_vector()
            spectator_location = vehicle_location - (forward_vector * behind_distance)
            spectator_location.z += above_height
            pitch = -15
            yaw = vehicle_rotation.yaw

        elif mode == "orbit":
            # Orbit around vehicle
            import math
            time_val = time.time() * 0.5
            angle = math.sin(time_val) * 0.5
            forward_vector = vehicle_rotation.get_forward_vector()
            right_vector = vehicle_rotation.get_right_vector()
            spectator_location = vehicle_location + (forward_vector * 5) + (right_vector * math.sin(time_val) * 10)
            spectator_location.z += 5
            pitch = -30
            yaw = vehicle_rotation.yaw + angle * 45

        elif mode == "top":
            # Top-down view
            spectator_location = vehicle_location
            spectator_location.z += 20
            pitch = -90
            yaw = vehicle_rotation.yaw

        elif mode == "first_person":
            # First person from vehicle
            spectator_location = vehicle_location
            spectator_location.z += 1.3
            pitch = 0
            yaw = vehicle_rotation.yaw

        else:
            # Default to behind
            behind_distance = 8.0
            above_height = 3.0
            forward_vector = vehicle_rotation.get_forward_vector()
            spectator_location = vehicle_location - (forward_vector * behind_distance)
            spectator_location.z += above_height
            pitch = -20
            yaw = vehicle_rotation.yaw

        # Set spectator transform
        spectator_transform = carla.Transform(
            spectator_location,
            carla.Rotation(pitch=pitch, yaw=yaw, roll=0)
        )
        spectator.set_transform(spectator_transform)

    except Exception as e:
        logger.debug(f"Error updating spectator position: {e}")


def run_simulation(world: carla.World, actor_vehicle: Optional[carla.Vehicle] = None, sensors: Optional[dict] = None) -> None:
    """Run simulation loop.

    Args:
        world: CARLA world.
        actor_vehicle: Optional main actor vehicle.
        sensors: Optional sensors dictionary.
    """
    logger.info("Simulation running... (Press Ctrl+C to stop)")
    world_settings = world.get_settings()
    fixed_delta = world_settings.fixed_delta_seconds

    # Setup spectator to follow actor vehicle
    spectator_state = {}
    if actor_vehicle:
        spectator_state = setup_spectator_follow_vehicle(world, actor_vehicle, mode="behind")
        logger.info("Spectator camera configured to follow actor vehicle")

    try:
        while True:
            start = time.perf_counter()
            world.tick()

            # Update spectator position to follow actor vehicle
            if spectator_state:
                update_spectator_position(world, spectator_state)

            elapsed = time.perf_counter() - start
            sleep_duration = fixed_delta - elapsed
            if sleep_duration > 0:
                time.sleep(sleep_duration)
    except KeyboardInterrupt:
        logger.info("Simulation stopped by user")


def _apply_settings(world: carla.World, world_config: dict) -> None:
    """Apply world configuration.

    Args:
        world: CARLA world.
        world_config: Configuration dict with 'settings' and 'weather'.
    """
    # Physics settings
    if "settings" in world_config:
        settings = world_config["settings"]
        ws = world.get_settings()

        # Disable sync temporarily
        ws.synchronous_mode = False
        world.apply_settings(ws)

        # Apply all settings
        ws = world.get_settings()
        ws.synchronous_mode = settings.get("synchronous_mode", True)
        ws.fixed_delta_seconds = settings.get("fixed_delta_seconds", 0.05)
        ws.no_rendering_mode = settings.get("no_rendering_mode", False)
        ws.substepping_enabled = settings.get("substepping", False)
        ws.max_substep_delta_time = settings.get("max_substep_delta_time", 0.01)
        ws.max_substeps = settings.get("max_substeps", 10)

        world.apply_settings(ws)
        logger.info("Physics settings applied")

    # Weather settings
    if "weather" in world_config:
        weather = world_config["weather"]
        w = carla.WeatherParameters(
            cloudiness=weather.get("cloudiness", 0.0),
            precipitation=weather.get("precipitation", 0.0),
            precipitation_deposits=weather.get("precipitation_deposits", 0.0),
            wind_intensity=weather.get("wind_intensity", 0.0),
            sun_azimuth_angle=weather.get("sun_azimuth_angle", 0.0),
            sun_altitude_angle=weather.get("sun_altitude_angle", 0.0),
            fog_density=weather.get("fog_density", 0.0),
            fog_distance=weather.get("fog_distance", 0.0),
            fog_falloff=0.0,
            wetness=weather.get("wetness", 0.0),
            scattering_intensity=weather.get("scattering_intensity", 0.0),
            mie_scattering_scale=weather.get("mie_scattering_scale", 0.0),
            rayleigh_scattering_scale=weather.get("rayleigh_scattering_scale", 0.0),
            dust_storm=weather.get("dust_storm", 0.0)
        )
        world.set_weather(w)
        logger.info("Weather settings applied")


def _destroy_actors(client: carla.Client, actor_list: list) -> None:
    """Destroy all actors.

    Args:
        client: CARLA client.
        actor_list: List of actors to destroy.
    """
    if actor_list:
        client.apply_batch([carla.command.DestroyActor(a) for a in actor_list])
        logger.info(f"Destroyed {len(actor_list)} actors")