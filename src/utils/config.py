# imports
import json


# helpers
class _Optional:
    """Wrap a schema entry (a type, a tuple of types, or a nested dict) to mark it
    optional: validated when the key is present, skipped when absent."""

    def __init__(self, schema):
        self.schema = schema


# constants
_SCHEMA = {
    "carla_client": {
        "host":        str,
        "port":        int,
        "timeout":     (int, float),
        "max_retries": int,
        "retry_delay": (int, float),
    },
    "world": {
        "map_name": str,
        "seed":     int,
        "actor_vehicle": {
            "enabled":        bool,
            "vehicle_filter": str,
            "spawn_index":    int,
            "use_lidar":      bool,
            "use_camera":     bool,
            "lidar": {
                "channels":           int,
                "points_per_second":  int,
                "rotation_frequency": (int, float),
                "range":              (int, float),
                "upper_fov":          (int, float),
                "lower_fov":          (int, float),
            },
            # Optional: RGB camera intrinsics + mounting pose. When absent, the
            # camera manager falls back to its defaults.
            "camera": _Optional({
                "width":  _Optional(int),
                "height": _Optional(int),
                "fov":    _Optional((int, float)),
                "x":      _Optional((int, float)),
                "y":      _Optional((int, float)),
                "z":      _Optional((int, float)),
                "pitch":  _Optional((int, float)),
                "yaw":    _Optional((int, float)),
                "roll":   _Optional((int, float)),
            }),
        },
        "settings": {
            "synchronous_mode":    bool,
            "fixed_delta_seconds": float,
            "no_rendering_mode":   bool,
            "substepping":         bool,
        },
        "weather": {
            "cloudiness":             (int, float),
            "precipitation":          (int, float),
            "precipitation_deposits": (int, float),
            "wind_intensity":         (int, float),
            "sun_azimuth_angle":      (int, float),
            "sun_altitude_angle":     (int, float),
            "fog_density":            (int, float),
            "fog_distance":           (int, float),
            "wetness":                (int, float),
        },
        "traffic_manager": {
            "port":             int,
            "synchronous_mode": bool,
        },
        "traffic": {
            "vehicle_count":  int,
            "vehicle_filter": str,
        },
        "pedestrians": {
            "count": int,
        },
    },
}


# functions
def _validate(data: dict, schema: dict, path: str = "") -> None:
    """Recursively validate a settings dict against a schema.

    Args:
        data: Dictionary to validate.
        schema: Expected structure; leaf values are types or tuples of types,
                nested dicts indicate a required sub-section.
        path: Dot-separated key path used to build error messages (e.g. world.lidar.range).

    Raises:
        ValueError: If a required key is missing.
        TypeError: If a value's type does not match the schema.
    """
    for key, expected in schema.items():
        current_path = f"{path}.{key}" if path else key
        optional = isinstance(expected, _Optional)
        if optional:
            expected = expected.schema
        if key not in data:
            if optional:
                continue
            raise ValueError(f"Missing required key: '{current_path}'")
        value = data[key]
        if isinstance(expected, dict):
            if not isinstance(value, dict):
                raise TypeError(f"'{current_path}' must be a dict, got {type(value).__name__}")
            _validate(value, expected, current_path)
        else:
            if not isinstance(value, expected):
                expected_names = (
                    " or ".join(t.__name__ for t in expected)
                    if isinstance(expected, tuple)
                    else expected.__name__
                )
                raise TypeError(
                    f"'{current_path}' must be {expected_names}, got {type(value).__name__}"
                )


def load_settings(settings_path: str) -> dict:
    """Load and validate settings from JSON file.

    Args:
        settings_path: Path to settings.json file.

    Returns:
        Dictionary with settings.

    Raises:
        FileNotFoundError: If file not found.
        json.JSONDecodeError: If invalid JSON.
        ValueError: If a required key is missing.
        TypeError: If a key has the wrong type.
    """
    with open(settings_path, "r", encoding="utf-8") as f:
        settings = json.load(f)
    _validate(settings, _SCHEMA)
    return settings
