# imports
import glob
import json
import logging
import logging.config
from pathlib import Path

# constants
PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOG_DIR = PROJECT_ROOT / "log"
CONFIG_FILE = Path(__file__).resolve().parent / "configs" / "base_config.json"


# functions
_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def validate_configuration(cfg: dict) -> None:
    """Validate the logging dictConfig format.

    Args:
        cfg: Configuration dictionary to validate.

    Raises:
        ValueError: If the configuration is invalid.
    """
    if cfg.get("version") != 1:
        raise ValueError("'version' must be 1")

    if not isinstance(cfg.get("disable_existing_loggers", False), bool):
        raise ValueError("'disable_existing_loggers' must be a boolean")

    formatters = cfg.get("formatters", {})
    if not isinstance(formatters, dict):
        raise ValueError("'formatters' must be a dict")

    handlers = cfg.get("handlers")
    if not isinstance(handlers, dict) or not handlers:
        raise ValueError("'handlers' must be a non-empty dict")

    if "file" not in handlers:
        raise ValueError("'handlers' must contain a 'file' handler")

    for name, handler in handlers.items():
        if not isinstance(handler, dict):
            raise ValueError(f"handler '{name}' must be a dict")
        if "class" not in handler:
            raise ValueError(f"handler '{name}' is missing required key 'class'")
        level = handler.get("level")
        if level is not None and level not in _VALID_LEVELS:
            raise ValueError(f"handler '{name}' has invalid level '{level}'")
        formatter_ref = handler.get("formatter")
        if formatter_ref is not None and formatter_ref not in formatters:
            raise ValueError(
                f"handler '{name}' references unknown formatter '{formatter_ref}'"
            )

    root = cfg.get("root", {})
    if not isinstance(root, dict):
        raise ValueError("'root' must be a dict")

    root_level = root.get("level")
    if root_level is not None and root_level not in _VALID_LEVELS:
        raise ValueError(f"'root' has invalid level '{root_level}'")

    root_handlers = root.get("handlers", [])
    if not isinstance(root_handlers, list):
        raise ValueError("'root.handlers' must be a list")

    for ref in root_handlers:
        if ref not in handlers:
            raise ValueError(f"'root.handlers' references unknown handler '{ref}'")


def retrieve_configuration() -> dict:
    """Load and validate logger configuration from JSON file.

    Returns:
        Dictionary with logging configuration.

    Raises:
        ValueError: If the configuration format is invalid.
    """
    with CONFIG_FILE.open(encoding="utf-8") as file:
        cfg = json.load(file)
    validate_configuration(cfg)
    return cfg


def setup_logger(
    log_filename: str = "app.log",
    logger_name: str | None = None,
) -> logging.Logger:
    """Configure and return a logger instance.

    Args:
        log_filename: Name of the log file.
        logger_name: Logger name; returns root logger if None.

    Returns:
        Configured Logger instance.
    """
    cfg = retrieve_configuration()

    log_path = LOG_DIR / log_filename
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    _remove_existing_logs(log_path)
    cfg["handlers"]["file"]["filename"] = str(log_path)

    logging.config.dictConfig(cfg)
    if logger_name:
        return logging.getLogger(logger_name)

    return logging.getLogger()


def _remove_existing_logs(log_file: Path) -> None:
    """Remove existing log file and any rotated variants.

    Args:
        log_file: Path to the log file to remove.
    """
    if log_file.exists():
        log_file.unlink()

    for rotated in glob.glob(f"{log_file}.*"):
        try:
            Path(rotated).unlink()
        except OSError:
            raise