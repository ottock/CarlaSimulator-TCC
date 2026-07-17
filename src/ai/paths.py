"""Make CARLA's bundled ``agents`` package importable.

The ``agents`` package (BasicAgent, GlobalRoutePlanner, ...) lives inside the
simulator at ``CARLA_0.9.16/PythonAPI/carla/agents`` and is not shipped in the
pip wheel. Adding its parent directory to ``sys.path`` lets us do
``from agents.navigation.basic_agent import BasicAgent``.

Call ``ensure_carla_agents_on_path()`` once, early, before importing anything
from ``agents``.
"""
import sys
from pathlib import Path

# src/ai/paths.py -> parents[2] is the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CARLA_AGENTS_PARENT = _REPO_ROOT / "CARLA_0.9.16" / "PythonAPI" / "carla"


def ensure_carla_agents_on_path():
    """Insert CARLA's ``agents`` parent dir onto ``sys.path`` (idempotent).

    Returns:
        ``Path`` to the directory that contains the ``agents`` package.
    """
    path_str = str(_CARLA_AGENTS_PARENT)
    if _CARLA_AGENTS_PARENT.is_dir() and path_str not in sys.path:
        sys.path.insert(0, path_str)
    return _CARLA_AGENTS_PARENT
