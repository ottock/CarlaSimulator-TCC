"""Tests for the CARLA ``agents`` path shim.

CARLA's ``agents`` package (which provides ``BasicAgent``, our Stage-A expert)
ships inside the simulator under ``PythonAPI/carla`` and is NOT part of the pip
wheel, so it must be added to ``sys.path`` before it can be imported.
"""
import sys

from ai.paths import ensure_carla_agents_on_path


def test_returns_existing_carla_agents_parent():
    root = ensure_carla_agents_on_path()
    assert root.name == "carla"
    assert (root / "agents").is_dir()


def test_inserts_directory_on_sys_path():
    root = ensure_carla_agents_on_path()
    assert str(root) in sys.path


def test_idempotent_no_duplicate():
    root = ensure_carla_agents_on_path()
    ensure_carla_agents_on_path()
    assert sys.path.count(str(root)) == 1
