"""Ponte para importar scripts/replay_car_log.py nos testes."""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from replay_car_log import attribution  # noqa: E402,F401
