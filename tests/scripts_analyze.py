"""Ponte para importar scripts/analyze_car_log.py nos testes.

O pytest tem pythonpath = ["src"], entao scripts/ nao e importavel diretamente.
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from analyze_car_log import (  # noqa: E402,F401
    blocked_runs, fps_stats, load_run, occlusion_arc, sector_to_deg)
