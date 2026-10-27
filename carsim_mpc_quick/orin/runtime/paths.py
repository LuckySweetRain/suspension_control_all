"""Resource locations for the Windows CARLA frontend, resolved from this package."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CARLA_ROOT = Path("E:/Desktop/workspace/CARLA_0.9.11/WindowsNoEditor")
DATABASE = Path("E:/Desktop/workspace/CarSim_database/carsim_active_suspension")
CARLA_EGG = CARLA_ROOT / "PythonAPI/carla/dist/carla-0.9.11-py3.7-win-amd64.egg"


def load_carla():
    """Load the installed legacy ABI only when a simulation entry requests it."""
    if str(CARLA_EGG) not in sys.path:
        sys.path.insert(0, str(CARLA_EGG))
    return importlib.import_module("carla")
