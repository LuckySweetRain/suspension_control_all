"""Shared resource configuration; imports never load CARLA or the solver library."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(os.environ.get("COSIM_CONFIG", ROOT / "configs/cosim.local.json"))
if not CONFIG_PATH.is_absolute():
    CONFIG_PATH = ROOT / CONFIG_PATH


def read_settings():
    settings = dict(
        carla_root="../CARLA_0.9.11/WindowsNoEditor",
        carsim_database="../CarSim_database/carsim_active_suspension",
        carsim_program=str(Path(ROOT.anchor) / "Carsim2020.0_Prog")
        if os.name == "nt"
        else "/opt/carsim_2020.0",
        carsim_library="",
        carsim_template="results/orin/runtime_run.par",
        carla_egg="",
        windows_conda="conda",
        windows_environment="carla0911",
        orin_host="192.168.50.2",
    )
    if CONFIG_PATH.is_file():
        supplied = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
        if not isinstance(supplied, dict) or set(supplied) - set(settings):
            raise ValueError("Unknown or invalid co-simulation configuration fields")
        settings.update(supplied)
    for key, value in settings.items():
        if not isinstance(value, str):
            raise ValueError("Co-simulation configuration must contain string values: " + key)
    return settings


def resolve_resource(value):
    """Resolve relative resources from the repository, never the caller's cwd."""
    path = Path(value).expanduser()
    # The WSL manager can inspect Windows assets but must not load their DLLs.
    if os.name != "nt" and len(value) > 2 and value[1] == ":":
        path = Path("/mnt") / value[0].lower() / value[2:].replace("\\", "/").lstrip("/")
    return (path if path.is_absolute() else ROOT / path).resolve()


SETTINGS = read_settings()
CARLA_ROOT = resolve_resource(SETTINGS["carla_root"])
DATABASE = resolve_resource(SETTINGS["carsim_database"])
CARSIM_PROGRAM = resolve_resource(SETTINGS["carsim_program"])
CARSIM_LIBRARY = (
    resolve_resource(SETTINGS["carsim_library"])
    if SETTINGS["carsim_library"]
    else CARSIM_PROGRAM
    / ("Programs/solvers/carsim_64.dll" if os.name == "nt" else "lib64/libcarsim.so.2020.0")
)
CARSIM_TEMPLATE = resolve_resource(SETTINGS["carsim_template"])
FONT_PATH = resolve_resource(os.environ.get("COSIM_FONT", "C:/Windows/Fonts/msyh.ttc"))
if os.name != "nt" and not FONT_PATH.is_file():
    FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
CARLA_EGG = (
    resolve_resource(SETTINGS["carla_egg"])
    if SETTINGS["carla_egg"]
    else CARLA_ROOT
    / "PythonAPI/carla/dist"
    / ("carla-0.9.11-py3.7-" + ("win-amd64.egg" if os.name == "nt" else "linux-x86_64.egg"))
)


def load_carla():
    """Load the installed legacy ABI only when a simulation entry requests it."""
    if str(CARLA_EGG) not in sys.path:
        sys.path.insert(0, str(CARLA_EGG))
    return importlib.import_module("carla")
