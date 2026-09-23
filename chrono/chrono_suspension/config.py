"""Independent configuration; all quantities use SI units."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parent

# The geometry/inertia originate from Chrono Sedan. Wheel rates below are
# explicit engineering assumptions until replaced by the sourced preset.
DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "physics": {"dt": 0.001, "control_dt": 0.01, "episode_seconds": 10.0},
    "vehicle": {
        "sprung_mass_kg": 1515.0,
        "inertia_kgm2": [222.8, 944.1, 1053.5],
        "front_distance_m": 1.388,
        "rear_distance_m": 1.388,
        "front_half_track_m": 0.7979,
        "rear_half_track_m": 0.8,
        "unsprung_mass_kg": [45.0, 45.0, 45.0, 45.0],
        "tire_radius_m": 0.3266,
        "body_height_m": 0.72,
        "suspension_stiffness_npm": [30000.0, 30000.0, 32000.0, 32000.0],
        "suspension_damping_nspm": [2200.0, 2200.0, 2400.0, 2400.0],
        "tire_stiffness_npm": [456715.5612] * 4,
        "tire_damping_nspm": [2291.7742] * 4,
        "spring_cubic_npm3": 0.0,
        "rebound_damping_ratio": 1.0,
        "compression_damping_ratio": 1.0,
        "bump_stop_start_m": 0.065,
        "travel_limit_m": 0.09,
        "bump_stop_npm2": 2e6,
    },
    "actuator": {
        "force_limit_n": 3500.0,
        "time_constant_s": 0.02,
        "delay_s": 0.01,
        "rate_limit_nps": 50000.0,
        "power_limit_w": 5000.0,
    },
    "road": {
        "kind": "random",
        "speed_mps": 15.0,
        "height_m": 0.03,
        "length_m": 0.8,
        "start_m": 9.0,
        "wavelength_m": 5.0,
        "roughness_m": 0.008,
        "left_right_correlation": 0.6,
    },
    "observation": {
        "history_steps": 10,
        "acceleration_noise_mps2": 0.05,
        "gyro_noise_radps": 0.002,
        "attitude_noise_rad": 0.001,
        "travel_noise_m": 0.0005,
        "sensor_delay_steps": 1,
        "preview_steps": 0,
        "preview_dt": 0.01,
    },
    "reward": {
        "acceleration": 1.0,
        "attitude": 0.1,
        "travel": 0.2,
        "tire_load": 0.1,
        "force": 0.01,
        "power": 0.01,
        "action_rate": 0.01,
        "unsafe": 10.0,
    },
    "training": {"n_envs": 4},
}


def merge_config(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def validate_config(cfg: dict) -> None:
    p, v, a = cfg["physics"], cfg["vehicle"], cfg["actuator"]
    for name in ("dt", "control_dt", "episode_seconds"):
        if not np.isfinite(p[name]) or p[name] <= 0:
            raise ValueError(f"physics.{name} must be finite and positive")
    for ratio in (p["control_dt"] / p["dt"], p["episode_seconds"] / p["control_dt"]):
        if ratio < 1 or not np.isclose(ratio, round(ratio), atol=1e-8, rtol=0):
            raise ValueError("Control and episode times must be integer multiples of their steps")
    for key in (
        "unsprung_mass_kg",
        "suspension_stiffness_npm",
        "suspension_damping_nspm",
        "tire_stiffness_npm",
        "tire_damping_nspm",
    ):
        values = np.asarray(v[key], dtype=float)
        if values.shape != (4,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"vehicle.{key} requires four finite positive values (FL FR RL RR)")
    for key in (
        "sprung_mass_kg",
        "front_distance_m",
        "rear_distance_m",
        "front_half_track_m",
        "rear_half_track_m",
        "tire_radius_m",
        "body_height_m",
        "travel_limit_m",
    ):
        if not np.isfinite(v[key]) or v[key] <= 0:
            raise ValueError(f"vehicle.{key} must be finite and positive")
    inertia = np.asarray(v["inertia_kgm2"], dtype=float)
    if inertia.shape != (3,) or np.any(inertia <= 0) or not np.all(np.isfinite(inertia)):
        raise ValueError("vehicle.inertia_kgm2 requires three finite positive values")
    if not 0 < v["bump_stop_start_m"] < v["travel_limit_m"]:
        raise ValueError("Bump stops must engage before the travel limit")
    for key in (
        "spring_cubic_npm3",
        "bump_stop_npm2",
        "rebound_damping_ratio",
        "compression_damping_ratio",
    ):
        if not np.isfinite(v[key]) or v[key] < 0:
            raise ValueError(f"vehicle.{key} must be finite and nonnegative")
    total_distance = v["front_distance_m"] + v["rear_distance_m"]
    preload = (
        v["sprung_mass_kg"]
        * 9.81
        / (2 * total_distance)
        * np.array([v["rear_distance_m"]] * 2 + [v["front_distance_m"]] * 2)
    )
    wheel_height = v["tire_radius_m"] - (
        preload + np.asarray(v["unsprung_mass_kg"]) * 9.81
    ) / np.asarray(v["tire_stiffness_npm"])
    if np.any(wheel_height <= 0) or np.any(v["body_height_m"] - 0.1 <= wheel_height):
        raise ValueError("Static tire compression and suspension mounting heights are invalid")
    if np.any(v["body_height_m"] - 0.1 - wheel_height <= v["travel_limit_m"]):
        raise ValueError("Static suspension length must exceed the permitted compression travel")
    for key in ("force_limit_n", "time_constant_s", "rate_limit_nps", "power_limit_w"):
        if not np.isfinite(a[key]) or a[key] <= 0:
            raise ValueError(f"actuator.{key} must be finite and positive")
    delay = a["delay_s"] / p["dt"]
    if not np.isfinite(delay) or delay < 0 or not np.isclose(delay, round(delay), atol=1e-8):
        raise ValueError("Actuator delay must be a nonnegative integer multiple of physics.dt")
    o = cfg["observation"]
    for key, minimum in (("history_steps", 1), ("sensor_delay_steps", 0), ("preview_steps", 0)):
        if not isinstance(o[key], int) or o[key] < minimum:
            raise ValueError(f"observation.{key} must be an integer >= {minimum}")
    for key in (
        "acceleration_noise_mps2",
        "gyro_noise_radps",
        "attitude_noise_rad",
        "travel_noise_m",
    ):
        if not np.isfinite(o[key]) or o[key] < 0:
            raise ValueError(f"observation.{key} must be finite and nonnegative")
    if not np.isfinite(o["preview_dt"]) or o["preview_dt"] <= 0:
        raise ValueError("observation.preview_dt must be finite and positive")
    for key, weight in cfg["reward"].items():
        if not np.isfinite(weight) or weight < 0:
            raise ValueError(f"reward.{key} must be finite and nonnegative")
    road = cfg["road"]
    if not isinstance(road, dict):
        raise ValueError("road must be a configuration mapping")
    if "scenarios" in road:
        scenarios = road["scenarios"]
        if not isinstance(scenarios, list) or not scenarios:
            raise ValueError("road.scenarios must be a nonempty list")
        if any(not isinstance(item, dict) or "kind" not in item for item in scenarios):
            raise ValueError("Every training road scenario must be a mapping with a kind")


def load_config(path: str | Path | dict | None = None) -> dict:
    cfg = deepcopy(DEFAULT_CONFIG)
    default_path = PROJECT / "configs" / "sedan_reference.yaml"
    if default_path.exists():
        with default_path.open(encoding="utf-8") as stream:
            cfg = merge_config(cfg, yaml.safe_load(stream) or {})
    if isinstance(path, dict):
        cfg = merge_config(cfg, path)
    elif path is not None:
        with Path(path).open(encoding="utf-8") as stream:
            cfg = merge_config(cfg, yaml.safe_load(stream) or {})
    validate_config(cfg)
    return cfg
