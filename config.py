from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load parent-relative YAML inheritance, rejecting cycles and invalid roots."""
    current = Path(path).resolve()
    paths: list[Path] = []
    layers: list[dict[str, Any]] = []
    while True:
        if current in paths:
            chain = " -> ".join(str(item) for item in [*paths, current])
            raise ValueError(f"Config inheritance cycle: {chain}")
        paths.append(current)
        with current.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError(f"Config must contain a YAML mapping: {current}")
        parent_name = data.pop("inherits", None)
        layers.append(data)
        if parent_name is None:
            break
        if not isinstance(parent_name, str) or not parent_name.strip():
            raise ValueError(f"inherits must be a non-empty file path: {current}")
        current = (current.parent / parent_name).resolve()
    result: dict[str, Any] = {}
    for layer in reversed(layers):
        result = deep_update(result, layer)
    return result


def half_car_dimensions(config: dict[str, Any]) -> tuple[int, int, int]:
    """Return action, preview-token, and base-observation dimensions for HalfCarEnv."""
    observation = config.get("observation", {})
    if str(observation.get("mode", "privileged")).lower() != "privileged":
        raise ValueError("HalfCarEnv only implements observation.mode=privileged")
    if observation.get("history", {}).get("enabled", False):
        raise ValueError("HalfCarEnv does not implement observation.history")
    if float(observation.get("state_noise_std", 0.0)) != 0.0:
        raise ValueError("HalfCarEnv does not implement observation.state_noise_std")
    base_dim = 14 + (2 if config.get("action_in_observation", False) else 0)
    rl = config.get("rl", {})
    for key, expected in (("act_dim", 2), ("preview_token_dim", 2), ("obs_dim_base", base_dim)):
        if key in rl and rl[key] != expected:
            raise ValueError(f"rl.{key} must be {expected} for HalfCarEnv")
    return 2, 2, base_dim


def apply_algorithm_overrides(config: dict[str, Any], algorithm: str) -> dict[str, Any]:
    result = deepcopy(config)
    overrides = result.get("rl", {}).get("algorithm_overrides", {}).get(algorithm.lower(), {})
    if overrides:
        result["rl"] = deep_update(result["rl"], overrides)
    return result


def as_namespace(data: dict[str, Any]):
    class Namespace:
        pass

    ns = Namespace()
    for key, value in data.items():
        if isinstance(value, dict):
            value = as_namespace(value)
        setattr(ns, key, value)
    return ns
