"""Dimensionless multi-objective CarSim reward; experimental weights are not paper constants."""

from __future__ import annotations

import numpy as np


def validate_reward(config: dict) -> None:
    expected = {
        "accel",
        "pitch",
        "travel",
        "tire",
        "effort",
        "slew",
        "projection",
        "violation",
        "unload",
    }
    if set(config["weights"]) != expected:
        raise ValueError("Reward weights must explicitly specify all v2 objectives")
    if set(config["scales"]) != {"accel_mps2", "pitch_radps2", "travel_mm", "force_N", "delta_N"}:
        raise ValueError("Reward scales must specify all physical units")
    for value in config["scales"].values():
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Reward scales must be finite and positive")
    if any(not np.isfinite(v) or v < 0 for v in config["weights"].values()):
        raise ValueError("Reward weights must be finite and nonnegative")
    if not np.isfinite(config["reward_scale"]) or config["reward_scale"] <= 0:
        raise ValueError("Reward scale must be positive")


def reward_terms(y, origin, raw, executed, previous, supervisor, config):
    s = config["scales"]
    travel = (y[44:48] - origin[44:48]) / s["travel_mm"]
    static_load = np.maximum(abs(origin[52:56]), 1)
    load_ratio = y[52:56] / static_load
    costs = dict(
        accel=(y[40] * 9.80665 / s["accel_mps2"]) ** 2,
        pitch=(y[43] / s["pitch_radps2"]) ** 2,
        travel=np.mean(travel**2),
        tire=np.mean(((y[52:56] - origin[52:56]) / static_load) ** 2),
        effort=np.mean((executed / s["force_N"]) ** 2),
        slew=np.mean(((executed - previous) / s["delta_N"]) ** 2),
        # Mandatory initialization/contact suppression is not treated as a policy mistake.
        projection=0.0 if supervisor else np.mean(((raw - executed) / s["force_N"]) ** 2),
        violation=np.mean(np.maximum(abs(travel) - 1, 0) ** 2),
        unload=np.mean(np.maximum(0.1 - load_ratio, 0) ** 2),
    )
    terms = {k: float(config["reward_scale"] * config["weights"][k] * v) for k, v in costs.items()}
    if not np.isfinite(list(terms.values())).all():
        raise ValueError("Nonfinite reward components")
    return terms
