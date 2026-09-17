from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import apply_algorithm_overrides, half_car_dimensions, load_config
from roads.road_profiles import RoadProfileFactory


def _number(value, name: str, errors: list[str], *, positive: bool = True, integer: bool = False):
    try:
        number = float(value)
        valid = not isinstance(value, bool) and math.isfinite(number)
        valid = valid and (number > 0.0 if positive else number >= 0.0)
        valid = valid and (not integer or number.is_integer())
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        qualifier = "positive" if positive else "non-negative"
        errors.append(f"{name} must be a finite {qualifier} {'integer' if integer else 'number'}")
        return None
    return number


def _check_weights(weights, names: list[str], errors: list[str], label: str):
    if weights is None:
        return
    if isinstance(weights, dict):
        if set(weights) != set(names):
            errors.append(f"{label} keys must match scenario names")
        values = list(weights.values())
    elif isinstance(weights, list):
        if len(weights) != len(names):
            errors.append(f"{label} length must match scenario count")
        values = weights
    else:
        errors.append(f"{label} must be a mapping or list")
        return
    numbers = [_number(value, label, errors, positive=False) for value in values]
    if all(value is not None for value in numbers) and not any(numbers):
        errors.append(f"{label} must contain at least one positive weight")


def validate_config_data(config: dict[str, Any], check_roads: bool = True) -> dict:
    """Validate the effective half-car configuration without constructing a policy."""
    errors: list[str] = []
    warnings: list[str] = []

    def section(key: str) -> dict:
        value = config.get(key, {})
        if not isinstance(value, dict):
            errors.append(f"{key} must be a mapping")
            return {}
        return value

    for key in (
        "dt",
        "control_dt",
        "episode_seconds",
        "force_limit",
        "wheelbase",
        "preview",
        "reward",
        "rl",
        "scenarios",
    ):
        if key not in config:
            errors.append(f"missing required config key: {key}")
    engine = str(section("environment").get("engine", "python")).lower()
    if engine != "python":
        errors.append("half-car training requires environment.engine=python")
    dt = _number(config.get("dt"), "dt", errors)
    control_dt = _number(config.get("control_dt"), "control_dt", errors)
    duration = _number(config.get("episode_seconds"), "episode_seconds", errors)
    for key, default in (
        ("force_limit", None),
        ("wheelbase", None),
        ("road_scale", 0.1),
        ("speed", 20.0),
    ):
        _number(config.get(key, default), key, errors)
    if dt is not None and control_dt is not None:
        ratio = control_dt / dt
        if (
            not math.isfinite(ratio)
            or ratio < 1
            or not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-6)
        ):
            errors.append("control_dt must be an integer multiple of dt")
    if duration is not None and control_dt is not None and duration < control_dt:
        errors.append("episode_seconds must span at least one control step")

    preview = section("preview")
    steps = _number(preview.get("steps"), "preview.steps", errors, integer=True)
    _number(preview.get("dt", control_dt), "preview.dt", errors)
    if steps is not None and steps > 512:
        errors.append(
            "preview.steps must not exceed the Transformer positional encoding limit (512)"
        )
    if section("preview_error").get("enabled", False):
        errors.append("HalfCarEnv does not implement preview_error")
    rl = section("rl")
    obs_dim_base = 14 + (2 if config.get("action_in_observation", False) else 0)
    try:
        _, _, obs_dim_base = half_car_dimensions(config)
    except (TypeError, ValueError, AttributeError) as exc:
        errors.append(f"invalid observation configuration: {exc}")
    for key in (
        "batch_size",
        "buffer_size",
        "update_every",
        "policy_delay",
        "save_every",
        "episodes",
        "preview_embed_dim",
        "transformer_layers",
        "transformer_heads",
        "transformer_ff_dim",
    ):
        if key in rl:
            _number(rl[key], f"rl.{key}", errors, integer=True)
    for key in ("warmup_steps", "update_after", "eval_every"):
        if key in rl:
            _number(rl[key], f"rl.{key}", errors, integer=True, positive=False)
    for key in ("actor_lr", "critic_lr", "gradient_clip"):
        _number(rl.get(key), f"rl.{key}", errors)
    for key in ("gamma", "tau"):
        value = _number(rl.get(key), f"rl.{key}", errors, positive=False)
        if value is not None and value > 1:
            errors.append(f"rl.{key} must be at most 1")
    try:
        embed_dim, heads = int(rl["preview_embed_dim"]), int(rl["transformer_heads"])
        if heads > 0 and embed_dim % heads:
            errors.append("rl.preview_embed_dim must be divisible by rl.transformer_heads")
    except (KeyError, TypeError, ValueError, OverflowError):
        errors.append("rl.preview_embed_dim and rl.transformer_heads must be integers")
    ppo = rl.get("ppo", {})
    if isinstance(ppo, dict):
        for key in ("train_epochs", "minibatch_size"):
            if key in ppo:
                _number(ppo[key], f"rl.ppo.{key}", errors, integer=True)
    else:
        errors.append("rl.ppo must be a mapping")
    safety = section("safety_limits")
    _number(safety.get("max_suspension_travel"), "safety_limits.max_suspension_travel", errors)
    reward = section("reward")
    for key in ("body_acc", "pitch_acc", "suspension", "tire_load", "control"):
        _number(reward.get(key), f"reward.{key}", errors, positive=False)

    scenarios = config.get("scenarios", [])
    if (
        not isinstance(scenarios, list)
        or not scenarios
        or not all(isinstance(item, dict) for item in scenarios)
    ):
        errors.append("scenarios must be a non-empty list of mappings")
        scenarios = []
    names = [str(item.get("name", "")) for item in scenarios]
    if any(not name.strip() for name in names) or len(set(names)) != len(names):
        errors.append("scenario names must be non-empty and unique")
    sampling = section("scenario_sampling")
    mode = str(sampling.get("mode", "cycle")).lower()
    if mode not in {"cycle", "uniform", "weighted"}:
        errors.append("scenario_sampling.mode must be cycle, uniform, or weighted")
    if mode == "weighted":
        _check_weights(sampling.get("weights"), names, errors, "scenario_sampling.weights")
    curriculum = sampling.get("curriculum", [])
    if not isinstance(curriculum, list) or not all(isinstance(phase, dict) for phase in curriculum):
        errors.append("scenario_sampling.curriculum must be a list of mappings")
        curriculum = []
    previous_until = 0
    for phase in curriculum:
        phase_names = phase.get("scenarios", names)
        if not isinstance(phase_names, list) or not all(
            isinstance(name, str) for name in phase_names
        ):
            errors.append("curriculum.scenarios must be a list of scenario names")
            continue
        phase_names = phase_names or names
        if not set(phase_names) <= set(names):
            errors.append("curriculum references unknown scenarios")
        phase_mode = str(phase.get("mode", mode)).lower()
        if phase_mode not in {"cycle", "uniform", "weighted"}:
            errors.append("curriculum.mode must be cycle, uniform, or weighted")
        if phase_mode == "weighted":
            weights = phase.get("weights", sampling.get("weights"))
            if isinstance(weights, dict) and "weights" not in phase:
                weights = {name: weights.get(name, 0.0) for name in phase_names}
            _check_weights(weights, phase_names, errors, "curriculum.weights")
        if "until_episode" in phase:
            until = _number(
                phase["until_episode"], "curriculum.until_episode", errors, integer=True
            )
            if until is not None:
                if until <= previous_until:
                    errors.append("curriculum.until_episode values must increase")
                previous_until = until

    road_results = []
    if check_roads and dt is not None and duration is not None:
        for scenario in scenarios:
            item = {
                "name": scenario.get("name", ""),
                "type": scenario.get("type", "iso"),
                "ok": True,
                "error": None,
            }
            try:
                # Match HalfCarEnv and repository-relative road paths.
                road = RoadProfileFactory.create(
                    {"speed": config.get("speed", 20.0), **scenario}, duration, dt
                )
                if not np.all(np.isfinite(road.values(np.linspace(0, duration, 64)))):
                    raise ValueError("road profile contains non-finite heights")
            except Exception as exc:  # Report malformed road data as validation errors.
                item.update(ok=False, error=str(exc))
                errors.append(f"scenario {scenario.get('name', '')} road profile invalid: {exc}")
            road_results.append(item)
    if section("imitation").get("enabled", False):
        dataset = Path(str(section("imitation").get("dataset", "")))
        if not dataset.is_file():
            warnings.append(
                f"imitation dataset missing: {dataset}; run scripts/collect_expert_dataset.py first"
            )
    if section("mpc_parameter_scheduling"):
        warnings.append(
            "mpc_parameter_scheduling is not implemented by the half-car policy adapter"
        )
    return {
        "engine": engine,
        "scenario_count": len(scenarios),
        "expected_act_dim": 2,
        "expected_preview_token_dim": 2,
        "expected_obs_dim_base": obs_dim_base,
        "errors": errors,
        "warnings": warnings,
        "roads": road_results,
        "passed": not errors,
    }


def validate_config(
    config_path: Path, check_roads: bool = True, algorithm: str | None = None
) -> dict:
    config_path = Path(config_path)
    try:
        config = load_config(config_path)
        if algorithm is not None:
            config = apply_algorithm_overrides(config, algorithm)
        report = validate_config_data(config, check_roads=check_roads)
    except (OSError, ValueError, TypeError, AttributeError, yaml.YAMLError) as exc:
        report = {"errors": [str(exc)], "warnings": [], "roads": [], "passed": False}
    return {"config": str(config_path.resolve()), "algorithm": algorithm, **report}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--algorithm", choices=["ddpg", "td3", "sac", "ppo"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--skip-road-check", action="store_true")
    args = parser.parse_args()
    report = validate_config(
        ROOT / args.config, check_roads=not args.skip_road_check, algorithm=args.algorithm
    )
    text = json.dumps(report, indent=2)
    if args.out:
        out_path = ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
