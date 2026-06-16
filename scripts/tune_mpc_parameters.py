from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from experiments.evaluation import compute_metrics, make_controller, make_env, rollout


GAIN_KEYS = ["k_z", "k_dz", "k_pitch", "k_dpitch", "k_roll", "k_droll", "k_preview"]


def _score(metrics: dict, weights: dict | None = None) -> float:
    weights = weights or {}
    return float(
        float(weights.get("body_acc", 1.0)) * metrics.get("BodyAccRMS_mps2", 0.0)
        + float(weights.get("pitch_acc", 0.2)) * metrics.get("PitchAccRMS_radps2", 0.0)
        + float(weights.get("roll_acc", 0.2)) * metrics.get("RollAccRMS_radps2", 0.0)
        + float(weights.get("suspension", 8.0)) * (
            metrics.get("MaxFrontSusp_m", 0.0)
            + metrics.get("MaxRearSusp_m", 0.0)
            + metrics.get("MaxSusp_fl_m", 0.0)
            + metrics.get("MaxSusp_fr_m", 0.0)
            + metrics.get("MaxSusp_rl_m", 0.0)
            + metrics.get("MaxSusp_rr_m", 0.0)
        )
        + float(weights.get("unsafe", 100.0)) * metrics.get("UnsafeSteps", 0.0)
    )


def _sample_gains(base: dict, rng: np.random.Generator, scale_low: float, scale_high: float) -> dict:
    out = {}
    for key in GAIN_KEYS:
        base_value = float(base.get(key, 0.0))
        scale = float(rng.uniform(scale_low, scale_high))
        out[key] = base_value * scale
    return out


def _evaluate_gain_config(config: dict, controller_name: str, scenario_limit: int | None, score_weights: dict | None) -> tuple[float, dict]:
    scenario_metrics = {}
    scores = []
    scenarios = list(config["scenarios"])
    if scenario_limit is not None:
        scenarios = scenarios[:scenario_limit]
    for scenario in scenarios:
        env = make_env(config, scenario=scenario, use_preview=True)
        try:
            controller = make_controller(controller_name, env, config)
            run = rollout(env, controller, use_preview=True)
            metrics = compute_metrics(run["data"], float(config.get("settle_seconds", 1.0)))
        finally:
            env.close()
        scenario_score = _score(metrics, score_weights)
        scores.append(scenario_score)
        scenario_metrics[scenario["name"]] = {"score": scenario_score, **metrics}
    return float(np.mean(scores)), scenario_metrics


def tune_mpc_parameters(
    config_path: Path,
    out_dir: Path,
    trials: int = 16,
    seed: int = 42,
    scenario_limit: int | None = None,
    controller_name: str = "FULL_CAR_MPC_LITE",
    scale_low: float = 0.5,
    scale_high: float = 1.5,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_config = load_config(config_path)
    base_gains = dict(base_config.get("reduced_full_car_mpc", {}))
    rng = np.random.default_rng(seed)
    rows = []
    best = {"score": float("inf"), "gains": None, "scenario_metrics": None}

    for trial in range(trials):
        gains = copy.deepcopy(base_gains) if trial == 0 else _sample_gains(base_gains, rng, scale_low, scale_high)
        trial_config = copy.deepcopy(base_config)
        trial_config["reduced_full_car_mpc"] = {**base_gains, **gains}
        score, scenario_metrics = _evaluate_gain_config(
            trial_config,
            controller_name=controller_name,
            scenario_limit=scenario_limit,
            score_weights=base_config.get("mpc_parameter_tuning", {}).get("score_weights"),
        )
        row = {"trial": trial, "score": score, **{key: float(trial_config["reduced_full_car_mpc"][key]) for key in GAIN_KEYS}}
        rows.append(row)
        if score < best["score"]:
            best = {"score": score, "gains": trial_config["reduced_full_car_mpc"], "scenario_metrics": scenario_metrics}

    trials_df = pd.DataFrame(rows).sort_values("score")
    trials_path = out_dir / "mpc_parameter_trials.csv"
    trials_df.to_csv(trials_path, index=False)
    best_config = copy.deepcopy(base_config)
    best_config["reduced_full_car_mpc"] = best["gains"]
    best_config_path = out_dir / "best_mpc_config.yaml"
    with best_config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(best_config, f, sort_keys=False, allow_unicode=True)
    manifest = {
        "config": str(config_path.resolve()),
        "controller": controller_name,
        "trials": int(trials),
        "scenario_limit": scenario_limit,
        "scale_low": float(scale_low),
        "scale_high": float(scale_high),
        "best_score": float(best["score"]),
        "best_gains": best["gains"],
        "best_scenario_metrics": best["scenario_metrics"],
        "trials_csv": str(trials_path),
        "best_config": str(best_config_path),
    }
    manifest_path = out_dir / "mpc_parameter_tuning_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    manifest["manifest"] = str(manifest_path)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mujoco_full_car_corner.yaml")
    parser.add_argument("--out", default="results/mpc_parameter_tuning")
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario-limit", type=int, default=None)
    parser.add_argument("--controller", default="FULL_CAR_MPC_LITE")
    parser.add_argument("--scale-low", type=float, default=0.5)
    parser.add_argument("--scale-high", type=float, default=1.5)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    manifest = tune_mpc_parameters(
        config_path=config_path,
        out_dir=out_dir,
        trials=args.trials,
        seed=args.seed,
        scenario_limit=args.scenario_limit,
        controller_name=args.controller,
        scale_low=args.scale_low,
        scale_high=args.scale_high,
    )
    print(f"Best score {manifest['best_score']:.6g}; config: {manifest['best_config']}")


if __name__ == "__main__":
    main()
