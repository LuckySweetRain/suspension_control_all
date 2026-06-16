from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.mpc_parameter_env import MPCParameterPolicyEnv


def train_mpc_parameter_policy(
    config_path: Path,
    out_dir: Path,
    iterations: int = 8,
    population: int = 16,
    elite_fraction: float = 0.25,
    seed: int = 42,
    scenario_limit: int | None = None,
    scale_low: float = 0.5,
    scale_high: float = 1.5,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = MPCParameterPolicyEnv(
        config_path=config_path,
        scenario_limit=scenario_limit,
        scale_low=scale_low,
        scale_high=scale_high,
    )
    rng = np.random.default_rng(seed)
    mean = np.zeros(env.action_space.shape, dtype=np.float64)
    std = np.ones(env.action_space.shape, dtype=np.float64) * 0.75
    elite_count = max(1, int(round(population * elite_fraction)))
    rows = []
    best = {"reward": -float("inf"), "score": float("inf"), "action": None, "gains": None, "metrics": None}

    try:
        for iteration in range(iterations):
            samples = np.clip(rng.normal(mean, std, size=(population, env.gain_dim)), -1.0, 1.0)
            rewards = []
            for member, action in enumerate(samples):
                env.reset(seed=seed + iteration * population + member)
                _, reward, _, _, info = env.step(action)
                score = -float(reward)
                rewards.append(float(reward))
                rows.append(
                    {
                        "iteration": iteration,
                        "member": member,
                        "scenario": info.get("active_scenario", ""),
                        "reward": float(reward),
                        "score": score,
                        **{f"action_{key}": float(value) for key, value in zip(info["gain_keys"], action)},
                        **{key: float(info["last_gains"][key]) for key in info["gain_keys"]},
                    }
                )
                if reward > best["reward"]:
                    best = {
                        "reward": float(reward),
                        "score": score,
                        "action": [float(v) for v in action],
                        "gains": info["last_gains"],
                        "metrics": info["last_metrics"],
                    }
            elite_idx = np.argsort(rewards)[-elite_count:]
            elite = samples[elite_idx]
            mean = np.mean(elite, axis=0)
            std = np.maximum(np.std(elite, axis=0), 0.05)
    finally:
        env.close()

    history_path = out_dir / "mpc_parameter_policy_history.csv"
    pd.DataFrame(rows).to_csv(history_path, index=False)
    best_config = env.config.copy()
    best_config["reduced_full_car_mpc"] = best["gains"]
    best_config_path = out_dir / "best_mpc_parameter_policy_config.yaml"
    with best_config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(best_config, f, sort_keys=False, allow_unicode=True)
    manifest = {
        "config": str(config_path.resolve()),
        "iterations": int(iterations),
        "population": int(population),
        "elite_fraction": float(elite_fraction),
        "scenario_limit": scenario_limit,
        "scale_low": float(scale_low),
        "scale_high": float(scale_high),
        "best_reward": float(best["reward"]),
        "best_score": float(best["score"]),
        "best_action": best["action"],
        "best_gains": best["gains"],
        "best_metrics": best["metrics"],
        "final_mean": [float(v) for v in mean],
        "final_std": [float(v) for v in std],
        "history_csv": str(history_path),
        "best_config": str(best_config_path),
        "method": "cem_parameter_policy",
    }
    manifest_path = out_dir / "mpc_parameter_policy_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    manifest["manifest"] = str(manifest_path)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mujoco_full_car_corner.yaml")
    parser.add_argument("--out", default="results/mpc_parameter_policy")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--elite-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario-limit", type=int, default=None)
    parser.add_argument("--scale-low", type=float, default=0.5)
    parser.add_argument("--scale-high", type=float, default=1.5)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    manifest = train_mpc_parameter_policy(
        config_path=config_path,
        out_dir=out_dir,
        iterations=args.iterations,
        population=args.population,
        elite_fraction=args.elite_fraction,
        seed=args.seed,
        scenario_limit=args.scenario_limit,
        scale_low=args.scale_low,
        scale_high=args.scale_high,
    )
    print(f"Best score {manifest['best_score']:.6g}; config: {manifest['best_config']}")


if __name__ == "__main__":
    main()
