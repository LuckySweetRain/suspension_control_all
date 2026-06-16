from __future__ import annotations

import argparse
import json
import sys
import copy
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from scripts.train_mpc_parameter_neural_policy import train_mpc_parameter_neural_policy
from scripts.train_mpc_parameter_policy import train_mpc_parameter_policy
from scripts.tune_mpc_parameters import _evaluate_gain_config, tune_mpc_parameters


PRIMARY_METRICS = [
    "BodyAccRMS_mps2",
    "PitchAccRMS_radps2",
    "RollAccRMS_radps2",
    "MaxFrontSusp_m",
    "MaxRearSusp_m",
    "ActionDeltaRMS_N",
    "ActuatorTrackingRMS_N",
    "ActuatorSaturationRatio",
    "ControlAbsEnergy_J",
    "UnsafeSteps",
    "EpisodeReturn",
]


def _fmt(value) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _mean_metrics(scenario_metrics: dict | None) -> dict:
    if not scenario_metrics:
        return {}
    rows = list(scenario_metrics.values())
    out = {}
    for metric in PRIMARY_METRICS:
        values = [float(row[metric]) for row in rows if metric in row and row[metric] is not None]
        if values:
            out[metric] = sum(values) / len(values)
    return out


def _select_scenarios(config: dict, offset: int | None, limit: int | None) -> tuple[dict, int | None, list[str]]:
    if offset is None and limit is None:
        return copy.deepcopy(config), None, [str(s.get("name", "")) for s in config.get("scenarios", [])]
    scenarios = list(config.get("scenarios", []))
    start = int(offset or 0)
    if start < 0:
        raise ValueError("eval_scenario_offset must be non-negative")
    selected = scenarios[start:]
    if limit is not None:
        selected = selected[: int(limit)]
    if not selected:
        raise ValueError(
            f"eval scenario split selected no scenarios: offset={offset}, limit={limit}, total={len(scenarios)}"
        )
    out = copy.deepcopy(config)
    out["scenarios"] = selected
    return out, len(selected), [str(s.get("name", "")) for s in selected]


def _eval_config_with_gains(config: dict, gains: dict, scenario_limit: int | None, score_weights: dict | None) -> tuple[float, dict]:
    trial_config = copy.deepcopy(config)
    trial_config["reduced_full_car_mpc"] = {**trial_config.get("reduced_full_car_mpc", {}), **gains}
    return _evaluate_gain_config(
        trial_config,
        controller_name="FULL_CAR_MPC_LITE",
        scenario_limit=scenario_limit,
        score_weights=score_weights,
    )


def _row(
    *,
    variant: str,
    method: str,
    best_score: float,
    best_reward: float,
    scenario_metrics: dict | None,
    best_config: str,
    manifest: str = "",
    checkpoint: str | None = None,
) -> dict:
    row = {
        "variant": variant,
        "method": method,
        "best_score": float(best_score),
        "best_reward": float(best_reward),
        "best_config": best_config,
        "manifest": manifest,
    }
    if checkpoint:
        row["checkpoint"] = checkpoint
    row.update(_mean_metrics(scenario_metrics))
    return row


def _write_report(rows: list[dict], report_path: Path):
    metric_cols = [metric for metric in PRIMARY_METRICS if any(metric in row for row in rows)]
    lines = [
        "# MPC Parameter Learning Benchmark",
        "",
        "This table compares fixed MPC-lite gains with parameter-learning variants that scale the same structured controller.",
        "",
        "| Variant | Method | Best Score | Policy Score | Score Delta | Policy Delta | Artifact |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        artifact = row.get("manifest") or row.get("best_config") or ""
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["variant"]),
                    str(row["method"]),
                    _fmt(row.get("best_score")),
                    _fmt(row.get("policy_score")),
                    _fmt(row.get("score_delta_vs_fixed")),
                    _fmt(row.get("policy_delta_vs_fixed")),
                    str(artifact),
                ]
            )
            + " |"
        )
    if metric_cols:
        lines += [
            "",
            "## Mean Metrics",
            "",
            "For the neural parameter policy, these metrics come from the deterministic final policy, not the sampled best action.",
            "",
        ]
        header = ["Variant", *metric_cols]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in rows:
            lines.append("| " + " | ".join([str(row["variant"]), *[_fmt(row.get(metric)) for metric in metric_cols]]) + " |")
    best_row = min(rows, key=lambda r: float(r.get("best_score", float("inf"))))
    lines += [
        "",
        f"Best variant by score: `{best_row['variant']}` (`{_fmt(best_row.get('best_score'))}`).",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_mpc_parameter_learning_benchmark(
    config_path: Path,
    out_dir: Path,
    scenario_limit: int | None = 1,
    tuning_trials: int = 8,
    cem_iterations: int = 4,
    cem_population: int = 8,
    neural_episodes: int = 8,
    neural_batch_size: int = 4,
    neural_anchor_coef: float = 0.1,
    neural_anchor_final_coef: float | None = None,
    neural_elite_bc_coef: float = 0.0,
    neural_deployment_gate: bool = False,
    neural_deployment_min_train_improvement: float = 0.0,
    neural_baseline_penalty_weight: float = 1.0,
    neural_unsafe_penalty_weight: float = 5.0,
    neural_suspension_penalty_weight: float = 1.0,
    neural_actuator_penalty_weight: float = 0.2,
    eval_scenario_offset: int | None = None,
    eval_scenario_limit: int | None = None,
    scale_low: float = 0.5,
    scale_high: float = 1.5,
    seed: int = 42,
    include_neural: bool = True,
    device: str | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_config = load_config(config_path)
    score_weights = base_config.get("mpc_parameter_tuning", {}).get("score_weights")
    eval_config, eval_limit, eval_scenarios = _select_scenarios(base_config, eval_scenario_offset, eval_scenario_limit)
    baseline_score, baseline_metrics = _evaluate_gain_config(
        eval_config,
        controller_name="FULL_CAR_MPC_LITE",
        scenario_limit=eval_limit or scenario_limit,
        score_weights=score_weights,
    )
    rows = [
        _row(
            variant="fixed_mpc_lite",
            method="fixed_reduced_full_car_mpc",
            best_score=float(baseline_score),
            best_reward=-float(baseline_score),
            scenario_metrics=baseline_metrics,
            best_config=str(config_path.resolve()),
        )
    ]

    tuning_manifest = tune_mpc_parameters(
        config_path=config_path,
        out_dir=out_dir / "random_tuning",
        trials=tuning_trials,
        seed=seed,
        scenario_limit=scenario_limit,
        scale_low=scale_low,
        scale_high=scale_high,
    )
    tuning_eval_score, tuning_eval_metrics = _eval_config_with_gains(
        eval_config,
        tuning_manifest["best_gains"],
        scenario_limit=eval_limit or scenario_limit,
        score_weights=score_weights,
    )
    rows.append(
        _row(
            variant="random_gain_search",
            method="random_mpc_gain_scaling",
            best_score=float(tuning_eval_score),
            best_reward=-float(tuning_eval_score),
            scenario_metrics=tuning_eval_metrics,
            best_config=tuning_manifest["best_config"],
            manifest=tuning_manifest["manifest"],
        )
    )

    cem_manifest = train_mpc_parameter_policy(
        config_path=config_path,
        out_dir=out_dir / "cem_policy",
        iterations=cem_iterations,
        population=cem_population,
        seed=seed,
        scenario_limit=scenario_limit,
        scale_low=scale_low,
        scale_high=scale_high,
    )
    cem_eval_score, cem_eval_metrics = _eval_config_with_gains(
        eval_config,
        cem_manifest["best_gains"],
        scenario_limit=eval_limit or scenario_limit,
        score_weights=score_weights,
    )
    rows.append(
        _row(
            variant="cem_parameter_policy",
            method=cem_manifest["method"],
            best_score=float(cem_eval_score),
            best_reward=-float(cem_eval_score),
            scenario_metrics=cem_eval_metrics,
            best_config=cem_manifest["best_config"],
            manifest=cem_manifest["manifest"],
        )
    )

    neural_manifest = None
    if include_neural:
        neural_manifest = train_mpc_parameter_neural_policy(
            config_path=config_path,
            out_dir=out_dir / "neural_policy",
            episodes=neural_episodes,
            batch_size=neural_batch_size,
            anchor_coef=neural_anchor_coef,
            anchor_final_coef=neural_anchor_final_coef,
            elite_bc_coef=neural_elite_bc_coef,
            deployment_gate=neural_deployment_gate,
            deployment_min_train_improvement=neural_deployment_min_train_improvement,
            seed=seed,
            scenario_limit=scenario_limit,
            scale_low=scale_low,
            scale_high=scale_high,
            baseline_penalty_weight=neural_baseline_penalty_weight,
            unsafe_penalty_weight=neural_unsafe_penalty_weight,
            suspension_penalty_weight=neural_suspension_penalty_weight,
            actuator_penalty_weight=neural_actuator_penalty_weight,
            eval_scenario_offset=eval_scenario_offset,
            eval_scenario_limit=eval_scenario_limit,
            device=device,
        )
        rows.append(
            _row(
                variant="neural_parameter_policy",
                method=neural_manifest["method"],
                best_score=float(neural_manifest["best_score"]),
                best_reward=float(neural_manifest["best_reward"]),
                scenario_metrics=neural_manifest["final_policy_metrics"],
                best_config=neural_manifest["best_config"],
                manifest=neural_manifest["manifest"],
                checkpoint=neural_manifest["checkpoint"],
            )
        )
        rows[-1]["policy_score"] = float(neural_manifest["final_policy_score"])
        rows[-1]["raw_policy_score"] = float(neural_manifest.get("raw_final_policy_score", neural_manifest["final_policy_score"]))
        rows[-1]["deployment_gate_active"] = bool(neural_manifest.get("deployment_gate_active", False))
        rows[-1]["train_policy_delta_vs_fixed"] = float(neural_manifest.get("train_policy_delta_vs_fixed", 0.0))
        rows[-1]["policy_actions_csv"] = neural_manifest["final_policy_actions_csv"]

    fixed_score = float(rows[0]["best_score"])
    for row in rows:
        row["score_delta_vs_fixed"] = float(row["best_score"]) - fixed_score
        if "policy_score" in row:
            row["policy_delta_vs_fixed"] = float(row["policy_score"]) - fixed_score

    comparison_path = out_dir / "mpc_parameter_learning_comparison.csv"
    pd.DataFrame(rows).sort_values("best_score").to_csv(comparison_path, index=False)
    report_path = out_dir / "mpc_parameter_learning_report.md"
    _write_report(rows, report_path)
    manifest = {
        "config": str(config_path.resolve()),
        "scenario_limit": scenario_limit,
        "eval_scenario_offset": eval_scenario_offset,
        "eval_scenario_limit": eval_scenario_limit,
        "eval_scenarios": eval_scenarios,
        "scale_low": float(scale_low),
        "scale_high": float(scale_high),
        "baseline_metrics": baseline_metrics,
        "tuning_manifest": tuning_manifest["manifest"],
        "cem_manifest": cem_manifest["manifest"],
        "neural_manifest": neural_manifest["manifest"] if neural_manifest else None,
        "neural_constraint_weights": {
            "anchor_coef": float(neural_anchor_coef),
            "anchor_final_coef": None if neural_anchor_final_coef is None else float(neural_anchor_final_coef),
            "elite_bc_coef": float(neural_elite_bc_coef),
            "deployment_gate": bool(neural_deployment_gate),
            "deployment_min_train_improvement": float(neural_deployment_min_train_improvement),
            "baseline_penalty_weight": float(neural_baseline_penalty_weight),
            "unsafe_penalty_weight": float(neural_unsafe_penalty_weight),
            "suspension_penalty_weight": float(neural_suspension_penalty_weight),
            "actuator_penalty_weight": float(neural_actuator_penalty_weight),
        },
        "comparison_csv": str(comparison_path),
        "report": str(report_path),
        "rows": rows,
    }
    manifest_path = out_dir / "mpc_parameter_learning_benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mujoco_full_car_corner.yaml")
    parser.add_argument("--out", default="results/mpc_parameter_learning_benchmark")
    parser.add_argument("--scenario-limit", type=int, default=1)
    parser.add_argument("--tuning-trials", type=int, default=8)
    parser.add_argument("--cem-iterations", type=int, default=4)
    parser.add_argument("--cem-population", type=int, default=8)
    parser.add_argument("--neural-episodes", type=int, default=8)
    parser.add_argument("--neural-batch-size", type=int, default=4)
    parser.add_argument("--neural-anchor-coef", type=float, default=0.1)
    parser.add_argument("--neural-anchor-final-coef", type=float, default=None)
    parser.add_argument("--neural-elite-bc-coef", type=float, default=0.0)
    parser.add_argument("--neural-deployment-gate", action="store_true")
    parser.add_argument("--neural-deployment-min-train-improvement", type=float, default=0.0)
    parser.add_argument("--neural-baseline-penalty-weight", type=float, default=1.0)
    parser.add_argument("--neural-unsafe-penalty-weight", type=float, default=5.0)
    parser.add_argument("--neural-suspension-penalty-weight", type=float, default=1.0)
    parser.add_argument("--neural-actuator-penalty-weight", type=float, default=0.2)
    parser.add_argument("--eval-scenario-offset", type=int, default=None)
    parser.add_argument("--eval-scenario-limit", type=int, default=None)
    parser.add_argument("--scale-low", type=float, default=0.5)
    parser.add_argument("--scale-high", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-neural", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    manifest = run_mpc_parameter_learning_benchmark(
        config_path=config_path,
        out_dir=out_dir,
        scenario_limit=args.scenario_limit,
        tuning_trials=args.tuning_trials,
        cem_iterations=args.cem_iterations,
        cem_population=args.cem_population,
        neural_episodes=args.neural_episodes,
        neural_batch_size=args.neural_batch_size,
        neural_anchor_coef=args.neural_anchor_coef,
        neural_anchor_final_coef=args.neural_anchor_final_coef,
        neural_elite_bc_coef=args.neural_elite_bc_coef,
        neural_deployment_gate=args.neural_deployment_gate,
        neural_deployment_min_train_improvement=args.neural_deployment_min_train_improvement,
        neural_baseline_penalty_weight=args.neural_baseline_penalty_weight,
        neural_unsafe_penalty_weight=args.neural_unsafe_penalty_weight,
        neural_suspension_penalty_weight=args.neural_suspension_penalty_weight,
        neural_actuator_penalty_weight=args.neural_actuator_penalty_weight,
        eval_scenario_offset=args.eval_scenario_offset,
        eval_scenario_limit=args.eval_scenario_limit,
        scale_low=args.scale_low,
        scale_high=args.scale_high,
        seed=args.seed,
        include_neural=not args.skip_neural,
        device=args.device,
    )
    print(f"Saved MPC parameter learning benchmark to {manifest['manifest']}")
    print(f"Report: {manifest['report']}")


if __name__ == "__main__":
    main()
