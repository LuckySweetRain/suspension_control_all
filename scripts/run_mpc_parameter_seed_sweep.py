from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_mpc_parameter_learning_benchmark import run_mpc_parameter_learning_benchmark
from config import load_config


def _format(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.6g}"


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _seed_row(seed: int, manifest: dict) -> dict:
    comparison = pd.read_csv(manifest["comparison_csv"])
    fixed = comparison[comparison["variant"] == "fixed_mpc_lite"].iloc[0]
    neural = comparison[comparison["variant"] == "neural_parameter_policy"].iloc[0]
    fixed_score = float(fixed["best_score"])
    policy_score = float(neural["policy_score"])
    row = {
        "seed": int(seed),
        "fixed_score": fixed_score,
        "neural_best_score": float(neural["best_score"]),
        "neural_raw_policy_score": float(neural.get("raw_policy_score", neural["policy_score"])),
        "neural_policy_score": policy_score,
        "neural_policy_delta_vs_fixed": policy_score - fixed_score,
        "neural_policy_supported": bool(policy_score <= fixed_score),
        "deployment_gate_active": _as_bool(neural.get("deployment_gate_active", False)),
        "train_policy_delta_vs_fixed": float(neural.get("train_policy_delta_vs_fixed", 0.0)),
        "benchmark_manifest": manifest["manifest"],
        "comparison_csv": manifest["comparison_csv"],
        "report": manifest["report"],
    }
    for metric in ["BodyAccRMS_mps2", "UnsafeSteps", "ControlAbsEnergy_J", "EpisodeReturn"]:
        if metric in fixed and metric in neural and pd.notna(fixed[metric]) and pd.notna(neural[metric]):
            row[f"neural_delta_{metric}"] = float(neural[metric]) - float(fixed[metric])
    return row


def _scenario_names(config_path: Path, scenario_limit: int | None, eval_scenario_offset: int | None, eval_scenario_limit: int | None):
    config = load_config(config_path)
    scenarios = list(config.get("scenarios", []))
    train = scenarios[: int(scenario_limit)] if scenario_limit is not None else scenarios
    start = int(eval_scenario_offset or 0)
    eval_set = scenarios[start:]
    if eval_scenario_limit is not None:
        eval_set = eval_set[: int(eval_scenario_limit)]
    return [str(s.get("name", "")) for s in train], [str(s.get("name", "")) for s in eval_set]


def _write_report(summary: dict, out_dir: Path):
    lines = [
        "# MPC Parameter Policy Seed Sweep",
        "",
        f"- Non-inferiority status: `{summary['noninferiority_status']}`",
        f"- Improvement status: `{summary['improvement_status']}`",
        f"- Seed count: `{summary['seed_count']}`",
        f"- Neural policy support rate: `{summary['support_rate']:.3f}`",
        f"- Mean policy delta vs fixed MPC-lite: `{summary['mean_policy_delta_vs_fixed']:.6g}`",
        f"- Mean unsafe delta vs fixed MPC-lite: `{summary['mean_unsafe_delta_vs_fixed']:.6g}`",
        f"- Train scenarios: `{', '.join(summary['train_scenarios'])}`",
        f"- Eval scenarios: `{', '.join(summary['eval_scenarios'])}`",
        "",
        "| Seed | Fixed Score | Neural Policy Score | Policy Delta | Supported |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["seeds"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["seed"]),
                    _format(row["fixed_score"]),
                    _format(row["neural_policy_score"]),
                    _format(row["neural_policy_delta_vs_fixed"]),
                    str(row["neural_policy_supported"]),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "A seed is marked supported when the deterministic neural parameter policy score is no worse than fixed MPC-lite.",
        "",
        "## Claim Gate",
        "",
        f"- Required support rate: `{summary['claim_thresholds']['min_support_rate']:.3f}`",
        f"- Max mean policy delta: `{summary['claim_thresholds']['max_mean_policy_delta']:.6g}`",
        f"- Max mean unsafe delta: `{summary['claim_thresholds']['max_mean_unsafe_delta']:.6g}`",
        f"- Non-inferiority failed gates: `{', '.join(summary['noninferiority_failed_gates']) if summary['noninferiority_failed_gates'] else 'none'}`",
        f"- Required mean improvement: `{summary['claim_thresholds']['min_mean_policy_improvement']:.6g}`",
        f"- Improvement failed gates: `{', '.join(summary['improvement_failed_gates']) if summary['improvement_failed_gates'] else 'none'}`",
        "",
    ]
    if summary.get("delta_figure"):
        rel = Path(summary["delta_figure"]).resolve().relative_to(out_dir.resolve())
        lines += [
            "## Figure",
            "",
            f"![Policy delta by seed]({rel.as_posix()})",
            "",
        ]
    if summary.get("latex_table"):
        rel = Path(summary["latex_table"]).resolve().relative_to(out_dir.resolve())
        lines += [
            "## Paper Table",
            "",
            f"LaTeX table: `{rel.as_posix()}`",
            "",
        ]
    (out_dir / "mpc_parameter_seed_sweep_report.md").write_text("\n".join(lines), encoding="utf-8")


def _plot_policy_delta(rows: list[dict], out_dir: Path) -> Path:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    seeds = [str(row["seed"]) for row in rows]
    deltas = [float(row["neural_policy_delta_vs_fixed"]) for row in rows]
    colors = ["tab:green" if delta <= 0.0 else "tab:red" for delta in deltas]
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(seeds)), 4))
    ax.bar(seeds, deltas, color=colors)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Policy score delta vs fixed MPC-lite")
    ax.set_title("Deterministic neural MPC-parameter policy delta")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = fig_dir / "mpc_parameter_policy_delta.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _write_latex_table(summary: dict, out_dir: Path) -> Path:
    path = out_dir / "mpc_parameter_seed_sweep_table.tex"
    rows = summary.get("seeds", [])
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Held-out seed sweep for the BP-NMPS policy against fixed MPC-lite. Lower score is better.}",
        r"\label{tab:bp_nmps_seed_sweep}",
        r"\begin{tabular}{rrrrrrr}",
        r"\toprule",
        r"Seed & Fixed score & BP-NMPS score & Score delta & Unsafe delta & Gate & Supported \\",
        r"\midrule",
    ]
    for row in rows:
        unsafe_delta = row.get("neural_delta_UnsafeSteps", 0.0)
        supported = "Yes" if row.get("neural_policy_supported") else "No"
        lines.append(
            f"{int(row['seed'])} & "
            f"{_format(row['fixed_score'])} & "
            f"{_format(row['neural_policy_score'])} & "
            f"{_format(row['neural_policy_delta_vs_fixed'])} & "
            f"{_format(unsafe_delta)} & "
            f"{'On' if row.get('deployment_gate_active') else 'Off'} & "
            f"{supported} \\\\"
        )
    lines += [
        r"\midrule",
        f"Mean &  &  & {_format(summary['mean_policy_delta_vs_fixed'])} & "
        f"{_format(summary['mean_unsafe_delta_vs_fixed'])} & "
        f" & "
        f"{summary['support_rate']:.3f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_mpc_parameter_seed_sweep(
    config_path: Path,
    out_dir: Path,
    seeds: list[int],
    scenario_limit: int | None = 1,
    tuning_trials: int = 4,
    cem_iterations: int = 2,
    cem_population: int = 4,
    neural_episodes: int = 4,
    neural_batch_size: int = 2,
    neural_anchor_coef: float = 0.1,
    neural_anchor_final_coef: float | None = None,
    neural_elite_bc_coef: float = 0.0,
    neural_deployment_gate: bool = False,
    neural_deployment_min_train_improvement: float = 0.0,
    scale_low: float = 0.5,
    scale_high: float = 1.5,
    neural_baseline_penalty_weight: float = 1.0,
    neural_unsafe_penalty_weight: float = 5.0,
    neural_suspension_penalty_weight: float = 1.0,
    neural_actuator_penalty_weight: float = 0.2,
    eval_scenario_offset: int | None = None,
    eval_scenario_limit: int | None = None,
    min_support_rate: float = 0.8,
    max_mean_policy_delta: float = 0.0,
    max_mean_unsafe_delta: float = 0.0,
    min_mean_policy_improvement: float = 0.0,
    device: str | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        seed_dir = out_dir / f"seed_{int(seed)}"
        manifest = run_mpc_parameter_learning_benchmark(
            config_path=config_path,
            out_dir=seed_dir,
            scenario_limit=scenario_limit,
            tuning_trials=tuning_trials,
            cem_iterations=cem_iterations,
            cem_population=cem_population,
            neural_episodes=neural_episodes,
            neural_batch_size=neural_batch_size,
            neural_anchor_coef=neural_anchor_coef,
            neural_anchor_final_coef=neural_anchor_final_coef,
            neural_elite_bc_coef=neural_elite_bc_coef,
            neural_deployment_gate=neural_deployment_gate,
            neural_deployment_min_train_improvement=neural_deployment_min_train_improvement,
            neural_baseline_penalty_weight=neural_baseline_penalty_weight,
            neural_unsafe_penalty_weight=neural_unsafe_penalty_weight,
            neural_suspension_penalty_weight=neural_suspension_penalty_weight,
            neural_actuator_penalty_weight=neural_actuator_penalty_weight,
            eval_scenario_offset=eval_scenario_offset,
            eval_scenario_limit=eval_scenario_limit,
            scale_low=scale_low,
            scale_high=scale_high,
            seed=int(seed),
            include_neural=True,
            device=device,
        )
        rows.append(_seed_row(int(seed), manifest))

    supported = sum(1 for row in rows if row["neural_policy_supported"])
    deltas = [float(row["neural_policy_delta_vs_fixed"]) for row in rows]
    unsafe_deltas = [float(row.get("neural_delta_UnsafeSteps", 0.0)) for row in rows]
    support_rate = float(supported / max(1, len(rows)))
    mean_policy_delta = float(sum(deltas) / max(1, len(deltas)))
    mean_unsafe_delta = float(sum(unsafe_deltas) / max(1, len(unsafe_deltas)))
    noninferiority_failed_gates = []
    if support_rate < float(min_support_rate):
        noninferiority_failed_gates.append("support_rate")
    if mean_policy_delta > float(max_mean_policy_delta):
        noninferiority_failed_gates.append("mean_policy_delta")
    if mean_unsafe_delta > float(max_mean_unsafe_delta):
        noninferiority_failed_gates.append("mean_unsafe_delta")
    improvement_failed_gates = list(noninferiority_failed_gates)
    if mean_policy_delta >= -float(min_mean_policy_improvement):
        improvement_failed_gates.append("mean_policy_improvement")
    delta_figure = _plot_policy_delta(rows, out_dir)
    train_scenarios, eval_scenarios = _scenario_names(config_path, scenario_limit, eval_scenario_offset, eval_scenario_limit)
    summary = {
        "config": str(config_path.resolve()),
        "out_dir": str(out_dir.resolve()),
        "train_scenarios": train_scenarios,
        "eval_scenarios": eval_scenarios,
        "seeds": rows,
        "seed_count": len(rows),
        "supported_seeds": int(supported),
        "support_rate": support_rate,
        "mean_policy_delta_vs_fixed": mean_policy_delta,
        "mean_unsafe_delta_vs_fixed": mean_unsafe_delta,
        "claim_thresholds": {
            "min_support_rate": float(min_support_rate),
            "max_mean_policy_delta": float(max_mean_policy_delta),
            "max_mean_unsafe_delta": float(max_mean_unsafe_delta),
            "min_mean_policy_improvement": float(min_mean_policy_improvement),
        },
        "noninferiority_failed_gates": noninferiority_failed_gates,
        "improvement_failed_gates": improvement_failed_gates,
        "noninferiority_status": "supported" if rows and not noninferiority_failed_gates else "needs_more_evidence",
        "improvement_status": "supported" if rows and not improvement_failed_gates else "needs_more_evidence",
        "status": "supported" if rows and not noninferiority_failed_gates else "needs_more_evidence",
        "delta_figure": str(delta_figure.resolve()),
    }
    csv_path = out_dir / "mpc_parameter_seed_sweep_summary.csv"
    json_path = out_dir / "mpc_parameter_seed_sweep_summary.json"
    latex_path = _write_latex_table(summary, out_dir)
    summary["latex_table"] = str(latex_path.resolve())
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_report(summary, out_dir)
    summary["csv_path"] = str(csv_path.resolve())
    summary["json_path"] = str(json_path.resolve())
    summary["markdown_path"] = str((out_dir / "mpc_parameter_seed_sweep_report.md").resolve())
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mujoco_full_car_corner.yaml")
    parser.add_argument("--out", default="results/mpc_parameter_seed_sweep")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--scenario-limit", type=int, default=1)
    parser.add_argument("--tuning-trials", type=int, default=4)
    parser.add_argument("--cem-iterations", type=int, default=2)
    parser.add_argument("--cem-population", type=int, default=4)
    parser.add_argument("--neural-episodes", type=int, default=4)
    parser.add_argument("--neural-batch-size", type=int, default=2)
    parser.add_argument("--neural-anchor-coef", type=float, default=0.1)
    parser.add_argument("--neural-anchor-final-coef", type=float, default=None)
    parser.add_argument("--neural-elite-bc-coef", type=float, default=0.0)
    parser.add_argument("--neural-deployment-gate", action="store_true")
    parser.add_argument("--neural-deployment-min-train-improvement", type=float, default=0.0)
    parser.add_argument("--scale-low", type=float, default=0.5)
    parser.add_argument("--scale-high", type=float, default=1.5)
    parser.add_argument("--neural-baseline-penalty-weight", type=float, default=1.0)
    parser.add_argument("--neural-unsafe-penalty-weight", type=float, default=5.0)
    parser.add_argument("--neural-suspension-penalty-weight", type=float, default=1.0)
    parser.add_argument("--neural-actuator-penalty-weight", type=float, default=0.2)
    parser.add_argument("--eval-scenario-offset", type=int, default=None)
    parser.add_argument("--eval-scenario-limit", type=int, default=None)
    parser.add_argument("--min-support-rate", type=float, default=0.8)
    parser.add_argument("--max-mean-policy-delta", type=float, default=0.0)
    parser.add_argument("--max-mean-unsafe-delta", type=float, default=0.0)
    parser.add_argument("--min-mean-policy-improvement", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    summary = run_mpc_parameter_seed_sweep(
        config_path=config_path,
        out_dir=out_dir,
        seeds=seeds,
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
        scale_low=args.scale_low,
        scale_high=args.scale_high,
        neural_baseline_penalty_weight=args.neural_baseline_penalty_weight,
        neural_unsafe_penalty_weight=args.neural_unsafe_penalty_weight,
        neural_suspension_penalty_weight=args.neural_suspension_penalty_weight,
        neural_actuator_penalty_weight=args.neural_actuator_penalty_weight,
        eval_scenario_offset=args.eval_scenario_offset,
        eval_scenario_limit=args.eval_scenario_limit,
        min_support_rate=args.min_support_rate,
        max_mean_policy_delta=args.max_mean_policy_delta,
        max_mean_unsafe_delta=args.max_mean_unsafe_delta,
        min_mean_policy_improvement=args.min_mean_policy_improvement,
        device=args.device,
    )
    print(f"Saved MPC parameter seed sweep to {summary['json_path']}")
    print(f"Report: {summary['markdown_path']}")


if __name__ == "__main__":
    main()
