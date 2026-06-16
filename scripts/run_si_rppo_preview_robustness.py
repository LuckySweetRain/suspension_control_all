"""Evaluate SI-RPPO variants under road-preview corruption cases."""

from __future__ import annotations

import argparse
import json
import math
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import deep_update, load_config
from experiments.evaluation import evaluate_all


PREVIEW_CASES = [
    {
        "name": "clean_preview",
        "overrides": {"preview_error": {"enabled": False}},
    },
    {
        "name": "nominal_preview_error",
        "overrides": {
            "preview_error": {
                "enabled": True,
                "delay_steps": 2,
                "height_noise_std": 0.003,
                "bias_std": 0.002,
                "dropout_prob": 0.05,
                "scale_error_std": 0.05,
            }
        },
    },
    {
        "name": "strong_preview_error",
        "overrides": {
            "preview_error": {
                "enabled": True,
                "delay_steps": 5,
                "height_noise_std": 0.008,
                "bias_std": 0.006,
                "dropout_prob": 0.15,
                "scale_error_std": 0.12,
            }
        },
    },
    {
        "name": "missing_preview_segments",
        "overrides": {
            "preview_error": {
                "enabled": True,
                "delay_steps": 3,
                "height_noise_std": 0.004,
                "bias_std": 0.003,
                "dropout_prob": 0.35,
                "scale_error_std": 0.08,
            }
        },
    },
]

SUMMARY_METRICS = [
    "EpisodeReturn",
    "UnsafeSteps",
    "BodyAccRMS_mps2",
    "PitchAccRMS_radps2",
    "RollAccRMS_radps2",
    "ActionDeltaRMS_N",
    "ActuatorTrackingRMS_N",
    "ActuatorSaturationRatio",
    "PolicyProjectionError",
    "PolicyImprovementGate",
]


@dataclass(frozen=True)
class PolicySpec:
    variant: str
    algorithm: str
    config: Path
    checkpoint: Path


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return data


def _policy_specs_from_manifest(
    manifest: dict[str, Any],
    variants: list[str] | None = None,
    baseline_algorithms: list[str] | None = None,
) -> list[PolicySpec]:
    allowed_variants = set(variants or [])
    allowed_baselines = {item.lower() for item in (baseline_algorithms or [])}
    specs: list[PolicySpec] = []
    for name, item in dict(manifest.get("variants", {})).items():
        if allowed_variants and name not in allowed_variants:
            continue
        specs.append(
            PolicySpec(
                variant=str(name),
                algorithm="ppo",
                config=_resolve_path(item["config"]),
                checkpoint=_resolve_path(item["checkpoint"]),
            )
        )
    for name, item in dict(manifest.get("off_policy_baselines", {})).items():
        algorithm = str(item.get("algorithm", "")).lower()
        if allowed_baselines and algorithm not in allowed_baselines:
            continue
        specs.append(
            PolicySpec(
                variant=str(name),
                algorithm=algorithm,
                config=_resolve_path(item["config"]),
                checkpoint=_resolve_path(item["checkpoint"]),
            )
        )
    return specs


def _prepare_eval_config(config: dict[str, Any], case: dict[str, Any], scenario_limit: int | None, controllers: list[str]) -> dict[str, Any]:
    eval_config = deep_update(deepcopy(config), dict(case.get("overrides", {})))
    if scenario_limit is not None:
        eval_config["scenarios"] = list(eval_config.get("scenarios", []))[: int(scenario_limit)]
    eval_config["evaluation"] = deep_update(dict(eval_config.get("evaluation", {})), {"controllers": controllers})
    return eval_config


def build_preview_case_summary(rows: pd.DataFrame) -> pd.DataFrame:
    available = [metric for metric in SUMMARY_METRICS if metric in rows.columns]
    if rows.empty or not available:
        return pd.DataFrame(columns=["PreviewCase", "Variant", "Controller", *available])
    return (
        rows.groupby(["PreviewCase", "Variant", "Controller"], dropna=False)[available]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(["PreviewCase", "Variant", "Controller"])
    )


def _mean_metric_delta(summary: pd.DataFrame, candidate: str, baseline: str) -> dict[str, float | None]:
    if summary.empty:
        return {}
    candidate_rows = summary[summary["Variant"] == candidate]
    baseline_rows = summary[summary["Variant"] == baseline]
    if candidate_rows.empty or baseline_rows.empty:
        return {}
    join_keys = ["PreviewCase"]
    if "Controller" in candidate_rows.columns and "Controller" in baseline_rows.columns:
        candidate_rows = candidate_rows.drop(columns=["Controller"])
        baseline_rows = baseline_rows.drop(columns=["Controller"])
    merged = candidate_rows.merge(baseline_rows, on=join_keys, suffixes=("_candidate", "_baseline"))
    deltas: dict[str, float | None] = {}
    for metric in SUMMARY_METRICS:
        c_key = f"{metric}_candidate"
        b_key = f"{metric}_baseline"
        if c_key in merged.columns and b_key in merged.columns:
            delta = merged[c_key].astype(float) - merged[b_key].astype(float)
            deltas[metric] = float(delta.mean())
    return deltas


def build_pairwise_delta_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparisons = [
        ("bc_ppo", "ppo_scratch", "SI-RPPO vs PPO scratch"),
        ("bc_ppo", "sac_baseline", "SI-RPPO vs SAC"),
        ("bc_ppo", "td3_baseline", "SI-RPPO vs TD3"),
        ("bc_ppo", "structured_baselines", "SI-RPPO vs structured baselines"),
    ]
    for candidate, baseline, label in comparisons:
        deltas = _mean_metric_delta(summary, candidate, baseline)
        if not deltas:
            continue
        row = {"Comparison": label, "Candidate": candidate, "Baseline": baseline}
        row.update({f"{metric}Delta": value for metric, value in deltas.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _write_report(summary: pd.DataFrame, pairwise: pd.DataFrame, out_dir: Path, metadata: dict[str, Any]) -> tuple[Path, Path]:
    report_path = out_dir / "si_rppo_preview_robustness.md"
    latex_path = out_dir / "si_rppo_preview_robustness.tex"
    lines = [
        "# SI-RPPO Preview Robustness",
        "",
        "This report evaluates already-trained SI-RPPO/PPO variants under road-preview corruption cases.",
        "",
        "## Scope",
        "",
        f"- Preview cases: `{', '.join(metadata.get('cases', []))}`",
        f"- Scenario limit: `{metadata.get('scenario_limit')}`",
        f"- Variants: `{', '.join(metadata.get('variants', []))}`",
        f"- Structured baselines included: `{metadata.get('include_structured_baselines')}`",
        "",
        "## Interpretation",
        "",
        "- Higher `EpisodeReturn` is better.",
        "- Lower safety, comfort, actuator, and projection metrics are better.",
        "- `PolicyImprovementGate` indicates how much the active PPO correction is opened by the deployed safety wrapper.",
        "",
    ]
    if summary.empty:
        lines += ["No summary rows were generated.", ""]
    else:
        table_columns = [column for column in ["PreviewCase", "Variant", "Controller", *SUMMARY_METRICS] if column in summary.columns]
        lines += [
            "## Mean Metrics",
            "",
            summary[table_columns].to_markdown(index=False),
            "",
        ]
    if not pairwise.empty:
        report_columns = [
            column
            for column in [
                "Comparison",
                "EpisodeReturnDelta",
                "UnsafeStepsDelta",
                "BodyAccRMS_mps2Delta",
                "PitchAccRMS_radps2Delta",
                "RollAccRMS_radps2Delta",
                "ActionDeltaRMS_NDelta",
                "PolicyProjectionErrorDelta",
                "PolicyImprovementGateDelta",
            ]
            if column in pairwise.columns
        ]
        lines += [
            "## Mean Pairwise Deltas",
            "",
            "Delta is candidate minus baseline; higher return is better, while lower safety/comfort/action/projection deltas are better.",
            "",
            pairwise[report_columns].to_markdown(index=False),
            "",
        ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{SI-RPPO preview-corruption robustness pairwise deltas.}",
        r"\label{tab:si_rppo_preview_robustness}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Comparison & Return $\Delta$ & Unsafe $\Delta$ & Body RMS $\Delta$ & Roll RMS $\Delta$ & Action $\Delta$ & Projection $\Delta$ \\",
        r"\midrule",
    ]
    if not pairwise.empty:
        def fmt(value: Any, digits: int = 2) -> str:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return "--"
            if math.isnan(numeric):
                return "--"
            return f"{numeric:.{digits}f}"

        for _, row in pairwise.iterrows():
            latex_lines.append(
                f"{str(row['Comparison']).replace('_', r'\_')} & "
                f"{fmt(row.get('EpisodeReturnDelta'), 2)} & "
                f"{fmt(row.get('UnsafeStepsDelta'), 2)} & "
                f"{fmt(row.get('BodyAccRMS_mps2Delta'), 3)} & "
                f"{fmt(row.get('RollAccRMS_radps2Delta'), 3)} & "
                f"{fmt(row.get('ActionDeltaRMS_NDelta'), 2)} & "
                f"{fmt(row.get('PolicyProjectionErrorDelta'), 3)} \\\\"
            )
    elif not summary.empty:
        latex_lines.append(r"\multicolumn{7}{c}{Pairwise deltas unavailable.} \\")
    latex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    latex_path.write_text("\n".join(latex_lines), encoding="utf-8")
    return report_path, latex_path


def run_si_rppo_preview_robustness(
    manifest_path: Path,
    out_dir: Path,
    variants: list[str] | None = None,
    baseline_algorithms: list[str] | None = None,
    scenario_limit: int | None = None,
    case_limit: int | None = None,
    include_structured_baselines: bool = True,
) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = PREVIEW_CASES[: int(case_limit)] if case_limit is not None else PREVIEW_CASES
    specs = _policy_specs_from_manifest(manifest, variants=variants, baseline_algorithms=baseline_algorithms)
    if not specs and not include_structured_baselines:
        raise ValueError("No policies selected and structured baselines disabled.")

    all_rows: list[pd.DataFrame] = []
    evaluations: list[dict[str, Any]] = []
    if include_structured_baselines:
        base_config = load_config(_resolve_path(manifest.get("materialized_base_config", manifest["base_config"])))
        for case in cases:
            eval_config = _prepare_eval_config(base_config, case, scenario_limit, ["PASSIVE", "FULL_CAR_MPC_LITE"])
            result_dir = out_dir / case["name"] / "structured_baselines"
            metrics, _ = evaluate_all(eval_config, checkpoints={}, result_dir=result_dir)
            metrics.insert(0, "Variant", "structured_baselines")
            metrics.insert(0, "PreviewCase", case["name"])
            all_rows.append(metrics)
            evaluations.append({"case": case["name"], "variant": "structured_baselines", "result_dir": str(result_dir)})

    for spec in specs:
        base_config = load_config(spec.config)
        controller_name = spec.algorithm.upper()
        for case in cases:
            eval_config = _prepare_eval_config(base_config, case, scenario_limit, [controller_name])
            result_dir = out_dir / case["name"] / spec.variant
            metrics, _ = evaluate_all(
                eval_config,
                checkpoints={spec.algorithm: str(spec.checkpoint)},
                result_dir=result_dir,
            )
            metrics.insert(0, "Variant", spec.variant)
            metrics.insert(0, "PreviewCase", case["name"])
            all_rows.append(metrics)
            evaluations.append(
                {
                    "case": case["name"],
                    "variant": spec.variant,
                    "algorithm": spec.algorithm,
                    "checkpoint": str(spec.checkpoint),
                    "result_dir": str(result_dir),
                }
            )

    combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    combined_path = out_dir / "si_rppo_preview_robustness.csv"
    combined.to_csv(combined_path, index=False)
    summary = build_preview_case_summary(combined)
    summary_path = out_dir / "si_rppo_preview_robustness_summary.csv"
    summary.to_csv(summary_path, index=False)
    pairwise = build_pairwise_delta_summary(summary)
    pairwise_path = out_dir / "si_rppo_preview_robustness_pairwise.csv"
    pairwise.to_csv(pairwise_path, index=False)
    result_manifest = {
        "source_manifest": str(manifest_path.resolve()),
        "out_dir": str(out_dir.resolve()),
        "cases": [case["name"] for case in cases],
        "scenario_limit": scenario_limit,
        "variants": [spec.variant for spec in specs],
        "baseline_algorithms": [spec.algorithm for spec in specs if spec.algorithm != "ppo"],
        "include_structured_baselines": bool(include_structured_baselines),
        "evaluations": evaluations,
        "combined_csv": str(combined_path),
        "summary_csv": str(summary_path),
        "pairwise_csv": str(pairwise_path),
    }
    report_path, latex_path = _write_report(summary, pairwise, out_dir, result_manifest)
    result_manifest["report"] = str(report_path)
    result_manifest["latex_table"] = str(latex_path)
    manifest_out = out_dir / "si_rppo_preview_robustness_manifest.json"
    manifest_out.write_text(json.dumps(result_manifest, indent=2) + "\n", encoding="utf-8")
    result_manifest["manifest"] = str(manifest_out)
    return result_manifest


def _split_csv(value: str) -> list[str] | None:
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/si_rppo_e20_improvement_gate_fair_baselines/si_rppo_ablation_manifest.json"),
    )
    parser.add_argument("--out", type=Path, default=Path("results/si_rppo_preview_robustness"))
    parser.add_argument("--variants", default="bc_ppo,ppo_scratch")
    parser.add_argument("--baseline-algorithms", default="sac")
    parser.add_argument("--eval-scenario-limit", type=int, default=None)
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--no-structured-baselines", action="store_true")
    args = parser.parse_args()
    manifest_path = _resolve_path(args.manifest)
    out_dir = _resolve_path(args.out)
    result = run_si_rppo_preview_robustness(
        manifest_path=manifest_path,
        out_dir=out_dir,
        variants=_split_csv(args.variants),
        baseline_algorithms=_split_csv(args.baseline_algorithms),
        scenario_limit=args.eval_scenario_limit,
        case_limit=args.case_limit,
        include_structured_baselines=not args.no_structured_baselines,
    )
    print(f"Saved SI-RPPO preview robustness report to {result['report']}")


if __name__ == "__main__":
    main()
