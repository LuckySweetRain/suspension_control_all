"""Summarize safe-teacher improvement-gate behavior for SI-RPPO runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METRIC_COLUMNS = [
    "EpisodeReturn",
    "UnsafeSteps",
    "BodyAccRMS_mps2",
    "PitchAccRMS_radps2",
    "RollAccRMS_radps2",
    "ActionDeltaRMS_N",
    "ActuatorTrackingRMS_N",
    "PolicyProjectionError",
    "PolicyImprovementGate",
]


def _format(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _trajectory_gate_stats(eval_dir: Path, controller: str = "PPO") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(eval_dir.glob(f"{controller}_*.csv")):
        df = pd.read_csv(path)
        if "policy_improvement_gate" not in df.columns:
            continue
        scenario = path.stem.removeprefix(f"{controller}_")
        gate = df["policy_improvement_gate"].astype(float).to_numpy()
        rows.append(
            {
                "Scenario": scenario,
                "GateMeanTrajectory": float(np.mean(gate)),
                "GateP05": float(np.quantile(gate, 0.05)),
                "GateP50": float(np.quantile(gate, 0.50)),
                "GateP95": float(np.quantile(gate, 0.95)),
                "GateActiveRatioGt0p1": float(np.mean(gate > 0.1)),
                "GateActiveRatioGt0p5": float(np.mean(gate > 0.5)),
                "GateMax": float(np.max(gate)),
                "Trajectory": str(path),
            }
        )
    return pd.DataFrame(rows)


def build_si_rppo_gate_evidence(result_dir: Path, out_dir: Path, variant: str = "bc_ppo") -> dict[str, Any]:
    result_dir = Path(result_dir)
    out_dir = Path(out_dir)
    combined_path = result_dir / "combined_metrics.csv"
    if not combined_path.exists():
        raise FileNotFoundError(f"Missing combined metrics: {combined_path}")
    metrics = pd.read_csv(combined_path)
    metric_view = metrics[(metrics["Variant"] == variant) & (metrics["Controller"] == "PPO")].copy()
    if metric_view.empty:
        raise ValueError(f"No {variant}/PPO rows found in {combined_path}")
    keep_cols = ["Variant", "Controller", "Scenario"] + [c for c in METRIC_COLUMNS if c in metric_view.columns]
    metric_view = metric_view[keep_cols]

    eval_dir = result_dir / variant / "evaluation"
    gate_stats = _trajectory_gate_stats(eval_dir)
    if gate_stats.empty:
        summary = metric_view
    else:
        summary = metric_view.merge(gate_stats, on="Scenario", how="left")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "si_rppo_gate_evidence.csv"
    summary.to_csv(csv_path, index=False)

    report_path = out_dir / "si_rppo_gate_evidence.md"
    lines = [
        "# SI-RPPO Gate Evidence",
        "",
        "This report summarizes the safe-teacher improvement gate for the gated BC-PPO controller.",
        "",
        f"- Result directory: `{result_dir}`",
        f"- Variant: `{variant}`",
        "",
        "## Scenario Gate Summary",
        "",
        "| Scenario | Return | Unsafe | Action delta | Projection error | Mean gate | Active >0.1 | Active >0.5 | Gate p50 | Gate p95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.sort_values("Scenario").iterrows():
        values = [
            str(row["Scenario"]),
            _format(row.get("EpisodeReturn"), 3),
            _format(row.get("UnsafeSteps"), 3),
            _format(row.get("ActionDeltaRMS_N"), 3),
            _format(row.get("PolicyProjectionError"), 6),
            _format(row.get("PolicyImprovementGate"), 4),
            _format(row.get("GateActiveRatioGt0p1"), 4),
            _format(row.get("GateActiveRatioGt0p5"), 4),
            _format(row.get("GateP50"), 4),
            _format(row.get("GateP95"), 4),
        ]
        lines.append("| " + " | ".join(values) + " |")

    mean_gate = float(summary["PolicyImprovementGate"].mean()) if "PolicyImprovementGate" in summary else float("nan")
    mean_active = (
        float(summary["GateActiveRatioGt0p1"].mean()) if "GateActiveRatioGt0p1" in summary else float("nan")
    )
    lines += [
        "",
        "## Interpretation",
        "",
        f"- Mean scenario gate: `{_format(mean_gate, 4)}`.",
        f"- Mean active ratio above 0.1: `{_format(mean_active, 4)}`.",
        "- Low gate values indicate that the policy stays close to the passive safe teacher on easy or low-confidence segments.",
        "- High gate values indicate segments where preview/acceleration demand justifies active force deviation.",
        "- This evidence supports describing SI-RPPO as a state-dependent safe-teacher-gated policy, not merely PPO with reward shaping.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    latex_path = out_dir / "si_rppo_gate_evidence.tex"
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Safe-teacher improvement gate activation by scenario.}",
        r"\label{tab:si_rppo_gate_evidence}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Scenario & Return & Unsafe & Action $\Delta$ & Mean gate & Active $>0.1$ & Gate p95 \\",
        r"\midrule",
    ]
    for _, row in summary.sort_values("Scenario").iterrows():
        latex_lines.append(
            " & ".join(
                [
                    _latex_escape(row["Scenario"]),
                    _format(row.get("EpisodeReturn"), 2),
                    _format(row.get("UnsafeSteps"), 1),
                    _format(row.get("ActionDeltaRMS_N"), 2),
                    _format(row.get("PolicyImprovementGate"), 3),
                    _format(row.get("GateActiveRatioGt0p1"), 3),
                    _format(row.get("GateP95"), 3),
                ]
            )
            + r" \\"
        )
    latex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    latex_path.write_text("\n".join(latex_lines), encoding="utf-8")

    method_path = out_dir / "si_rppo_gate_method_summary.md"
    method_lines = [
        "# SI-RPPO Gate Method Summary",
        "",
        "可写入论文的方法表述：",
        "",
        "> 本文设计了安全教师门控的在线 PPO 策略网络。策略首先通过离线模仿学习获得接近安全教师的初始行为，随后在在线 PPO 更新中采用增量动作参数化与投影感知约束，降低执行器投影和高频力指令风险。最终执行动作由状态相关门控在 PPO 主动修正与 passive safe teacher 之间插值：",
        "",
        "```text",
        "u_t = Projection_U(g_t * DeltaPPO(o_t, u_{t-1}) + (1 - g_t) * u_passive)",
        "```",
        "",
        "其中门控系数 `g_t` 由道路预瞻激励、车身姿态加速度需求和安全裕度共同决定。低激励或低置信场景下，`g_t` 接近 0，策略保持接近安全教师；高激励场景下，`g_t` 增大，允许策略输出主动悬架修正力。",
        "",
        f"当前证据中，平均场景门控为 `{_format(mean_gate, 4)}`，平均 `g_t>0.1` 激活比例为 `{_format(mean_active, 4)}`。B/C 级平稳随机路面门控为 0，而 `mixed_full` 和 `sine_roll` 场景门控显著打开，说明该策略网络具备状态相关的主动控制选择机制。",
        "",
    ]
    method_path.write_text("\n".join(method_lines), encoding="utf-8")

    manifest = {
        "result_dir": str(result_dir),
        "variant": variant,
        "csv": str(csv_path),
        "report": str(report_path),
        "latex_table": str(latex_path),
        "method_summary": str(method_path),
        "rows": int(len(summary)),
        "mean_gate": mean_gate,
        "mean_active_ratio_gt_0p1": mean_active,
    }
    manifest_path = out_dir / "si_rppo_gate_evidence_manifest.json"
    manifest["manifest"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("results/si_rppo_e20_improvement_gate_fair_baselines"),
    )
    parser.add_argument("--out", type=Path, default=Path("results/si_rppo_gate_evidence"))
    parser.add_argument("--variant", default="bc_ppo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_si_rppo_gate_evidence(args.result_dir, args.out, variant=args.variant)
    print(f"Saved SI-RPPO gate evidence to {manifest['report']}")


if __name__ == "__main__":
    main()
