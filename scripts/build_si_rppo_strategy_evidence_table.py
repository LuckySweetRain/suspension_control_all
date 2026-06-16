"""Build a thesis-facing SI-RPPO strategy-network evidence table."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(numeric):
        return "n/a"
    return f"{numeric:.{digits}f}"


def _metric_delta(comparison: dict[str, Any], metric: str) -> float | None:
    value = comparison.get("metrics", {}).get(metric, {}).get("delta")
    if value is None:
        return None
    return float(value)


def _comparison_by_name(report: dict[str, Any], name: str) -> dict[str, Any] | None:
    for item in report.get("comparisons", []):
        if item.get("name") == name:
            return item
    return None


def _preview_pairwise(pairwise: pd.DataFrame, label: str) -> pd.Series | None:
    if pairwise.empty or "Comparison" not in pairwise:
        return None
    rows = pairwise[pairwise["Comparison"] == label]
    if rows.empty:
        return None
    return rows.iloc[0]


def _gate_summary(gate: pd.DataFrame) -> dict[str, float]:
    if gate.empty:
        return {"mean_gate": float("nan"), "active_gt_0p1": float("nan"), "max_gate": float("nan")}
    mean_gate_col = "PolicyImprovementGate" if "PolicyImprovementGate" in gate else "GateMeanTrajectory"
    active_col = "GateActiveRatioGt0p1"
    max_col = "GateMax"
    return {
        "mean_gate": float(gate[mean_gate_col].mean()) if mean_gate_col in gate else float("nan"),
        "active_gt_0p1": float(gate[active_col].mean()) if active_col in gate else float("nan"),
        "max_gate": float(gate[max_col].max()) if max_col in gate else float("nan"),
    }


def _latex_escape(value: Any) -> str:
    return str(value).replace("_", r"\_")


def build_si_rppo_strategy_evidence_table(
    claim_report_path: Path,
    gate_csv_path: Path,
    preview_pairwise_csv_path: Path,
    out_dir: Path,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with claim_report_path.open("r", encoding="utf-8") as f:
        claim_report = json.load(f)
    gate = pd.read_csv(gate_csv_path) if gate_csv_path.exists() else pd.DataFrame()
    preview = pd.read_csv(preview_pairwise_csv_path) if preview_pairwise_csv_path.exists() else pd.DataFrame()

    rows: list[dict[str, str]] = []

    projection = _comparison_by_name(claim_report, "projection_aware_imitation")
    if projection is not None:
        rows.append(
            {
                "Component": "Projection-aware imitation PPO",
                "Evidence": "BC-PPO vs PPO scratch",
                "MainResult": (
                    f"return {_fmt(_metric_delta(projection, 'EpisodeReturn'), 1)}, "
                    f"unsafe {_fmt(_metric_delta(projection, 'UnsafeSteps'), 1)}, "
                    f"action {_fmt(_metric_delta(projection, 'ActionDeltaRMS_N'), 1)}, "
                    f"projection {_fmt(_metric_delta(projection, 'PolicyProjectionError'), 3)}"
                ),
                "Status": str(projection.get("status", "missing")),
                "ThesisUse": "main PPO deployability claim",
                "Boundary": "compared to PPO scratch, not universal RL superiority",
            }
        )

    imitation = _comparison_by_name(claim_report, "imitation_initialization")
    if imitation is not None:
        rows.append(
            {
                "Component": "Offline imitation initialization",
                "Evidence": "BC initialization vs random PPO",
                "MainResult": (
                    f"return {_fmt(_metric_delta(imitation, 'EpisodeReturn'), 1)}, "
                    f"action {_fmt(_metric_delta(imitation, 'ActionDeltaRMS_N'), 1)}"
                ),
                "Status": str(imitation.get("status", "missing")),
                "ThesisUse": "early policy stabilization",
                "Boundary": "supports initialization, not final optimality alone",
            }
        )

    gate_stats = _gate_summary(gate)
    rows.append(
        {
            "Component": "Safe-teacher improvement gate",
            "Evidence": "gate activation by road scenario",
            "MainResult": (
                f"mean gate {_fmt(gate_stats['mean_gate'], 3)}, "
                f"active>0.1 {_fmt(gate_stats['active_gt_0p1'], 3)}, "
                f"max {_fmt(gate_stats['max_gate'], 3)}"
            ),
            "Status": "supported" if not math.isnan(gate_stats["mean_gate"]) else "missing",
            "ThesisUse": "state-dependent active-force selection",
            "Boundary": "current gate is hand-designed, not yet learned",
        }
    )

    for comparison, status, use, boundary in [
        (
            "SI-RPPO vs PPO scratch",
            "supported",
            "preview-corruption robustness over PPO",
            "same trained checkpoints and configured preview cases",
        ),
        (
            "SI-RPPO vs SAC",
            "tradeoff",
            "smoothness/projection advantage against strong SAC baseline",
            "return is not superior to SAC in current evidence",
        ),
        (
            "SI-RPPO vs TD3",
            "supported",
            "preview-corruption robustness over TD3",
            "TD3 checkpoint is same-budget baseline",
        ),
    ]:
        row = _preview_pairwise(preview, comparison)
        if row is None:
            continue
        rows.append(
            {
                "Component": "Preview-robust policy deployment",
                "Evidence": comparison,
                "MainResult": (
                    f"return {_fmt(row.get('EpisodeReturnDelta'), 1)}, "
                    f"unsafe {_fmt(row.get('UnsafeStepsDelta'), 1)}, "
                    f"action {_fmt(row.get('ActionDeltaRMS_NDelta'), 1)}, "
                    f"projection {_fmt(row.get('PolicyProjectionErrorDelta'), 3)}"
                ),
                "Status": status,
                "ThesisUse": use,
                "Boundary": boundary,
            }
        )

    residual = _comparison_by_name(claim_report, "residual_prior_structure")
    if residual is not None:
        rows.append(
            {
                "Component": "Residual prior branch",
                "Evidence": "residual BC-PPO vs direct BC-PPO",
                "MainResult": f"return {_fmt(_metric_delta(residual, 'EpisodeReturn'), 1)}, unsafe {_fmt(_metric_delta(residual, 'UnsafeSteps'), 1)}",
                "Status": str(residual.get("status", "missing")),
                "ThesisUse": "negative ablation / not main claim",
                "Boundary": "do not claim residual branch as current strongest policy",
            }
        )

    table = pd.DataFrame(rows)
    csv_path = out_dir / "si_rppo_strategy_evidence_table.csv"
    md_path = out_dir / "si_rppo_strategy_evidence_table.md"
    latex_path = out_dir / "si_rppo_strategy_evidence_table.tex"
    manifest_path = out_dir / "si_rppo_strategy_evidence_table_manifest.json"
    table.to_csv(csv_path, index=False)

    md_lines = [
        "# SI-RPPO Strategy-Network Evidence Table",
        "",
        "This table consolidates the evidence for the strategy-network design claim.",
        "",
        table.to_markdown(index=False) if not table.empty else "No rows generated.",
        "",
        "## Wording Boundary",
        "",
        "- The defensible claim is policy-class deployability: imitation initialization, projection-aware losses, safe-teacher gating, and preview-robust smooth control.",
        "- Do not claim universal superiority over SAC; current evidence shows a return tradeoff against SAC but smoother actions and much lower projection error.",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{SI-RPPO strategy-network evidence summary.}",
        r"\label{tab:si_rppo_strategy_evidence}",
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Component & Evidence & Main result & Status \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        latex_lines.append(
            f"{_latex_escape(row['Component'])} & "
            f"{_latex_escape(row['Evidence'])} & "
            f"{_latex_escape(row['MainResult'])} & "
            f"{_latex_escape(row['Status'])} \\\\"
        )
    latex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    latex_path.write_text("\n".join(latex_lines), encoding="utf-8")

    manifest = {
        "csv": str(csv_path),
        "markdown": str(md_path),
        "latex": str(latex_path),
        "row_count": int(len(table)),
        "claim_report": str(claim_report_path),
        "gate_csv": str(gate_csv_path),
        "preview_pairwise_csv": str(preview_pairwise_csv_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--claim-report",
        type=Path,
        default=Path("results/si_rppo_e20_improvement_gate_fair_baselines/si_rppo_claim_report.json"),
    )
    parser.add_argument(
        "--gate-csv",
        type=Path,
        default=Path("results/si_rppo_gate_evidence/si_rppo_gate_evidence.csv"),
    )
    parser.add_argument(
        "--preview-pairwise-csv",
        type=Path,
        default=Path("results/si_rppo_preview_robustness/si_rppo_preview_robustness_pairwise.csv"),
    )
    parser.add_argument("--out", type=Path, default=Path("results/si_rppo_strategy_evidence"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_si_rppo_strategy_evidence_table(
        claim_report_path=args.claim_report,
        gate_csv_path=args.gate_csv,
        preview_pairwise_csv_path=args.preview_pairwise_csv,
        out_dir=args.out,
    )
    print(f"Saved SI-RPPO strategy evidence to {manifest['markdown']}")


if __name__ == "__main__":
    main()
