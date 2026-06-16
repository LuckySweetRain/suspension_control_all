"""Build a requirement-by-requirement thesis completion audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _exists(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_dir():
        return True
    return path.stat().st_size > 0


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _status(done: bool, strong: bool = True) -> str:
    if done and strong:
        return "evidence_ready"
    if done:
        return "partial"
    return "missing"


def build_completion_audit(
    out_path: Path = Path("docs/thesis_completion_audit.md"),
    bp_summary_path: Path = Path("results/mpc_parameter_seed_sweep_formal_gate/mpc_parameter_seed_sweep_summary.json"),
    si_claim_path: Path = Path("results/si_rppo_e20_improvement_gate_fair_baselines/si_rppo_claim_report.json"),
    si_strategy_manifest_path: Path = Path("results/si_rppo_strategy_evidence/si_rppo_strategy_evidence_table_manifest.json"),
) -> dict[str, Any]:
    bp = _load_json(bp_summary_path)
    si = _load_json(si_claim_path)
    strategy = _load_json(si_strategy_manifest_path)

    spdf_artifacts = [
        Path("controllers"),
        Path("tests/test_core.py"),
        Path("docs/thesis_claim_matrix.md"),
    ]
    thesis_integration_artifacts = [
        Path("docs/thesis_chapter_draft_cn.tex"),
        Path("docs/thesis_latex_snippets.tex"),
        Path("docs/thesis_latex_integration_guide.md"),
        Path("docs/thesis_final_submission_checklist.md"),
        Path("docs/thesis_readiness_report.md"),
        Path("docs/thesis_standalone_preview.tex"),
        Path("docs/latex_preview/thesis_standalone_preview.pdf"),
    ]
    bp_artifacts = [
        bp_summary_path,
        Path("results/bp_nmps_schedule_evidence/bp_nmps_schedule_evidence.md"),
        Path("results/bp_nmps_preview_robustness/bp_nmps_preview_robustness.md"),
        Path("results/bp_nmps_evidence_table/bp_nmps_evidence_table.md"),
    ]
    si_artifacts = [
        si_claim_path,
        Path("results/si_rppo_gate_evidence/si_rppo_gate_evidence.md"),
        Path("results/si_rppo_preview_robustness/si_rppo_preview_robustness.md"),
        Path("results/si_rppo_strategy_evidence/si_rppo_strategy_evidence_table.md"),
    ]

    bp_ready = (
        bp.get("noninferiority_status") == "supported"
        and bp.get("improvement_status") == "supported"
        and float(bp.get("support_rate", 0.0)) >= 0.8
        and all(_exists(path) for path in bp_artifacts)
    )
    si_core_ready = si.get("core_status") == "supported" and si.get("status") == "ready"
    si_strategy_ready = int(strategy.get("row_count", 0)) >= 6 and _exists(Path(strategy.get("markdown", "")))
    si_ready = si_core_ready and si_strategy_ready and all(_exists(path) for path in si_artifacts)

    rows = [
        {
            "Objective": "1. SPDF method",
            "Status": _status(all(_exists(path) for path in spdf_artifacts)),
            "Evidence": "SPDF controller/test coverage and thesis claim matrix baseline entry",
            "CurrentProof": "Completed classical expert baseline for comparison.",
            "RemainingGap": "Move final SPDF baseline result/table into the final thesis chapter.",
        },
        {
            "Objective": "2. MPC method with RL parameter learning",
            "Status": _status(bp_ready, strong=bp_ready),
            "Evidence": "BP-NMPS formal seed sweep, schedule evidence, preview robustness, evidence table",
            "CurrentProof": (
                f"5-seed support {_fmt(bp.get('support_rate'))}, "
                f"mean score delta {_fmt(bp.get('mean_policy_delta_vs_fixed'), 6)}, "
                f"unsafe delta {_fmt(bp.get('mean_unsafe_delta_vs_fixed'))}"
            ),
            "RemainingGap": "Longer-budget seed sweep and broader held-out road/preview-error matrix before final broad claims.",
        },
        {
            "Objective": "3. RL strategy-network innovation",
            "Status": _status(si_ready, strong=si_ready),
            "Evidence": "SI-RPPO claim report, gate evidence, preview robustness, strategy evidence summary",
            "CurrentProof": (
                f"core status {si.get('core_status', 'missing')}; "
                f"strategy evidence rows {strategy.get('row_count', 'missing')}"
            ),
            "RemainingGap": "Learned/uncertainty-aware gate and longer harder-road SAC/TD3 comparisons for stronger CCF-A-level framing.",
        },
    ]

    overall = "partial"
    if rows[0]["Status"] == "evidence_ready" and rows[1]["Status"] == "evidence_ready" and rows[2]["Status"] == "evidence_ready":
        overall = "evidence_ready_with_remaining_thesis_work"

    thesis_integration_ready = all(_exists(path) for path in thesis_integration_artifacts)

    lines = [
        "# Thesis Completion Audit",
        "",
        "This audit maps the stated thesis objectives to current artifacts. It is intentionally conservative: evidence-ready does not mean the final thesis is fully written.",
        "",
        f"- Overall status: `{overall}`",
        "",
        "| Objective | Status | Evidence | Current proof | Remaining gap |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["Objective"],
                    row["Status"],
                    row["Evidence"],
                    row["CurrentProof"],
                    row["RemainingGap"],
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Defensible Thesis Position",
        "",
        "- The route should remain direct road-preview control, assuming already perceived road unevenness is available.",
        "- SPDF is a completed baseline, not the main contribution.",
        "- BP-NMPS is the MPC contribution: a baseline-preserving neural MPC parameter scheduler with deployment fallback.",
        "- SI-RPPO is the RL contribution: a projection-aware, safe-teacher-gated PPO policy class with preview-robust smooth deployment evidence.",
        "- The current evidence should not be phrased as universal superiority over SAC or all road/perception conditions.",
        "",
        "## Next Highest-Value Work",
        "",
        f"- Thesis integration artifacts ready: `{thesis_integration_ready}`.",
        "",
        "1. Follow `docs/thesis_latex_integration_guide.md` when moving `docs/thesis_chapter_draft_cn.tex` and `docs/thesis_latex_snippets.tex` into the final thesis source.",
        "2. Use `docs/thesis_final_submission_checklist.md` as the final manual checklist before submission.",
        "3. Run `python scripts/check_thesis_readiness.py` before migrating into the final school template.",
        "4. Ensure the final thesis preamble includes `\\usepackage{booktabs,graphicx}` before inserting the generated evidence tables.",
        "5. Use `docs/thesis_standalone_preview.tex` and `docs/latex_preview/thesis_standalone_preview.pdf` for local compile checks before editing the school template.",
        "6. Add longer-budget/harder-road SI-RPPO runs if time allows.",
        "7. Prototype or at least specify a learned/uncertainty-aware gate as future work or an extra ablation.",
        "8. Keep BP-NMPS claims conservative unless broader held-out road matrices are run.",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return {"out": str(out_path), "overall_status": overall, "rows": rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/thesis_completion_audit.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_completion_audit(out_path=args.out)
    print(f"Saved thesis completion audit to {audit['out']}")


if __name__ == "__main__":
    main()
