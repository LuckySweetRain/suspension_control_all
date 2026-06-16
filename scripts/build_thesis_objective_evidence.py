"""Build a machine-checkable thesis objective evidence matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _exists(path: Path) -> bool:
    return path.exists() and (path.is_dir() or path.stat().st_size > 0)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _artifact_status(paths: list[Path]) -> list[dict[str, object]]:
    return [{"path": str(path), "exists": _exists(path)} for path in paths]


def _all_exist(paths: list[Path]) -> bool:
    return all(_exists(path) for path in paths)


def build_thesis_objective_evidence(
    out_dir: Path = Path("docs"),
    bp_summary_path: Path = Path("results/mpc_parameter_seed_sweep_formal_gate/mpc_parameter_seed_sweep_summary.json"),
    si_claim_path: Path = Path("results/si_rppo_e20_improvement_gate_fair_baselines/si_rppo_claim_report.json"),
    si_strategy_manifest_path: Path = Path("results/si_rppo_strategy_evidence/si_rppo_strategy_evidence_table_manifest.json"),
) -> dict[str, object]:
    bp = _load_json(bp_summary_path)
    si = _load_json(si_claim_path)
    si_strategy = _load_json(si_strategy_manifest_path)

    spdf_artifacts = [
        Path("controllers/__init__.py"),
        Path("tests/test_core.py"),
        Path("docs/thesis_claim_matrix.md"),
        Path("docs/thesis_chapter_draft_cn.tex"),
    ]
    bp_artifacts = [
        bp_summary_path,
        Path("results/bp_nmps_schedule_evidence/bp_nmps_schedule_evidence.md"),
        Path("results/bp_nmps_preview_robustness/bp_nmps_preview_robustness.md"),
        Path("results/bp_nmps_evidence_table/bp_nmps_evidence_table.md"),
        Path("docs/thesis_latex_snippets.tex"),
    ]
    si_artifacts = [
        si_claim_path,
        Path("results/si_rppo_gate_evidence/si_rppo_gate_evidence.md"),
        Path("results/si_rppo_preview_robustness/si_rppo_preview_robustness.md"),
        Path("results/si_rppo_strategy_evidence/si_rppo_strategy_evidence_table.md"),
        Path("docs/si_rppo_ccfa_innovation_framing.md"),
        Path("docs/thesis_latex_snippets.tex"),
    ]

    bp_support_rate = float(bp.get("support_rate", 0.0) or 0.0)
    bp_ready = (
        _all_exist(bp_artifacts)
        and bp.get("noninferiority_status") == "supported"
        and bp.get("improvement_status") == "supported"
        and bp_support_rate >= 0.8
    )
    si_ready = (
        _all_exist(si_artifacts)
        and si.get("status") == "ready"
        and si.get("core_status") == "supported"
        and int(si_strategy.get("row_count", 0) or 0) >= 6
    )

    objectives = [
        {
            "id": "spdf",
            "objective": "SPDF方法，已经完成",
            "evidence_status": "evidence_ready" if _all_exist(spdf_artifacts) else "incomplete",
            "proof_scope": "classical baseline implemented and represented in thesis-facing materials",
            "key_result": "SPDF is positioned as the completed non-learning baseline.",
            "defensible_claim": "SPDF can be used as a completed traditional-control baseline, not as the main learning contribution.",
            "remaining_gap": "Final school thesis still needs the selected SPDF baseline result/table inserted in the template.",
            "artifacts": _artifact_status(spdf_artifacts),
        },
        {
            "id": "bp_nmps",
            "objective": "MPC方法，加入强化学习进行MPC参数学习",
            "evidence_status": "evidence_ready" if bp_ready else "incomplete",
            "proof_scope": "baseline-preserving neural MPC-lite parameter scheduling under held-out seed/probe evidence",
            "key_result": (
                f"support_rate={_fmt(bp.get('support_rate'), 3)}, "
                f"mean_policy_delta_vs_fixed={_fmt(bp.get('mean_policy_delta_vs_fixed'), 6)}, "
                f"mean_unsafe_delta_vs_fixed={_fmt(bp.get('mean_unsafe_delta_vs_fixed'), 3)}"
            ),
            "defensible_claim": "BP-NMPS is a conservative RL-assisted MPC parameter scheduler with deployment fallback.",
            "remaining_gap": "Do not claim full MPC replacement; broader long-budget road matrices remain future strengthening work.",
            "artifacts": _artifact_status(bp_artifacts),
        },
        {
            "id": "si_rppo",
            "objective": "强化学习策略网络设计，达到CCFA论文级别创新水平",
            "evidence_status": "evidence_ready" if si_ready else "incomplete",
            "proof_scope": "safe-teacher-gated, projection-aware PPO policy class with strategy-network evidence",
            "key_result": (
                f"claim_status={si.get('status', 'missing')}, "
                f"core_status={si.get('core_status', 'missing')}, "
                f"strategy_rows={si_strategy.get('row_count', 'missing')}"
            ),
            "defensible_claim": "SI-RPPO can be framed as a deployability-oriented policy-network contribution, with careful SAC tradeoff wording.",
            "remaining_gap": "CCF-A-level wording should stay method-level unless longer harder-road and learned-gate ablations are added.",
            "artifacts": _artifact_status(si_artifacts),
        },
    ]
    failed = [row for row in objectives if row["evidence_status"] != "evidence_ready"]
    status = "objective_evidence_ready" if not failed else "objective_evidence_incomplete"

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "thesis_objective_evidence.json"
    md_path = out_dir / "thesis_objective_evidence.md"
    payload: dict[str, object] = {
        "status": status,
        "ready_count": len(objectives) - len(failed),
        "failed_count": len(failed),
        "objectives": objectives,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Thesis Objective Evidence",
        "",
        "This matrix verifies the user-stated thesis objectives against current artifacts. It is stricter than a narrative claim matrix: every objective must point to concrete files and measured evidence.",
        "",
        f"- Status: `{status}`",
        f"- Evidence-ready objectives: `{payload['ready_count']}`",
        f"- Incomplete objectives: `{payload['failed_count']}`",
        "",
        "| Objective | Evidence status | Proof scope | Key result | Defensible claim | Remaining gap |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in objectives:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["objective"]),
                    str(row["evidence_status"]),
                    str(row["proof_scope"]),
                    str(row["key_result"]),
                    str(row["defensible_claim"]),
                    str(row["remaining_gap"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Artifact Checks", ""])
    for row in objectives:
        lines.append(f"### {row['id']}")
        for artifact in row["artifacts"]:
            mark = "pass" if artifact["exists"] else "fail"
            lines.append(f"- `{mark}` `{artifact['path']}`")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    payload["json"] = str(json_path)
    payload["markdown"] = str(md_path)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_thesis_objective_evidence(out_dir=args.out_dir)
    print(f"Saved thesis objective evidence to {report['markdown']} ({report['status']})")


if __name__ == "__main__":
    main()
