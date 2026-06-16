"""Build a thesis-facing claim matrix from current experiment artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return data


def _fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _status_label(status: Any) -> str:
    if not status:
        return "missing"
    return str(status)


def _bp_nmps_summary(data: dict[str, Any] | None) -> tuple[list[str], list[str]]:
    if data is None:
        return (
            [
                "| BP-NMPS MPC parameter learning | missing | n/a | n/a | n/a | "
                "Run `scripts/run_mpc_parameter_seed_sweep.py` formal protocol. |"
            ],
            ["- BP-NMPS evidence JSON was not found."],
        )

    seed_count = data.get("seed_count")
    support_rate = data.get("support_rate")
    mean_delta = data.get("mean_policy_delta_vs_fixed")
    unsafe_delta = data.get("mean_unsafe_delta_vs_fixed")
    gate_rate = None
    seeds = data.get("seeds") or []
    if seeds:
        gate_rate = sum(1 for row in seeds if row.get("deployment_gate_active")) / len(seeds)

    row = (
        "| BP-NMPS MPC parameter learning | "
        f"{_status_label(data.get('improvement_status') or data.get('status'))} | "
        f"{seed_count} seeds, support {_fmt_float(support_rate, 3)} | "
        f"score delta {_fmt_float(mean_delta, 6)} | "
        f"unsafe delta {_fmt_float(unsafe_delta, 3)}, gate rate {_fmt_float(gate_rate, 3)} | "
        "Claim as conservative adaptive MPC-parameter scheduling, not black-box MPC replacement. |"
    )
    notes = [
        "- BP-NMPS uses fixed MPC-lite as a deployable baseline and schedules MPC-lite gains through a neural parameter policy.",
        "- The deployment gate is part of the algorithm: failed calibration falls back to fixed MPC-lite on held-out evaluation.",
    ]
    return [row], notes


def _si_rppo_summary(data: dict[str, Any] | None) -> tuple[list[str], list[str]]:
    if data is None:
        return (
            [
                "| Safe-teacher gated SI-RPPO | missing | n/a | n/a | n/a | "
                "Run `scripts/run_si_rppo_ablation.py` and claim report generation. |"
            ],
            ["- SI-RPPO claim-report JSON was not found."],
        )

    projection_claim = None
    for comparison in data.get("comparisons", []):
        if comparison.get("name") == "projection_aware_imitation":
            projection_claim = comparison
            break

    metrics = (projection_claim or {}).get("metrics", {})
    ret = metrics.get("EpisodeReturn", {}).get("delta")
    unsafe = metrics.get("UnsafeSteps", {}).get("delta")
    action_delta = metrics.get("ActionDeltaRMS_N", {}).get("delta")
    projection_error = metrics.get("PolicyProjectionError", {}).get("delta")

    row = (
        "| Safe-teacher gated SI-RPPO policy network | "
        f"{_status_label((projection_claim or {}).get('status') or data.get('core_status'))} | "
        "3-seed core + fair full matrix | "
        f"return delta {_fmt_float(ret, 4)} | "
        f"unsafe delta {_fmt_float(unsafe, 4)}, action-delta delta {_fmt_float(action_delta, 4)}, "
        f"projection-error delta {_fmt_float(projection_error, 6)} | "
        "Claim as actuator-feasible online PPO improvement over PPO scratch; compare carefully against SAC. |"
    )
    notes = [
        "- SI-RPPO novelty is the policy class: offline imitation, delta action parameterization, projection-aware losses, and safe-teacher improvement gating.",
        "- The current broad superiority claim is not proven; SAC remains a strong return/safety baseline in short runs.",
    ]
    return [row], notes


def build_markdown(
    bp_nmps: dict[str, Any] | None,
    si_rppo: dict[str, Any] | None,
    delta_ppo: dict[str, Any] | None,
    bp_nmps_schedule_report: Path | None = None,
    bp_nmps_preview_robustness_report: Path | None = None,
    bp_nmps_preview_robustness_latex: Path | None = None,
    bp_nmps_preview_robustness_summary_cn: Path | None = None,
    si_rppo_gate_report: Path | None = None,
    si_rppo_gate_latex: Path | None = None,
    si_rppo_gate_method_summary: Path | None = None,
    si_rppo_innovation_framing: Path | None = None,
    si_rppo_preview_robustness_report: Path | None = None,
    si_rppo_preview_robustness_latex: Path | None = None,
    si_rppo_strategy_evidence_report: Path | None = None,
    si_rppo_strategy_evidence_latex: Path | None = None,
    thesis_method_draft: Path | None = None,
    thesis_chapter_draft: Path | None = None,
    thesis_chapter_latex: Path | None = None,
    thesis_standalone_preview: Path | None = None,
    thesis_standalone_preview_pdf: Path | None = None,
    direct_preview_plan: Path | None = None,
    thesis_workflow_index: Path | None = None,
    thesis_evidence_package: Path | None = None,
    thesis_latex_snippets: Path | None = None,
    thesis_latex_integration_guide: Path | None = None,
    thesis_defense_qa: Path | None = None,
    thesis_final_submission_checklist: Path | None = None,
    thesis_readiness_report: Path | None = None,
    thesis_export_package: Path | None = None,
    thesis_completion_audit: Path | None = None,
) -> str:
    bp_rows, bp_notes = _bp_nmps_summary(bp_nmps)
    si_rows, si_notes = _si_rppo_summary(si_rppo)
    delta_seed_count = (delta_ppo or {}).get("seed_count")
    delta_supported = (delta_ppo or {}).get("supported_seeds")
    schedule_status = (
        str(bp_nmps_schedule_report)
        if bp_nmps_schedule_report is not None and bp_nmps_schedule_report.exists()
        else "missing"
    )
    preview_robustness_status = (
        str(bp_nmps_preview_robustness_report)
        if bp_nmps_preview_robustness_report is not None and bp_nmps_preview_robustness_report.exists()
        else "missing"
    )
    preview_robustness_latex_status = (
        str(bp_nmps_preview_robustness_latex)
        if bp_nmps_preview_robustness_latex is not None and bp_nmps_preview_robustness_latex.exists()
        else "missing"
    )
    preview_robustness_summary_status = (
        str(bp_nmps_preview_robustness_summary_cn)
        if bp_nmps_preview_robustness_summary_cn is not None and bp_nmps_preview_robustness_summary_cn.exists()
        else "missing"
    )
    si_rppo_gate_status = (
        str(si_rppo_gate_report)
        if si_rppo_gate_report is not None and si_rppo_gate_report.exists()
        else "missing"
    )
    si_rppo_gate_latex_status = (
        str(si_rppo_gate_latex)
        if si_rppo_gate_latex is not None and si_rppo_gate_latex.exists()
        else "missing"
    )
    si_rppo_method_status = (
        str(si_rppo_gate_method_summary)
        if si_rppo_gate_method_summary is not None and si_rppo_gate_method_summary.exists()
        else "missing"
    )
    si_rppo_innovation_status = (
        str(si_rppo_innovation_framing)
        if si_rppo_innovation_framing is not None and si_rppo_innovation_framing.exists()
        else "missing"
    )
    si_rppo_preview_status = (
        str(si_rppo_preview_robustness_report)
        if si_rppo_preview_robustness_report is not None and si_rppo_preview_robustness_report.exists()
        else "missing"
    )
    si_rppo_preview_latex_status = (
        str(si_rppo_preview_robustness_latex)
        if si_rppo_preview_robustness_latex is not None and si_rppo_preview_robustness_latex.exists()
        else "missing"
    )
    si_rppo_strategy_status = (
        str(si_rppo_strategy_evidence_report)
        if si_rppo_strategy_evidence_report is not None and si_rppo_strategy_evidence_report.exists()
        else "missing"
    )
    si_rppo_strategy_latex_status = (
        str(si_rppo_strategy_evidence_latex)
        if si_rppo_strategy_evidence_latex is not None and si_rppo_strategy_evidence_latex.exists()
        else "missing"
    )
    thesis_method_status = (
        str(thesis_method_draft)
        if thesis_method_draft is not None and thesis_method_draft.exists()
        else "missing"
    )
    thesis_chapter_status = (
        str(thesis_chapter_draft)
        if thesis_chapter_draft is not None and thesis_chapter_draft.exists()
        else "missing"
    )
    thesis_chapter_latex_status = (
        str(thesis_chapter_latex)
        if thesis_chapter_latex is not None and thesis_chapter_latex.exists()
        else "missing"
    )
    standalone_preview_status = (
        str(thesis_standalone_preview)
        if thesis_standalone_preview is not None and thesis_standalone_preview.exists()
        else "missing"
    )
    standalone_preview_pdf_status = (
        str(thesis_standalone_preview_pdf)
        if thesis_standalone_preview_pdf is not None and thesis_standalone_preview_pdf.exists()
        else "missing"
    )
    direct_preview_status = (
        str(direct_preview_plan)
        if direct_preview_plan is not None and direct_preview_plan.exists()
        else "missing"
    )
    thesis_workflow_status = (
        str(thesis_workflow_index)
        if thesis_workflow_index is not None and thesis_workflow_index.exists()
        else "missing"
    )
    evidence_package_status = (
        str(thesis_evidence_package)
        if thesis_evidence_package is not None and thesis_evidence_package.exists()
        else "missing"
    )
    thesis_latex_status = (
        str(thesis_latex_snippets)
        if thesis_latex_snippets is not None and thesis_latex_snippets.exists()
        else "missing"
    )
    thesis_latex_integration_status = (
        str(thesis_latex_integration_guide)
        if thesis_latex_integration_guide is not None and thesis_latex_integration_guide.exists()
        else "missing"
    )
    thesis_defense_qa_status = (
        str(thesis_defense_qa)
        if thesis_defense_qa is not None and thesis_defense_qa.exists()
        else "missing"
    )
    thesis_final_checklist_status = (
        str(thesis_final_submission_checklist)
        if thesis_final_submission_checklist is not None and thesis_final_submission_checklist.exists()
        else "missing"
    )
    thesis_readiness_status = (
        str(thesis_readiness_report)
        if thesis_readiness_report is not None and thesis_readiness_report.exists()
        else "missing"
    )
    thesis_export_status = (
        str(thesis_export_package)
        if thesis_export_package is not None and thesis_export_package.exists()
        else "missing"
    )
    completion_audit_status = (
        str(thesis_completion_audit)
        if thesis_completion_audit is not None and thesis_completion_audit.exists()
        else "missing"
    )

    lines = [
        "# Thesis Claim Matrix",
        "",
        "This file is generated from experiment artifacts and is intended to keep thesis claims aligned with current evidence.",
        "",
        "## Objective Alignment",
        "",
        "| Thesis item | Current status | Evidence strength | Main metric movement | Safety evidence | Thesis wording boundary |",
        "| --- | --- | --- | --- | --- | --- |",
        "| SPDF baseline | completed | report/framework baseline | n/a | n/a | Treat as classical expert baseline, not the final learning contribution. |",
        *bp_rows,
        *si_rows,
        "",
        "## Current Defensible Claims",
        "",
        "- SPDF can be presented as a completed traditional controller baseline for comparison.",
        *bp_notes,
        *si_notes,
        "",
        "## Evidence Pointers",
        "",
        f"- BP-NMPS seed count: `{(bp_nmps or {}).get('seed_count', 'missing')}`.",
        f"- BP-NMPS non-inferiority status: `{(bp_nmps or {}).get('noninferiority_status', 'missing')}`.",
        f"- BP-NMPS improvement status: `{(bp_nmps or {}).get('improvement_status', 'missing')}`.",
        f"- BP-NMPS schedule evidence: `{schedule_status}`.",
        f"- BP-NMPS preview-robustness evidence: `{preview_robustness_status}`.",
        f"- BP-NMPS preview-robustness LaTeX table: `{preview_robustness_latex_status}`.",
        f"- BP-NMPS preview-robustness Chinese summary: `{preview_robustness_summary_status}`.",
        f"- Delta PPO repeated-seed support: `{delta_supported}/{delta_seed_count}`.",
        f"- SI-RPPO overall claim report status: `{(si_rppo or {}).get('status', 'missing')}`.",
        f"- SI-RPPO core status: `{(si_rppo or {}).get('core_status', 'missing')}`.",
        f"- SI-RPPO gate evidence: `{si_rppo_gate_status}`.",
        f"- SI-RPPO gate LaTeX table: `{si_rppo_gate_latex_status}`.",
        f"- SI-RPPO method summary: `{si_rppo_method_status}`.",
        f"- SI-RPPO innovation framing: `{si_rppo_innovation_status}`.",
        f"- SI-RPPO preview-robustness evidence: `{si_rppo_preview_status}`.",
        f"- SI-RPPO preview-robustness LaTeX table: `{si_rppo_preview_latex_status}`.",
        f"- SI-RPPO strategy-network evidence: `{si_rppo_strategy_status}`.",
        f"- SI-RPPO strategy-network LaTeX table: `{si_rppo_strategy_latex_status}`.",
        f"- Thesis method/experiment draft: `{thesis_method_status}`.",
        f"- Thesis chapter draft: `{thesis_chapter_status}`.",
        f"- Thesis chapter LaTeX draft: `{thesis_chapter_latex_status}`.",
        f"- Thesis standalone LaTeX preview: `{standalone_preview_status}`.",
        f"- Thesis standalone preview PDF: `{standalone_preview_pdf_status}`.",
        f"- Direct road-preview control plan: `{direct_preview_status}`.",
        f"- Thesis workflow index: `{thesis_workflow_status}`.",
        f"- Thesis evidence package: `{evidence_package_status}`.",
        f"- Thesis LaTeX snippets: `{thesis_latex_status}`.",
        f"- Thesis LaTeX integration guide: `{thesis_latex_integration_status}`.",
        f"- Thesis defense Q&A: `{thesis_defense_qa_status}`.",
        f"- Thesis final submission checklist: `{thesis_final_checklist_status}`.",
        f"- Thesis readiness report: `{thesis_readiness_status}`.",
        f"- Thesis export package: `{thesis_export_status}`.",
        f"- Thesis completion audit: `{completion_audit_status}`.",
        "",
        "## Next Experiments Before Final Thesis Claims",
        "",
        "1. Run longer-budget BP-NMPS seed sweeps to verify gate rate and mean improvement stability.",
        "2. Extend BP-NMPS preview robustness to more held-out road classes and stronger preview-corruption profiles.",
        "3. Strengthen SI-RPPO with learned or uncertainty-aware gate evidence, especially against SAC under harder road cases where passive-like policies are insufficient.",
        "4. Evaluate SI-RPPO under preview-noise, delay, truncation, and missing-segment tests.",
        "5. Move selected LaTeX tables into the final thesis document.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bp-nmps-summary",
        type=Path,
        default=Path("results/mpc_parameter_seed_sweep_formal_gate/mpc_parameter_seed_sweep_summary.json"),
    )
    parser.add_argument(
        "--si-rppo-report",
        type=Path,
        default=Path("results/si_rppo_e20_improvement_gate_fair_baselines/si_rppo_claim_report.json"),
    )
    parser.add_argument(
        "--delta-ppo-evidence",
        type=Path,
        default=Path("results/delta_ppo_evidence_table/delta_ppo_evidence_table.json"),
    )
    parser.add_argument(
        "--bp-nmps-schedule-report",
        type=Path,
        default=Path("results/bp_nmps_schedule_evidence/bp_nmps_schedule_evidence.md"),
    )
    parser.add_argument(
        "--bp-nmps-preview-robustness-report",
        type=Path,
        default=Path("results/bp_nmps_preview_robustness/bp_nmps_preview_robustness.md"),
    )
    parser.add_argument(
        "--bp-nmps-preview-robustness-latex",
        type=Path,
        default=Path("results/bp_nmps_preview_robustness/bp_nmps_preview_robustness.tex"),
    )
    parser.add_argument(
        "--bp-nmps-preview-robustness-summary-cn",
        type=Path,
        default=Path("results/bp_nmps_preview_robustness/bp_nmps_preview_robustness_summary_cn.md"),
    )
    parser.add_argument(
        "--si-rppo-gate-report",
        type=Path,
        default=Path("results/si_rppo_gate_evidence/si_rppo_gate_evidence.md"),
    )
    parser.add_argument(
        "--si-rppo-gate-latex",
        type=Path,
        default=Path("results/si_rppo_gate_evidence/si_rppo_gate_evidence.tex"),
    )
    parser.add_argument(
        "--si-rppo-gate-method-summary",
        type=Path,
        default=Path("results/si_rppo_gate_evidence/si_rppo_gate_method_summary.md"),
    )
    parser.add_argument(
        "--si-rppo-innovation-framing",
        type=Path,
        default=Path("docs/si_rppo_ccfa_innovation_framing.md"),
    )
    parser.add_argument(
        "--si-rppo-preview-robustness-report",
        type=Path,
        default=Path("results/si_rppo_preview_robustness/si_rppo_preview_robustness.md"),
    )
    parser.add_argument(
        "--si-rppo-preview-robustness-latex",
        type=Path,
        default=Path("results/si_rppo_preview_robustness/si_rppo_preview_robustness.tex"),
    )
    parser.add_argument(
        "--si-rppo-strategy-evidence-report",
        type=Path,
        default=Path("results/si_rppo_strategy_evidence/si_rppo_strategy_evidence_table.md"),
    )
    parser.add_argument(
        "--si-rppo-strategy-evidence-latex",
        type=Path,
        default=Path("results/si_rppo_strategy_evidence/si_rppo_strategy_evidence_table.tex"),
    )
    parser.add_argument(
        "--thesis-method-draft",
        type=Path,
        default=Path("docs/thesis_method_experiment_draft.md"),
    )
    parser.add_argument(
        "--thesis-chapter-draft",
        type=Path,
        default=Path("docs/thesis_chapter_draft_cn.md"),
    )
    parser.add_argument(
        "--thesis-chapter-latex",
        type=Path,
        default=Path("docs/thesis_chapter_draft_cn.tex"),
    )
    parser.add_argument(
        "--thesis-standalone-preview",
        type=Path,
        default=Path("docs/thesis_standalone_preview.tex"),
    )
    parser.add_argument(
        "--thesis-standalone-preview-pdf",
        type=Path,
        default=Path("docs/latex_preview/thesis_standalone_preview.pdf"),
    )
    parser.add_argument(
        "--direct-preview-plan",
        type=Path,
        default=Path("docs/direct_road_preview_control_plan.md"),
    )
    parser.add_argument(
        "--thesis-workflow-index",
        type=Path,
        default=Path("docs/thesis_workflow_index.md"),
    )
    parser.add_argument(
        "--thesis-evidence-package",
        type=Path,
        default=Path("docs/thesis_evidence_package.md"),
    )
    parser.add_argument(
        "--thesis-latex-snippets",
        type=Path,
        default=Path("docs/thesis_latex_snippets.tex"),
    )
    parser.add_argument(
        "--thesis-latex-integration-guide",
        type=Path,
        default=Path("docs/thesis_latex_integration_guide.md"),
    )
    parser.add_argument(
        "--thesis-defense-qa",
        type=Path,
        default=Path("docs/thesis_defense_qa.md"),
    )
    parser.add_argument(
        "--thesis-final-submission-checklist",
        type=Path,
        default=Path("docs/thesis_final_submission_checklist.md"),
    )
    parser.add_argument(
        "--thesis-readiness-report",
        type=Path,
        default=Path("docs/thesis_readiness_report.md"),
    )
    parser.add_argument(
        "--thesis-export-package",
        type=Path,
        default=Path("docs/thesis_export_package/README.md"),
    )
    parser.add_argument(
        "--thesis-completion-audit",
        type=Path,
        default=Path("docs/thesis_completion_audit.md"),
    )
    parser.add_argument("--out", type=Path, default=Path("docs/thesis_claim_matrix.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bp_nmps = _load_json(args.bp_nmps_summary)
    si_rppo = _load_json(args.si_rppo_report)
    delta_ppo = _load_json(args.delta_ppo_evidence)
    markdown = build_markdown(
        bp_nmps=bp_nmps,
        si_rppo=si_rppo,
        delta_ppo=delta_ppo,
        bp_nmps_schedule_report=args.bp_nmps_schedule_report,
        bp_nmps_preview_robustness_report=args.bp_nmps_preview_robustness_report,
        bp_nmps_preview_robustness_latex=args.bp_nmps_preview_robustness_latex,
        bp_nmps_preview_robustness_summary_cn=args.bp_nmps_preview_robustness_summary_cn,
        si_rppo_gate_report=args.si_rppo_gate_report,
        si_rppo_gate_latex=args.si_rppo_gate_latex,
        si_rppo_gate_method_summary=args.si_rppo_gate_method_summary,
        si_rppo_innovation_framing=args.si_rppo_innovation_framing,
        si_rppo_preview_robustness_report=args.si_rppo_preview_robustness_report,
        si_rppo_preview_robustness_latex=args.si_rppo_preview_robustness_latex,
        si_rppo_strategy_evidence_report=args.si_rppo_strategy_evidence_report,
        si_rppo_strategy_evidence_latex=args.si_rppo_strategy_evidence_latex,
        thesis_method_draft=args.thesis_method_draft,
        thesis_chapter_draft=args.thesis_chapter_draft,
        thesis_chapter_latex=args.thesis_chapter_latex,
        thesis_standalone_preview=args.thesis_standalone_preview,
        thesis_standalone_preview_pdf=args.thesis_standalone_preview_pdf,
        direct_preview_plan=args.direct_preview_plan,
        thesis_workflow_index=args.thesis_workflow_index,
        thesis_evidence_package=args.thesis_evidence_package,
        thesis_latex_snippets=args.thesis_latex_snippets,
        thesis_latex_integration_guide=args.thesis_latex_integration_guide,
        thesis_defense_qa=args.thesis_defense_qa,
        thesis_final_submission_checklist=args.thesis_final_submission_checklist,
        thesis_readiness_report=args.thesis_readiness_report,
        thesis_export_package=args.thesis_export_package,
        thesis_completion_audit=args.thesis_completion_audit,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown, encoding="utf-8")
    print(f"Saved thesis claim matrix to {args.out}")


if __name__ == "__main__":
    main()
