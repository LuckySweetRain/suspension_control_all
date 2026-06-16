"""Check thesis-readiness artifacts for the direct road-preview route."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    status: str
    detail: str


REQUIRED_FILES = (
    Path("docs/thesis_chapter_draft_cn.tex"),
    Path("docs/thesis_latex_snippets.tex"),
    Path("docs/thesis_standalone_preview.tex"),
    Path("docs/latex_preview/thesis_standalone_preview.pdf"),
    Path("docs/thesis_latex_integration_guide.md"),
    Path("docs/thesis_final_submission_checklist.md"),
    Path("docs/thesis_defense_qa.md"),
    Path("docs/si_rppo_ccfa_innovation_framing.md"),
    Path("docs/thesis_workflow_index.md"),
    Path("docs/thesis_claim_matrix.md"),
    Path("docs/thesis_completion_audit.md"),
    Path("docs/thesis_objective_evidence.md"),
    Path("docs/thesis_objective_evidence.json"),
    Path("docs/thesis_claim_boundary_audit.md"),
    Path("docs/thesis_claim_boundary_audit.json"),
    Path("docs/thesis_export_package/manifest.json"),
    Path("docs/thesis_export_package/README.md"),
    Path("docs/thesis_export_package.zip"),
    Path("docs/thesis_export_package.zip.sha256"),
)

REQUIRED_TABLE_LABELS = (
    "tab:bp_nmps_seed_sweep",
    "tab:bp_nmps_evidence",
    "tab:bp_nmps_preview_robustness",
    "tab:si_rppo_gate_evidence",
    "tab:si_rppo_preview_robustness",
    "tab:si_rppo_strategy_evidence",
)

REQUIRED_BOUNDARY_PHRASES = (
    "already perceived road unevenness",
    "Do not claim that PreScan",
    "Do not claim that Chrono/VLA",
    "Do not state that SI-RPPO is universally superior to SAC",
)


def _exists_nonempty(path: Path) -> bool:
    return path.exists() and (path.is_dir() or path.stat().st_size > 0)


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _check_required_files(paths: Iterable[Path]) -> list[ReadinessCheck]:
    checks: list[ReadinessCheck] = []
    for path in paths:
        checks.append(
            ReadinessCheck(
                name=f"file:{path.as_posix()}",
                status="pass" if _exists_nonempty(path) else "fail",
                detail=str(path),
            )
        )
    return checks


def _check_labels(snippets_path: Path = Path("docs/thesis_latex_snippets.tex")) -> list[ReadinessCheck]:
    text = _read_text(snippets_path)
    return [
        ReadinessCheck(
            name=f"latex_label:{label}",
            status="pass" if label in text else "fail",
            detail=str(snippets_path),
        )
        for label in REQUIRED_TABLE_LABELS
    ]


def _check_boundary_phrases(
    checklist_path: Path = Path("docs/thesis_final_submission_checklist.md"),
) -> list[ReadinessCheck]:
    text = _read_text(checklist_path)
    return [
        ReadinessCheck(
            name=f"claim_boundary:{phrase}",
            status="pass" if phrase in text else "fail",
            detail=str(checklist_path),
        )
        for phrase in REQUIRED_BOUNDARY_PHRASES
    ]


def _check_latex_packages(
    snippets_path: Path = Path("docs/thesis_latex_snippets.tex"),
    preview_path: Path = Path("docs/thesis_standalone_preview.tex"),
) -> list[ReadinessCheck]:
    snippets = _read_text(snippets_path)
    preview = _read_text(preview_path)
    return [
        ReadinessCheck(
            name="latex_requirement:booktabs_graphicx",
            status="pass" if "\\usepackage{booktabs,graphicx}" in snippets else "fail",
            detail=str(snippets_path),
        ),
        ReadinessCheck(
            name="latex_preview_package:graphicx",
            status="pass" if "\\usepackage{graphicx}" in preview else "fail",
            detail=str(preview_path),
        ),
        ReadinessCheck(
            name="latex_table_fit:resizebox",
            status="pass" if "\\resizebox{\\textwidth}{!}{%" in snippets else "fail",
            detail=str(snippets_path),
        ),
    ]


def _check_claim_boundary_audit(
    audit_path: Path = Path("docs/thesis_claim_boundary_audit.json"),
) -> list[ReadinessCheck]:
    if not audit_path.exists():
        return [ReadinessCheck(name="claim_boundary_audit:status", status="fail", detail=str(audit_path))]
    try:
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [ReadinessCheck(name="claim_boundary_audit:status", status="fail", detail=str(audit_path))]
    return [
        ReadinessCheck(
            name="claim_boundary_audit:status",
            status="pass" if payload.get("status") == "claim_boundaries_pass" else "fail",
            detail=str(audit_path),
        )
    ]


def build_thesis_readiness_report(
    out_dir: Path = Path("docs"),
) -> dict[str, object]:
    checks: list[ReadinessCheck] = []
    checks.extend(_check_required_files(REQUIRED_FILES))
    checks.extend(_check_labels())
    checks.extend(_check_boundary_phrases())
    checks.extend(_check_latex_packages())
    checks.extend(_check_claim_boundary_audit())

    failed = [check for check in checks if check.status != "pass"]
    status = "ready_for_template_migration" if not failed else "incomplete"

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "thesis_readiness_report.json"
    md_path = out_dir / "thesis_readiness_report.md"
    payload = {
        "status": status,
        "passed": len(checks) - len(failed),
        "failed": len(failed),
        "checks": [check.__dict__ for check in checks],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Thesis Readiness Report",
        "",
        f"- Status: `{status}`",
        f"- Passed checks: `{payload['passed']}`",
        f"- Failed checks: `{payload['failed']}`",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for check in checks:
        lines.append(f"| {check.name} | {check.status} | `{check.detail}` |")
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
    report = build_thesis_readiness_report(out_dir=args.out_dir)
    print(f"Saved thesis readiness report to {report['markdown']}")


if __name__ == "__main__":
    main()
