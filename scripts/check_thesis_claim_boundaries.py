"""Audit thesis-facing files for unsupported overclaims."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_TARGETS = (
    Path("docs/thesis_chapter_draft_cn.tex"),
    Path("docs/thesis_chapter_draft_cn.md"),
    Path("docs/thesis_defense_qa.md"),
    Path("docs/si_rppo_ccfa_innovation_framing.md"),
    Path("docs/thesis_final_submission_checklist.md"),
)

GUARD_TERMS = (
    "不应",
    "不应过度声称",
    "不声称",
    "不是",
    "不将",
    "不要",
    "不能",
    "not",
    "do not",
    "do not claim",
    "don't",
    "boundary",
    "do not say",
    "future work",
    "future",
    "belongs in future",
    "treated as",
    "tradeoff",
)


@dataclass(frozen=True)
class BoundaryPattern:
    name: str
    pattern: str
    reason: str


BOUNDARY_PATTERNS = (
    BoundaryPattern(
        name="prescan_core_route",
        pattern=r"PreScan.*(?:核心|core|main route|implemented)",
        reason="PreScan road-contour extraction must not be claimed as the implemented core route.",
    ),
    BoundaryPattern(
        name="chrono_vla_platform",
        pattern=r"(?:Chrono|VLA).*(?:thesis platform|核心|main thesis|main contribution|平台作为核心)",
        reason="Chrono/VLA must remain future work, not the thesis platform.",
    ),
    BoundaryPattern(
        name="vla_completed_platform",
        pattern=r"(?:VLA|视觉--语言--动作|vision-language-action).*(?:完成|implemented|platform|平台)",
        reason="The current thesis does not complete an end-to-end VLA active-suspension platform.",
    ),
    BoundaryPattern(
        name="bp_nmps_replaces_mpc",
        pattern=r"(?:BP-NMPS|神经策略).*(?:全面替代\s*MPC|fully replaces MPC|full MPC replacement|replace MPC)",
        reason="BP-NMPS is MPC parameter scheduling with fallback, not full MPC replacement.",
    ),
    BoundaryPattern(
        name="si_rppo_universal_superiority",
        pattern=r"SI-RPPO.*(?:全面优于\s*SAC|universally superior to SAC|beats SAC|全面最优|universally better)",
        reason="SI-RPPO has a SAC tradeoff in the current matrix, not universal superiority.",
    ),
)

REQUIRED_SAFE_PHRASES = (
    "已感知路面不平度",
    "PreScan",
    "BP-NMPS",
    "SI-RPPO",
    "SAC",
)


def check_thesis_claim_boundaries(
    out_dir: Path = Path("docs"),
    targets: tuple[Path, ...] = DEFAULT_TARGETS,
) -> dict[str, object]:
    checked_files: list[dict[str, object]] = []
    violations: list[dict[str, object]] = []
    guarded_mentions: list[dict[str, object]] = []
    corpus = ""
    for target in targets:
        text = target.read_text(encoding="utf-8") if target.exists() else ""
        corpus += "\n" + text
        lines = text.splitlines()
        checked_files.append({"path": str(target), "exists": target.exists(), "line_count": len(lines)})
        for index, line in enumerate(lines, start=1):
            context = "\n".join(lines[max(0, index - 6) : index]).lower()
            for boundary in BOUNDARY_PATTERNS:
                if not re.search(boundary.pattern, line, flags=re.IGNORECASE):
                    continue
                record = {
                    "file": str(target),
                    "line": index,
                    "pattern": boundary.name,
                    "text": line.strip(),
                    "reason": boundary.reason,
                }
                if _has_guard(context):
                    guarded_mentions.append(record)
                else:
                    violations.append(record)

    required_phrase_checks = [
        {"phrase": phrase, "present": phrase in corpus}
        for phrase in REQUIRED_SAFE_PHRASES
    ]
    missing_required = [row for row in required_phrase_checks if not row["present"]]
    missing_files = [row for row in checked_files if not row["exists"]]
    status = "claim_boundaries_pass" if not violations and not missing_required and not missing_files else "claim_boundaries_fail"

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "thesis_claim_boundary_audit.json"
    md_path = out_dir / "thesis_claim_boundary_audit.md"
    payload: dict[str, object] = {
        "status": status,
        "violation_count": len(violations),
        "guarded_mention_count": len(guarded_mentions),
        "missing_required_count": len(missing_required),
        "checked_files": checked_files,
        "required_phrase_checks": required_phrase_checks,
        "violations": violations,
        "guarded_mentions": guarded_mentions,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(md_path, payload)
    payload["json"] = str(json_path)
    payload["markdown"] = str(md_path)
    return payload


def _has_guard(context: str) -> bool:
    return any(term in context for term in GUARD_TERMS)


def _write_markdown(path: Path, payload: dict[str, object]) -> None:
    lines = [
        "# Thesis Claim Boundary Audit",
        "",
        "This audit scans thesis-facing drafts for unsupported overclaims. Guarded mentions inside boundary or do-not-say contexts are allowed.",
        "",
        f"- Status: `{payload['status']}`",
        f"- Violations: `{payload['violation_count']}`",
        f"- Guarded mentions: `{payload['guarded_mention_count']}`",
        f"- Missing required phrases: `{payload['missing_required_count']}`",
        "",
        "## Required Route Phrases",
        "",
        "| Phrase | Present |",
        "| --- | --- |",
    ]
    for row in payload["required_phrase_checks"]:  # type: ignore[index]
        lines.append(f"| {row['phrase']} | {row['present']} |")
    lines.extend(["", "## Violations", ""])
    violations = payload["violations"]  # type: ignore[index]
    if violations:
        for row in violations:
            lines.append(f"- `{row['file']}:{row['line']}` `{row['pattern']}`: {row['text']}")
    else:
        lines.append("- None.")
    lines.extend(["", "## Guarded Mentions", ""])
    for row in payload["guarded_mentions"]:  # type: ignore[index]
        lines.append(f"- `{row['file']}:{row['line']}` `{row['pattern']}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = check_thesis_claim_boundaries(out_dir=args.out_dir)
    print(f"Saved thesis claim boundary audit to {report['markdown']} ({report['status']})")


if __name__ == "__main__":
    main()
