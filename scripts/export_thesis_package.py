"""Export thesis-ready artifacts into one handoff directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExportItem:
    source: Path
    target: Path
    required: bool = True


DEFAULT_EXPORT_ITEMS = (
    ExportItem(Path("docs/thesis_chapter_draft_cn.tex"), Path("latex/thesis_chapter_draft_cn.tex")),
    ExportItem(Path("docs/thesis_latex_snippets.tex"), Path("latex/thesis_latex_snippets.tex")),
    ExportItem(Path("docs/thesis_standalone_preview.tex"), Path("latex/thesis_standalone_preview.tex")),
    ExportItem(Path("docs/latex_preview/thesis_standalone_preview.pdf"), Path("latex/thesis_standalone_preview.pdf")),
    ExportItem(Path("docs/thesis_latex_integration_guide.md"), Path("guides/thesis_latex_integration_guide.md")),
    ExportItem(Path("docs/thesis_workflow_index.md"), Path("guides/thesis_workflow_index.md")),
    ExportItem(Path("docs/thesis_final_submission_checklist.md"), Path("guides/thesis_final_submission_checklist.md")),
    ExportItem(Path("docs/thesis_defense_qa.md"), Path("guides/thesis_defense_qa.md")),
    ExportItem(Path("docs/si_rppo_ccfa_innovation_framing.md"), Path("guides/si_rppo_ccfa_innovation_framing.md")),
    ExportItem(Path("docs/thesis_evidence_package.md"), Path("evidence/thesis_evidence_package.md")),
    ExportItem(Path("docs/thesis_claim_matrix.md"), Path("evidence/thesis_claim_matrix.md")),
    ExportItem(Path("docs/thesis_completion_audit.md"), Path("evidence/thesis_completion_audit.md")),
    ExportItem(Path("docs/thesis_objective_evidence.md"), Path("evidence/thesis_objective_evidence.md")),
    ExportItem(Path("docs/thesis_objective_evidence.json"), Path("evidence/thesis_objective_evidence.json")),
    ExportItem(Path("docs/thesis_claim_boundary_audit.md"), Path("evidence/thesis_claim_boundary_audit.md")),
    ExportItem(Path("docs/thesis_claim_boundary_audit.json"), Path("evidence/thesis_claim_boundary_audit.json")),
    ExportItem(Path("docs/thesis_readiness_report.md"), Path("evidence/thesis_readiness_report.md")),
    ExportItem(Path("docs/thesis_readiness_report.json"), Path("evidence/thesis_readiness_report.json")),
)


def export_thesis_package(
    out_dir: Path = Path("docs/thesis_export_package"),
    items: tuple[ExportItem, ...] = DEFAULT_EXPORT_ITEMS,
) -> dict[str, object]:
    copied: list[dict[str, str]] = []
    missing: list[str] = []
    for item in items:
        if not item.source.exists():
            if item.required:
                missing.append(str(item.source))
            continue
        destination = out_dir / item.target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.source, destination)
        copied.append({"source": str(item.source), "target": str(destination)})

    status = "exported" if not missing else "incomplete"
    manifest = {
        "status": status,
        "out_dir": str(out_dir),
        "copied_count": len(copied),
        "missing": missing,
        "copied": copied,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    readme_path = out_dir / "README.md"
    lines = [
        "# Thesis Export Package",
        "",
        "This directory contains the thesis-ready artifacts for migration into the final university thesis template.",
        "",
        f"- Status: `{status}`",
        f"- Copied files: `{len(copied)}`",
        f"- Missing required files: `{len(missing)}`",
        "",
        "## Suggested Use",
        "",
        "1. Copy `latex/thesis_chapter_draft_cn.tex` into the final thesis source.",
        "2. Insert or input selected tables from `latex/thesis_latex_snippets.tex`.",
        "3. Add `\\usepackage{booktabs,graphicx}` to the final thesis preamble.",
        "4. Use `guides/thesis_final_submission_checklist.md` before submission.",
        "5. Keep `evidence/thesis_readiness_report.md` with the experiment evidence package.",
        "6. Or run `python scripts/stage_thesis_template_import.py <template_dir>` to stage these files under a non-destructive `thesis_import/` directory.",
        "7. Verify the zip with `python scripts/verify_thesis_package.py` before handing it off.",
        "",
    ]
    readme_path.write_text("\n".join(lines), encoding="utf-8")
    archive_path = Path(shutil.make_archive(str(out_dir), "zip", root_dir=out_dir))
    archive_sha256 = _sha256_file(archive_path)
    archive_size = archive_path.stat().st_size
    archive_sha256_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    archive_sha256_path.write_text(f"{archive_sha256}  {archive_path.name}\n", encoding="utf-8")
    lines.extend(
        [
            "## Archive",
            "",
            f"- Zip: `{archive_path}`",
            f"- Size bytes: `{archive_size}`",
            f"- SHA-256: `{archive_sha256}`",
            f"- SHA-256 sidecar: `{archive_sha256_path}`",
            "",
        ]
    )
    readme_path.write_text("\n".join(lines), encoding="utf-8")
    manifest_path = out_dir / "manifest.json"
    manifest["manifest"] = str(manifest_path)
    manifest["readme"] = str(readme_path)
    manifest["archive"] = str(archive_path)
    manifest["archive_sha256_file"] = str(archive_sha256_path)
    manifest["archive_size_bytes"] = int(archive_size)
    manifest["archive_sha256"] = archive_sha256
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("docs/thesis_export_package"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = export_thesis_package(out_dir=args.out_dir)
    print(f"Exported thesis package to {manifest['out_dir']} ({manifest['status']})")


if __name__ == "__main__":
    main()
