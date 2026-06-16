"""Stage thesis artifacts inside a final thesis template directory.

The script is deliberately non-destructive: it copies files into a
``thesis_import`` subdirectory and writes insertion instructions, but it does
not edit the template's main .tex file.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


IMPORT_FILES = (
    (Path("docs/thesis_export_package/latex/thesis_chapter_draft_cn.tex"), Path("latex/thesis_chapter_draft_cn.tex")),
    (Path("docs/thesis_export_package/latex/thesis_latex_snippets.tex"), Path("latex/thesis_latex_snippets.tex")),
    (Path("docs/thesis_export_package/latex/thesis_standalone_preview.pdf"), Path("latex/thesis_standalone_preview.pdf")),
    (Path("docs/thesis_export_package/guides/thesis_latex_integration_guide.md"), Path("guides/thesis_latex_integration_guide.md")),
    (Path("docs/thesis_export_package/guides/thesis_workflow_index.md"), Path("guides/thesis_workflow_index.md")),
    (Path("docs/thesis_export_package/guides/thesis_final_submission_checklist.md"), Path("guides/thesis_final_submission_checklist.md")),
    (Path("docs/thesis_export_package/guides/thesis_defense_qa.md"), Path("guides/thesis_defense_qa.md")),
    (Path("docs/thesis_export_package/evidence/thesis_readiness_report.md"), Path("evidence/thesis_readiness_report.md")),
)


def stage_thesis_template_import(
    template_dir: Path,
    import_subdir: str = "thesis_import",
) -> dict[str, object]:
    if not template_dir.exists():
        raise FileNotFoundError(template_dir)
    if not template_dir.is_dir():
        raise NotADirectoryError(template_dir)

    import_dir = template_dir / import_subdir
    copied: list[dict[str, str]] = []
    missing: list[str] = []
    for source, relative_target in IMPORT_FILES:
        if not source.exists():
            missing.append(str(source))
            continue
        target = import_dir / relative_target
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append({"source": str(source), "target": str(target)})

    instructions = import_dir / "THESIS_IMPORT_INSTRUCTIONS.md"
    lines = [
        "# Thesis Import Instructions",
        "",
        "This directory was staged from `suspension_control_all` for manual insertion into the final thesis template.",
        "",
        "## Preamble",
        "",
        "Ensure the final thesis preamble includes:",
        "",
        "```tex",
        "\\usepackage{amsmath}",
        "\\usepackage{booktabs,graphicx}",
        "\\usepackage{array}",
        "```",
        "",
        "## Main Chapter",
        "",
        "Copy or input:",
        "",
        "```tex",
        "\\input{thesis_import/latex/thesis_chapter_draft_cn.tex}",
        "```",
        "",
        "## Evidence Tables",
        "",
        "Copy selected tables or input the snippet file near the relevant chapter/appendix:",
        "",
        "```tex",
        "\\input{thesis_import/latex/thesis_latex_snippets.tex}",
        "```",
        "",
        "## Before Submission",
        "",
        "Review `thesis_import/guides/thesis_final_submission_checklist.md` and keep the claim boundaries unchanged.",
        "",
    ]
    instructions.parent.mkdir(parents=True, exist_ok=True)
    instructions.write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "status": "staged" if not missing else "incomplete",
        "template_dir": str(template_dir),
        "import_dir": str(import_dir),
        "copied_count": len(copied),
        "missing": missing,
        "copied": copied,
        "instructions": str(instructions),
    }
    manifest_path = import_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template_dir", type=Path)
    parser.add_argument("--import-subdir", default="thesis_import")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = stage_thesis_template_import(args.template_dir, import_subdir=args.import_subdir)
    print(f"Staged thesis import to {manifest['import_dir']} ({manifest['status']})")


if __name__ == "__main__":
    main()
