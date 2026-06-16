"""Build a standalone LaTeX preview document for thesis chapter artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_thesis_standalone_preview(
    out_path: Path = Path("docs/thesis_standalone_preview.tex"),
    chapter_path: Path = Path("docs/thesis_chapter_draft_cn.tex"),
    snippets_path: Path = Path("docs/thesis_latex_snippets.tex"),
) -> dict[str, str]:
    if not chapter_path.exists():
        raise FileNotFoundError(chapter_path)
    if not snippets_path.exists():
        raise FileNotFoundError(snippets_path)

    lines = [
        "% Auto-generated standalone preview for the thesis chapter and evidence tables.",
        r"\documentclass[UTF8,zihao=-4]{ctexbook}",
        r"\usepackage[a4paper,margin=2.5cm]{geometry}",
        r"\usepackage{amsmath}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{array}",
        r"\usepackage{hyperref}",
        r"\hypersetup{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}",
        r"\setlength{\parindent}{2em}",
        r"\setlength{\parskip}{0.35em}",
        r"\sloppy",
        r"\begin{document}",
        r"\frontmatter",
        r"\title{基于路面预瞄的主动悬架控制方法与实验预览}",
        r"\author{自动生成草稿}",
        r"\date{\today}",
        r"\maketitle",
        r"\tableofcontents",
        r"\mainmatter",
        rf"\input{{{chapter_path.as_posix()}}}",
        r"\appendix",
        r"\chapter{实验表格片段}",
        rf"\input{{{snippets_path.as_posix()}}}",
        r"\end{document}",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return {"out": str(out_path), "chapter": str(chapter_path), "snippets": str(snippets_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/thesis_standalone_preview.tex"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_thesis_standalone_preview(out_path=args.out)
    print(f"Saved thesis standalone LaTeX preview to {result['out']}")


if __name__ == "__main__":
    main()
