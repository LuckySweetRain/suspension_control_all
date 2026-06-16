"""Build thesis-ready LaTeX snippets from current experiment tables."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LatexSnippet:
    title: str
    path: Path
    note: str
    fit_to_width: bool = True


DEFAULT_SNIPPETS = (
    LatexSnippet(
        title="BP-NMPS formal seed sweep",
        path=Path("results/mpc_parameter_seed_sweep_formal_gate/mpc_parameter_seed_sweep_table.tex"),
        note="Use for the main BP-NMPS non-regression/improvement table.",
    ),
    LatexSnippet(
        title="BP-NMPS evidence across probes",
        path=Path("results/bp_nmps_evidence_table/bp_nmps_evidence_table.tex"),
        note="Use in ablation or supplementary evidence to show the gate-enabled setting is the strongest supported probe.",
    ),
    LatexSnippet(
        title="BP-NMPS preview robustness",
        path=Path("results/bp_nmps_preview_robustness/bp_nmps_preview_robustness.tex"),
        note="Use for the direct road-preview route; report unsafe and comfort deltas alongside score.",
    ),
    LatexSnippet(
        title="SI-RPPO safe-teacher gate evidence",
        path=Path("results/si_rppo_gate_evidence/si_rppo_gate_evidence.tex"),
        note="Use to support the policy-network design claim, not broad superiority over SAC.",
    ),
    LatexSnippet(
        title="SI-RPPO preview robustness",
        path=Path("results/si_rppo_preview_robustness/si_rppo_preview_robustness.tex"),
        note="Use as preliminary preview-robustness evidence when the manifest confirms the intended scenario and baseline scope.",
    ),
    LatexSnippet(
        title="SI-RPPO strategy-network evidence summary",
        path=Path("results/si_rppo_strategy_evidence/si_rppo_strategy_evidence_table.tex"),
        note="Use to connect each strategy-network design component to its evidence and wording boundary.",
    ),
)


def build_thesis_latex_snippets(
    snippets: tuple[LatexSnippet, ...] = DEFAULT_SNIPPETS,
    out_path: Path = Path("docs/thesis_latex_snippets.tex"),
) -> Path:
    missing = [snippet.path for snippet in snippets if not snippet.path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing LaTeX snippet source(s): {joined}")

    lines = [
        "% Thesis LaTeX snippets generated from current experiment artifacts.",
        "% Required packages in the thesis preamble: \\usepackage{booktabs,graphicx}.",
        "% These tables support the direct road-preview active-suspension route.",
        "",
    ]
    for snippet in snippets:
        table = snippet.path.read_text(encoding="utf-8").strip()
        if snippet.fit_to_width:
            table = fit_tabular_to_textwidth(table)
        lines.extend(
            [
                f"% --- {snippet.title} ---",
                f"% Source: {snippet.path.as_posix()}",
                f"% Note: {snippet.note}",
                table,
                "",
            ]
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def fit_tabular_to_textwidth(table: str) -> str:
    """Wrap the tabular body so wide experiment tables fit thesis pages."""
    begin = "\\begin{tabular}"
    end = "\\end{tabular}"
    if "\\resizebox{\\textwidth}" in table:
        return table
    if begin not in table or end not in table:
        return table
    table = table.replace(begin, "\\resizebox{\\textwidth}{!}{%\n" + begin, 1)
    table = table.replace(end, end + "%\n}", 1)
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/thesis_latex_snippets.tex"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = build_thesis_latex_snippets(out_path=args.out)
    print(f"Saved thesis LaTeX snippets to {out_path}")


if __name__ == "__main__":
    main()
