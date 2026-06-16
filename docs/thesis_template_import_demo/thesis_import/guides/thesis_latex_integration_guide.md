# Thesis LaTeX Integration Guide

This guide records the final-thesis integration path for the direct road-preview active-suspension route. It intentionally treats Chrono/PreScan visual road extraction as a future extension, not as the main thesis dependency.

## Files to Integrate

Copy or input these generated artifacts into the final thesis template:

- `docs/thesis_chapter_draft_cn.tex`: method and experiment chapter draft.
- `docs/thesis_latex_snippets.tex`: thesis-ready evidence tables for BP-NMPS and SI-RPPO.
- `docs/thesis_standalone_preview.tex`: local compile wrapper for checking the chapter and table snippets before moving them into the final template.

## Required Preamble Packages

The final thesis preamble should include:

```tex
\usepackage{amsmath}
\usepackage{booktabs,graphicx}
\usepackage{array}
```

`graphicx` is required because the generated evidence snippets wrap wide experiment tables with `\resizebox{\textwidth}{!}{...}`.

## Recommended Placement

Use `docs/thesis_chapter_draft_cn.tex` as the basis for the main method/experiment chapter. In the final thesis, place the strongest evidence tables near the corresponding claims:

- BP-NMPS formal seed sweep: main MPC/RL parameter-learning subsection.
- BP-NMPS preview robustness: road-preview robustness subsection.
- SI-RPPO gate evidence: strategy-network design subsection.
- SI-RPPO preview robustness: RL experiment subsection.
- SI-RPPO strategy-network evidence summary: end of the RL method/experiment chapter or appendix.

The broader BP-NMPS probe table can go in an ablation subsection or appendix because it mainly explains why the deployment-gated setting is the defensible final claim.

## Claim Boundary

Use conservative wording:

- SPDF: completed baseline/classical control method.
- BP-NMPS: RL-assisted MPC parameter scheduler with deployment fallback; current evidence supports non-regression and small improvement under held-out evaluation.
- SI-RPPO: strategy-network innovation based on safe-teacher imitation, projection-aware PPO, and state-dependent gate evidence; do not claim broad superiority over SAC because the current matrix shows a return tradeoff.

The core thesis route is direct use of already perceived road unevenness / road-preview information for control, not end-to-end visual road reconstruction.
