# Thesis Workflow Index

This is the top-level workflow for the current thesis route.

## Route

```text
already perceived road unevenness / road preview sequence
        -> SPDF baseline
        -> BP-NMPS MPC parameter scheduling
        -> SI-RPPO safe-teacher-gated policy network
        -> thesis chapter / tables / defense materials
```

The thesis route does not depend on PreScan road-contour extraction, Chrono vision, or a VLA platform. Those can be future extensions.

## 1. Check Current Readiness

Run:

```text
python scripts/check_thesis_readiness.py
```

Expected current status:

```text
ready_for_template_migration
```

Read:

- `docs/thesis_readiness_report.md`
- `docs/thesis_completion_audit.md`
- `docs/thesis_objective_evidence.md`
- `docs/thesis_claim_boundary_audit.md`
- `docs/thesis_claim_matrix.md`

## 2. Export Handoff Package

Run:

```text
python scripts/export_thesis_package.py
```

Output:

- `docs/thesis_export_package/`
- `docs/thesis_export_package.zip`
- `docs/thesis_export_package.zip.sha256`

This package contains `latex/`, `guides/`, and `evidence/` directories for migration into the final university thesis template.

Before handing off the archive, verify it with:

```text
python scripts/verify_thesis_package.py
```

## 3. Stage Into a Thesis Template

After a final school thesis template directory is available, run:

```text
python scripts/stage_thesis_template_import.py <template_dir>
```

This is non-destructive. It creates `<template_dir>/thesis_import/` and writes `THESIS_IMPORT_INSTRUCTIONS.md`. It does not edit the template's `main.tex`.

## 4. Insert Into Final Thesis

In the final thesis preamble, include:

```tex
\usepackage{amsmath}
\usepackage{booktabs,graphicx}
\usepackage{array}
```

Then manually insert or adapt:

```tex
\input{thesis_import/latex/thesis_chapter_draft_cn.tex}
\input{thesis_import/latex/thesis_latex_snippets.tex}
```

Use the school template's chapter/table placement rules if they differ.

## 5. Claim Boundaries

Keep these boundaries unchanged:

- SPDF is the completed classical baseline.
- BP-NMPS is conservative MPC parameter scheduling with deployment fallback, not full MPC replacement.
- SI-RPPO is a safe-teacher-gated, projection-aware PPO policy class, not universal superiority over SAC.
- Residual prior remains a weak/negative ablation, not the final contribution.
- Chrono/PreScan/VLA are future extensions, not thesis core.

## 6. Defense Prep

Use:

- `docs/thesis_defense_qa.md`
- `docs/si_rppo_ccfa_innovation_framing.md`
- `docs/thesis_final_submission_checklist.md`

These files are intended to keep the final thesis and defense wording aligned with the actual evidence.
