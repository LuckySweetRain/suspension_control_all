# Thesis Import Instructions

This directory was staged from `suspension_control_all` for manual insertion into the final thesis template.

## Preamble

Ensure the final thesis preamble includes:

```tex
\usepackage{amsmath}
\usepackage{booktabs,graphicx}
\usepackage{array}
```

## Main Chapter

Copy or input:

```tex
\input{thesis_import/latex/thesis_chapter_draft_cn.tex}
```

## Evidence Tables

Copy selected tables or input the snippet file near the relevant chapter/appendix:

```tex
\input{thesis_import/latex/thesis_latex_snippets.tex}
```

## Before Submission

Review `thesis_import/guides/thesis_final_submission_checklist.md` and keep the claim boundaries unchanged.
