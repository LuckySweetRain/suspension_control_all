# Thesis Completion Audit

This audit maps the stated thesis objectives to current artifacts. It is intentionally conservative: evidence-ready does not mean the final thesis is fully written.

- Overall status: `evidence_ready_with_remaining_thesis_work`

| Objective | Status | Evidence | Current proof | Remaining gap |
| --- | --- | --- | --- | --- |
| 1. SPDF method | evidence_ready | SPDF controller/test coverage and thesis claim matrix baseline entry | Completed classical expert baseline for comparison. | Move final SPDF baseline result/table into the final thesis chapter. |
| 2. MPC method with RL parameter learning | evidence_ready | BP-NMPS formal seed sweep, schedule evidence, preview robustness, evidence table | 5-seed support 1.000, mean score delta -0.001314, unsafe delta 0.000 | Longer-budget seed sweep and broader held-out road/preview-error matrix before final broad claims. |
| 3. RL strategy-network innovation | evidence_ready | SI-RPPO claim report, gate evidence, preview robustness, strategy evidence summary | core status supported; strategy evidence rows 7 | Learned/uncertainty-aware gate and longer harder-road SAC/TD3 comparisons for stronger CCF-A-level framing. |

## Defensible Thesis Position

- The route should remain direct road-preview control, assuming already perceived road unevenness is available.
- SPDF is a completed baseline, not the main contribution.
- BP-NMPS is the MPC contribution: a baseline-preserving neural MPC parameter scheduler with deployment fallback.
- SI-RPPO is the RL contribution: a projection-aware, safe-teacher-gated PPO policy class with preview-robust smooth deployment evidence.
- The current evidence should not be phrased as universal superiority over SAC or all road/perception conditions.

## Next Highest-Value Work

- Thesis integration artifacts ready: `True`.

1. Follow `docs/thesis_latex_integration_guide.md` when moving `docs/thesis_chapter_draft_cn.tex` and `docs/thesis_latex_snippets.tex` into the final thesis source.
2. Use `docs/thesis_final_submission_checklist.md` as the final manual checklist before submission.
3. Run `python scripts/check_thesis_readiness.py` before migrating into the final school template.
4. Ensure the final thesis preamble includes `\usepackage{booktabs,graphicx}` before inserting the generated evidence tables.
5. Use `docs/thesis_standalone_preview.tex` and `docs/latex_preview/thesis_standalone_preview.pdf` for local compile checks before editing the school template.
6. Add longer-budget/harder-road SI-RPPO runs if time allows.
7. Prototype or at least specify a learned/uncertainty-aware gate as future work or an extra ablation.
8. Keep BP-NMPS claims conservative unless broader held-out road matrices are run.
