# Thesis Claim Matrix

This file is generated from experiment artifacts and is intended to keep thesis claims aligned with current evidence.

## Objective Alignment

| Thesis item | Current status | Evidence strength | Main metric movement | Safety evidence | Thesis wording boundary |
| --- | --- | --- | --- | --- | --- |
| SPDF baseline | completed | report/framework baseline | n/a | n/a | Treat as classical expert baseline, not the final learning contribution. |
| BP-NMPS MPC parameter learning | supported | 5 seeds, support 1.000 | score delta -0.001314 | unsafe delta 0.000, gate rate 0.600 | Claim as conservative adaptive MPC-parameter scheduling, not black-box MPC replacement. |
| Safe-teacher gated SI-RPPO policy network | supported | 3-seed core + fair full matrix | return delta 6326.7494 | unsafe delta -20.8000, action-delta delta -158.1090, projection-error delta -0.479976 | Claim as actuator-feasible online PPO improvement over PPO scratch; compare carefully against SAC. |

## Current Defensible Claims

- SPDF can be presented as a completed traditional controller baseline for comparison.
- BP-NMPS uses fixed MPC-lite as a deployable baseline and schedules MPC-lite gains through a neural parameter policy.
- The deployment gate is part of the algorithm: failed calibration falls back to fixed MPC-lite on held-out evaluation.
- SI-RPPO novelty is the policy class: offline imitation, delta action parameterization, projection-aware losses, and safe-teacher improvement gating.
- The current broad superiority claim is not proven; SAC remains a strong return/safety baseline in short runs.

## Evidence Pointers

- BP-NMPS seed count: `5`.
- BP-NMPS non-inferiority status: `supported`.
- BP-NMPS improvement status: `supported`.
- BP-NMPS schedule evidence: `results\bp_nmps_schedule_evidence\bp_nmps_schedule_evidence.md`.
- BP-NMPS preview-robustness evidence: `results\bp_nmps_preview_robustness\bp_nmps_preview_robustness.md`.
- BP-NMPS preview-robustness LaTeX table: `results\bp_nmps_preview_robustness\bp_nmps_preview_robustness.tex`.
- BP-NMPS preview-robustness Chinese summary: `results\bp_nmps_preview_robustness\bp_nmps_preview_robustness_summary_cn.md`.
- Delta PPO repeated-seed support: `3/3`.
- SI-RPPO overall claim report status: `ready`.
- SI-RPPO core status: `supported`.
- SI-RPPO gate evidence: `results\si_rppo_gate_evidence\si_rppo_gate_evidence.md`.
- SI-RPPO gate LaTeX table: `results\si_rppo_gate_evidence\si_rppo_gate_evidence.tex`.
- SI-RPPO method summary: `results\si_rppo_gate_evidence\si_rppo_gate_method_summary.md`.
- SI-RPPO innovation framing: `docs\si_rppo_ccfa_innovation_framing.md`.
- SI-RPPO preview-robustness evidence: `results\si_rppo_preview_robustness\si_rppo_preview_robustness.md`.
- SI-RPPO preview-robustness LaTeX table: `results\si_rppo_preview_robustness\si_rppo_preview_robustness.tex`.
- SI-RPPO strategy-network evidence: `results\si_rppo_strategy_evidence\si_rppo_strategy_evidence_table.md`.
- SI-RPPO strategy-network LaTeX table: `results\si_rppo_strategy_evidence\si_rppo_strategy_evidence_table.tex`.
- Thesis method/experiment draft: `docs\thesis_method_experiment_draft.md`.
- Thesis chapter draft: `docs\thesis_chapter_draft_cn.md`.
- Thesis chapter LaTeX draft: `docs\thesis_chapter_draft_cn.tex`.
- Thesis standalone LaTeX preview: `docs\thesis_standalone_preview.tex`.
- Thesis standalone preview PDF: `docs\latex_preview\thesis_standalone_preview.pdf`.
- Direct road-preview control plan: `docs\direct_road_preview_control_plan.md`.
- Thesis workflow index: `docs\thesis_workflow_index.md`.
- Thesis evidence package: `docs\thesis_evidence_package.md`.
- Thesis LaTeX snippets: `docs\thesis_latex_snippets.tex`.
- Thesis LaTeX integration guide: `docs\thesis_latex_integration_guide.md`.
- Thesis defense Q&A: `docs\thesis_defense_qa.md`.
- Thesis final submission checklist: `docs\thesis_final_submission_checklist.md`.
- Thesis readiness report: `docs\thesis_readiness_report.md`.
- Thesis export package: `docs\thesis_export_package\README.md`.
- Thesis completion audit: `docs\thesis_completion_audit.md`.

## Next Experiments Before Final Thesis Claims

1. Run longer-budget BP-NMPS seed sweeps to verify gate rate and mean improvement stability.
2. Extend BP-NMPS preview robustness to more held-out road classes and stronger preview-corruption profiles.
3. Strengthen SI-RPPO with learned or uncertainty-aware gate evidence, especially against SAC under harder road cases where passive-like policies are insufficient.
4. Evaluate SI-RPPO under preview-noise, delay, truncation, and missing-segment tests.
5. Move selected LaTeX tables into the final thesis document.
