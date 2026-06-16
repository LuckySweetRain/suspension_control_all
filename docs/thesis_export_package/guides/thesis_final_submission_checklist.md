# Thesis Final Submission Checklist

Use this checklist before moving from the engineering repository to the final university thesis template.

## Must Complete

- [ ] Move `docs/thesis_chapter_draft_cn.tex` into the final thesis source as the method/experiment chapter basis.
- [ ] Move or `\input{}` the selected tables from `docs/thesis_latex_snippets.tex`.
- [ ] Add required packages to the final thesis preamble:

```tex
\usepackage{amsmath}
\usepackage{booktabs,graphicx}
\usepackage{array}
```

- [ ] Compile the final thesis with XeLaTeX after inserting the generated tables.
- [ ] Check that all table references resolve:
  - `tab:bp_nmps_seed_sweep`
  - `tab:bp_nmps_evidence`
  - `tab:bp_nmps_preview_robustness`
  - `tab:si_rppo_gate_evidence`
  - `tab:si_rppo_preview_robustness`
  - `tab:si_rppo_strategy_evidence`
- [ ] Keep `docs/latex_preview/thesis_standalone_preview.pdf` as the local reference PDF for the generated chapter and tables.

## Evidence to Cite

- [ ] SPDF: cite as the completed classical baseline.
- [ ] BP-NMPS: cite `results/mpc_parameter_seed_sweep_formal_gate/mpc_parameter_seed_sweep_summary.json` and the corresponding LaTeX table.
- [ ] BP-NMPS schedule: cite `results/bp_nmps_schedule_evidence/bp_nmps_schedule_evidence.md` for gain interpretability.
- [ ] BP-NMPS preview robustness: cite `results/bp_nmps_preview_robustness/bp_nmps_preview_robustness.md`.
- [ ] SI-RPPO claim report: cite `results/si_rppo_e20_improvement_gate_fair_baselines/si_rppo_claim_report.md`.
- [ ] SI-RPPO gate evidence: cite `results/si_rppo_gate_evidence/si_rppo_gate_evidence.md`.
- [ ] SI-RPPO strategy evidence: cite `results/si_rppo_strategy_evidence/si_rppo_strategy_evidence_table.md`.
- [ ] SI-RPPO innovation framing: cite or reuse wording from `docs/si_rppo_ccfa_innovation_framing.md`.
- [ ] Defense wording: reuse concise answers from `docs/thesis_defense_qa.md`.

## Claim Boundaries

- [ ] State that the thesis uses already perceived road unevenness / road-preview information.
- [ ] Do not claim that PreScan visual contour extraction is implemented as the core route.
- [ ] Do not claim that Chrono/VLA is the thesis platform.
- [ ] State BP-NMPS as conservative MPC parameter scheduling with deployment fallback.
- [ ] Do not state that BP-NMPS fully replaces MPC.
- [ ] State SI-RPPO as a safe-teacher-gated, projection-aware PPO policy class.
- [ ] Do not state that SI-RPPO is universally superior to SAC.
- [ ] Keep the residual prior branch as a negative/weak ablation, not the main contribution.

## Optional Strengthening Before Final Submission

- [ ] Run longer-budget BP-NMPS seed sweeps if time permits.
- [ ] Extend BP-NMPS preview robustness to more road classes or stronger preview corruption.
- [ ] Add harder-road SI-RPPO comparisons against SAC and TD3.
- [ ] Add a learned or uncertainty-aware gate prototype, or describe it clearly as future work.
- [ ] Add a short limitation paragraph covering seed count, simulation-only validation, road distribution scope, preview corruption scope, and lack of hardware-in-the-loop validation.

## Final One-Sentence Thesis Position

本文以已感知路面不平度预瞄为输入，构建 SPDF 传统基线、BP-NMPS 保守 MPC 参数学习方法和 SI-RPPO 安全教师门控策略网络，重点解决主动悬架控制中可解释、可部署、执行器可行的预瞄控制问题，而不将视觉重建或 VLA 平台作为核心贡献。
