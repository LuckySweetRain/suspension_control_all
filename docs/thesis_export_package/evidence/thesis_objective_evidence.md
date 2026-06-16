# Thesis Objective Evidence

This matrix verifies the user-stated thesis objectives against current artifacts. It is stricter than a narrative claim matrix: every objective must point to concrete files and measured evidence.

- Status: `objective_evidence_ready`
- Evidence-ready objectives: `3`
- Incomplete objectives: `0`

| Objective | Evidence status | Proof scope | Key result | Defensible claim | Remaining gap |
| --- | --- | --- | --- | --- | --- |
| SPDF方法，已经完成 | evidence_ready | classical baseline implemented and represented in thesis-facing materials | SPDF is positioned as the completed non-learning baseline. | SPDF can be used as a completed traditional-control baseline, not as the main learning contribution. | Final school thesis still needs the selected SPDF baseline result/table inserted in the template. |
| MPC方法，加入强化学习进行MPC参数学习 | evidence_ready | baseline-preserving neural MPC-lite parameter scheduling under held-out seed/probe evidence | support_rate=1.000, mean_policy_delta_vs_fixed=-0.001314, mean_unsafe_delta_vs_fixed=0.000 | BP-NMPS is a conservative RL-assisted MPC parameter scheduler with deployment fallback. | Do not claim full MPC replacement; broader long-budget road matrices remain future strengthening work. |
| 强化学习策略网络设计，达到CCFA论文级别创新水平 | evidence_ready | safe-teacher-gated, projection-aware PPO policy class with strategy-network evidence | claim_status=ready, core_status=supported, strategy_rows=7 | SI-RPPO can be framed as a deployability-oriented policy-network contribution, with careful SAC tradeoff wording. | CCF-A-level wording should stay method-level unless longer harder-road and learned-gate ablations are added. |

## Artifact Checks

### spdf
- `pass` `controllers\__init__.py`
- `pass` `tests\test_core.py`
- `pass` `docs\thesis_claim_matrix.md`
- `pass` `docs\thesis_chapter_draft_cn.tex`

### bp_nmps
- `pass` `results\mpc_parameter_seed_sweep_formal_gate\mpc_parameter_seed_sweep_summary.json`
- `pass` `results\bp_nmps_schedule_evidence\bp_nmps_schedule_evidence.md`
- `pass` `results\bp_nmps_preview_robustness\bp_nmps_preview_robustness.md`
- `pass` `results\bp_nmps_evidence_table\bp_nmps_evidence_table.md`
- `pass` `docs\thesis_latex_snippets.tex`

### si_rppo
- `pass` `results\si_rppo_e20_improvement_gate_fair_baselines\si_rppo_claim_report.json`
- `pass` `results\si_rppo_gate_evidence\si_rppo_gate_evidence.md`
- `pass` `results\si_rppo_preview_robustness\si_rppo_preview_robustness.md`
- `pass` `results\si_rppo_strategy_evidence\si_rppo_strategy_evidence_table.md`
- `pass` `docs\si_rppo_ccfa_innovation_framing.md`
- `pass` `docs\thesis_latex_snippets.tex`
