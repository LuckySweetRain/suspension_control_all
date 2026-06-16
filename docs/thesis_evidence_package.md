# Thesis Evidence Package

This package indexes the current thesis-ready evidence files for the direct road-preview active-suspension route.

## 1. Route Decision

Current route:

```text
already perceived road unevenness / road preview sequence
        -> SPDF, fixed MPC-lite, BP-NMPS, SI-RPPO
        -> active suspension control
```

Chrono vision, PreScan contour extraction, and SimplerEnv-like VLA training are future extensions, not the thesis core.

Primary route document:

- `docs/direct_road_preview_control_plan.md`

## 2. Claim Matrix

Use this file as the top-level status and wording boundary:

- `docs/thesis_workflow_index.md`
- `docs/thesis_claim_matrix.md`
- `docs/thesis_completion_audit.md`
- `docs/thesis_defense_qa.md`
- `docs/thesis_final_submission_checklist.md`
- `docs/thesis_readiness_report.md`
- `docs/thesis_export_package/README.md`
- `docs/thesis_export_package.zip`

Current defensible claims:

- SPDF is a completed classical expert baseline.
- BP-NMPS is a conservative adaptive MPC parameter scheduler with calibration-set deployment fallback.
- SI-RPPO is a safe-teacher-gated, projection-aware online PPO policy class; it should not be claimed as universally superior to SAC/TD3.

## 3. BP-NMPS Evidence

### Formal seed sweep

- Summary: `results/mpc_parameter_seed_sweep_formal_gate/mpc_parameter_seed_sweep_summary.json`
- Report: `results/mpc_parameter_seed_sweep_formal_gate/mpc_parameter_seed_sweep_report.md`
- LaTeX table: `results/mpc_parameter_seed_sweep_formal_gate/mpc_parameter_seed_sweep_table.tex`
- Experiment log: `docs/bp_nmps_experiment_log.md`

Key numbers:

- Seeds: 42, 43, 44, 45, 46.
- Support rate: 1.000.
- Mean deployed policy delta: -0.00131430.
- Mean unsafe delta: 0.00000.
- Gate rate: 0.600.

### Evidence table across probes

- Markdown: `results/bp_nmps_evidence_table/bp_nmps_evidence_table.md`
- LaTeX: `results/bp_nmps_evidence_table/bp_nmps_evidence_table.tex`
- JSON: `results/bp_nmps_evidence_table/bp_nmps_evidence_table.json`

### Gain schedule interpretability

- Report: `results/bp_nmps_schedule_evidence/bp_nmps_schedule_evidence.md`
- CSV: `results/bp_nmps_schedule_evidence/bp_nmps_schedule_evidence.csv`
- Scenario CSV: `results/bp_nmps_schedule_evidence/bp_nmps_schedule_evidence_by_scenario.csv`

Interpretation:

- Gate-on seeds deploy fixed MPC-lite and keep gain deltas at zero.
- Gate-off seeds expose small, interpretable MPC-lite gain changes.

### Preview robustness

- Report: `results/bp_nmps_preview_robustness/bp_nmps_preview_robustness.md`
- CSV: `results/bp_nmps_preview_robustness/bp_nmps_preview_robustness.csv`
- LaTeX: `results/bp_nmps_preview_robustness/bp_nmps_preview_robustness.tex`
- Chinese summary: `results/bp_nmps_preview_robustness/bp_nmps_preview_robustness_summary_cn.md`

Key case means:

| Preview case | Score delta | Unsafe delta | Body RMS delta | Pitch RMS delta |
| --- | ---: | ---: | ---: | ---: |
| clean_preview | -170.081463 | -1.700 | -0.036521 | -0.015179 |
| nominal_preview_error | -220.063626 | -2.200 | -0.025656 | -0.012234 |
| strong_preview_error | -230.067552 | -2.300 | -0.017520 | -0.013252 |

Wording boundary:

- The full matrix supports BP-NMPS non-regression and robustness under the listed seeds, held-out scenarios, and preview-corruption cases.
- Do not state universal robustness over all possible roads or all perception errors.
- Score deltas are strongly affected by unsafe-step penalty; report unsafe and comfort metrics alongside score.

## 4. SI-RPPO Evidence

### Claim report

- Report: `results/si_rppo_e20_improvement_gate_fair_baselines/si_rppo_claim_report.md`
- JSON: `results/si_rppo_e20_improvement_gate_fair_baselines/si_rppo_claim_report.json`
- Combined metrics: `results/si_rppo_e20_improvement_gate_fair_baselines/combined_metrics.csv`
- Experiment log: `docs/si_rppo_experiment_log.md`
- Innovation framing: `docs/si_rppo_ccfa_innovation_framing.md`

Key claim:

- Gated BC-PPO improves over PPO scratch in return, unsafe steps, body/pitch/roll comfort, action smoothness, actuator tracking, and projection reliance.
- SAC remains a strong return/safety baseline; avoid claiming broad superiority over all off-policy methods.

### Gate evidence

- Report: `results/si_rppo_gate_evidence/si_rppo_gate_evidence.md`
- CSV: `results/si_rppo_gate_evidence/si_rppo_gate_evidence.csv`
- LaTeX: `results/si_rppo_gate_evidence/si_rppo_gate_evidence.tex`
- Method summary: `results/si_rppo_gate_evidence/si_rppo_gate_method_summary.md`

Key gate pattern:

| Scenario | Mean gate |
| --- | ---: |
| class_b_correlated | 0.0000 |
| class_c_correlated | 0.0000 |
| pothole_roll | 0.0363 |
| mixed_full | 0.4001 |
| sine_roll | 0.9074 |

Interpretation:

- Low-excitation road cases stay close to the passive safe teacher.
- High-excitation cases open the active PPO correction gate.
- This supports the strategy-network design claim: SI-RPPO changes the deployed policy class, not merely the reward function.

### Preview robustness

- Script: `scripts/run_si_rppo_preview_robustness.py`
- Report: `results/si_rppo_preview_robustness/si_rppo_preview_robustness.md`
- Summary CSV: `results/si_rppo_preview_robustness/si_rppo_preview_robustness_summary.csv`
- Pairwise CSV: `results/si_rppo_preview_robustness/si_rppo_preview_robustness_pairwise.csv`
- LaTeX table: `results/si_rppo_preview_robustness/si_rppo_preview_robustness.tex`
- Manifest: `results/si_rppo_preview_robustness/si_rppo_preview_robustness_manifest.json`

Purpose:

- Re-evaluate already-trained SI-RPPO/PPO variants under clean, noisy/delayed, strong-error, and missing-segment preview cases.
- Report safety, comfort, action smoothness, projection error, and improvement-gate activation under the same direct road-preview assumption used by BP-NMPS.
- Use this as robustness evidence for the policy-network design, not as a universal superiority claim over SAC/TD3.

Current run:

- Scope: 4 preview cases, all configured evaluation scenarios, `bc_ppo` vs `ppo_scratch`, `SAC`, and `TD3`, with PASSIVE and FULL_CAR_MPC_LITE structured baselines.
- Against PPO scratch, `bc_ppo` improves mean return by about `+7062`, reduces unsafe steps by `-17.70`, reduces action-delta RMS by `-158.01 N`, and reduces projection error by `-0.493`.
- Against TD3, `bc_ppo` improves mean return by about `+8063`, reduces unsafe steps by `-76.25`, and remains much smoother.
- Against SAC, `bc_ppo` is not return-superior (`-381.57` return delta), but it is smoother (`-138.99 N` action-delta RMS) and has much lower projection error (`-0.290`).
- Wording boundary: this supports deployability, safety, smoothness, and projection-aware PPO improvement; it still does not justify a broad claim that SI-RPPO universally outperforms SAC.

### Strategy-network evidence summary

- Script: `scripts/build_si_rppo_strategy_evidence_table.py`
- Report: `results/si_rppo_strategy_evidence/si_rppo_strategy_evidence_table.md`
- CSV: `results/si_rppo_strategy_evidence/si_rppo_strategy_evidence_table.csv`
- LaTeX: `results/si_rppo_strategy_evidence/si_rppo_strategy_evidence_table.tex`

Purpose:

- Tie each SI-RPPO strategy-network component to a specific artifact and a defensible thesis claim.
- Keep negative evidence visible: the residual prior branch is currently a negative ablation, while the SAC comparison is a return/smoothness tradeoff rather than a clean win.

## 5. Thesis Drafts

- Method and experiment draft: `docs/thesis_method_experiment_draft.md`
- Generated chapter draft: `docs/thesis_chapter_draft_cn.md`
- Generated LaTeX chapter draft: `docs/thesis_chapter_draft_cn.tex`
- Standalone LaTeX preview: `docs/thesis_standalone_preview.tex`
- Compiled standalone preview PDF: `docs/latex_preview/thesis_standalone_preview.pdf`
- MPC parameter policy plan: `docs/mpc_parameter_policy_plan.md`
- Direct road-preview control plan: `docs/direct_road_preview_control_plan.md`
- LaTeX table snippets: `docs/thesis_latex_snippets.tex`

The LaTeX snippets collect the current thesis-ready BP-NMPS and SI-RPPO tables. They assume the final thesis preamble includes `\usepackage{booktabs,graphicx}` because wide experiment tables are wrapped to fit `\textwidth`.

## 6. Reproduction Commands

BP-NMPS formal seed sweep:

```text
python scripts/run_mpc_parameter_seed_sweep.py --config configs/mujoco_full_car_corner.yaml --out results/mpc_parameter_seed_sweep_formal_gate --seeds 42,43,44,45,46 --scenario-limit 3 --eval-scenario-offset 3 --eval-scenario-limit 2 --tuning-trials 8 --cem-iterations 4 --cem-population 8 --neural-episodes 32 --neural-batch-size 4 --neural-anchor-coef 0.1 --neural-anchor-final-coef 0.02 --neural-elite-bc-coef 0.02 --neural-deployment-gate --scale-low 0.8 --scale-high 1.2 --min-support-rate 0.8 --max-mean-policy-delta 0.0 --max-mean-unsafe-delta 0.0 --min-mean-policy-improvement 0.001 --device cpu
```

BP-NMPS preview robustness:

```text
python scripts/run_bp_nmps_preview_robustness.py --summary results/mpc_parameter_seed_sweep_formal_gate/mpc_parameter_seed_sweep_summary.json --out results/bp_nmps_preview_robustness
```

BP-NMPS evidence table:

```text
python scripts/build_mpc_parameter_evidence_table.py --inputs results/mpc_parameter_seed_sweep_formal_probe results/mpc_parameter_seed_sweep_elite002_probe results/mpc_parameter_seed_sweep_deployment_gate_probe results/mpc_parameter_seed_sweep_formal_gate --labels anchor_only,elite_distill,elite_plus_gate_probe,formal_gate_5seed --out results/bp_nmps_evidence_table
```

SI-RPPO gate evidence:

```text
python scripts/build_si_rppo_gate_evidence.py --result-dir results/si_rppo_e20_improvement_gate_fair_baselines --out results/si_rppo_gate_evidence
```

SI-RPPO preview robustness:

```text
python scripts/run_si_rppo_preview_robustness.py --manifest results/si_rppo_e20_improvement_gate_fair_baselines/si_rppo_ablation_manifest.json --out results/si_rppo_preview_robustness
```

SI-RPPO strategy evidence table:

```text
python scripts/build_si_rppo_strategy_evidence_table.py
```

Top-level claim matrix:

```text
python scripts/build_thesis_claim_matrix.py
```

Thesis LaTeX snippets:

```text
python scripts/build_thesis_latex_snippets.py
```

Thesis completion audit:

```text
python scripts/build_thesis_completion_audit.py
```

Chinese thesis chapter draft:

```text
python scripts/build_thesis_chapter_draft.py
```

Chinese thesis LaTeX chapter draft:

```text
python scripts/build_thesis_chapter_latex.py
```

Standalone LaTeX preview:

```text
python scripts/build_thesis_standalone_preview.py
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=docs\latex_preview docs\thesis_standalone_preview.tex
```

## 7. Remaining Work

1. Run longer-budget BP-NMPS seed sweeps to verify that gate rate and mean improvement are stable.
2. Extend BP-NMPS preview robustness to more held-out road classes and stronger preview-corruption profiles.
3. Extend the SI-RPPO preview-robustness matrix to longer horizons and more explicitly hard road classes.
4. Strengthen SI-RPPO against SAC/TD3 using longer training budgets and harder road cases.
5. Replace or augment the hand-designed SI-RPPO gate with learned or uncertainty-aware improvement certificates.
6. Move the selected LaTeX tables into the final thesis document.
