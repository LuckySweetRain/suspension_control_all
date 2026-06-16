# BP-NMPS Experiment Log

## 2026-06-05: Held-Out Parameter Policy Probe

Command:

```text
python scripts/run_mpc_parameter_seed_sweep.py --config configs/mujoco_full_car_corner.yaml --out results/mpc_parameter_seed_sweep_formal_probe --seeds 42,43,44 --scenario-limit 3 --eval-scenario-offset 3 --eval-scenario-limit 2 --tuning-trials 4 --cem-iterations 2 --cem-population 4 --neural-episodes 8 --neural-batch-size 4 --neural-anchor-coef 0.1 --neural-anchor-final-coef 0.02 --scale-low 0.8 --scale-high 1.2 --min-support-rate 0.8 --max-mean-policy-delta 0.0 --max-mean-unsafe-delta 0.0 --min-mean-policy-improvement 0.001 --device cpu
```

Scope:

- Training/calibration scenarios: `class_b_correlated`, `class_c_correlated`, `sine_roll`.
- Held-out scenarios: `pothole_roll`, `mixed_full`.
- Seeds: 42, 43, 44.
- Neural policy uses anchor annealing but no elite action distillation.

Result:

| Seed | Fixed score | BP-NMPS score | Delta | Supported |
| ---: | ---: | ---: | ---: | --- |
| 42 | 2.65188 | 2.65188 | 0.00000 | yes |
| 43 | 2.65188 | 2.65188 | 0.00000 | yes |
| 44 | 2.65188 | 2.65188 | 0.00000 | yes |

Interpretation:

- Non-inferiority is supported because the deterministic policy exactly preserves fixed MPC-lite.
- Improvement is not supported. The learned mean remains too close to zero, so sampled best actions are not converted into deployable policy behavior.
- This motivates elite action distillation.

## 2026-06-05: Elite Distillation Probe

Command:

```text
python scripts/run_mpc_parameter_seed_sweep.py --config configs/mujoco_full_car_corner.yaml --out results/mpc_parameter_seed_sweep_elite002_probe --seeds 42,43,44 --scenario-limit 3 --eval-scenario-offset 3 --eval-scenario-limit 2 --tuning-trials 4 --cem-iterations 2 --cem-population 4 --neural-episodes 8 --neural-batch-size 4 --neural-anchor-coef 0.1 --neural-anchor-final-coef 0.02 --neural-elite-bc-coef 0.02 --scale-low 0.8 --scale-high 1.2 --min-support-rate 0.8 --max-mean-policy-delta 0.0 --max-mean-unsafe-delta 0.0 --min-mean-policy-improvement 0.001 --device cpu
```

Result:

| Seed | Fixed score | Raw BP-NMPS score | Delta | Supported |
| ---: | ---: | ---: | ---: | --- |
| 42 | 2.65188 | 2.64737 | -0.00452 | yes |
| 43 | 2.65188 | 2.64507 | -0.00682 | yes |
| 44 | 2.65188 | 2.65840 | +0.00652 | no |

Interpretation:

- Elite distillation works: the deterministic policy no longer remains exactly equal to fixed MPC-lite.
- Seeds 42 and 43 improve held-out score, body acceleration, actuator energy, and return without increasing unsafe steps.
- Seed 44 degrades, so non-inferiority is not robust enough without a deployment gate.
- This motivates a calibration-set baseline fallback.

## 2026-06-05: Baseline-Safe Deployment Gate Probe

Command:

```text
python scripts/run_mpc_parameter_seed_sweep.py --config configs/mujoco_full_car_corner.yaml --out results/mpc_parameter_seed_sweep_deployment_gate_probe --seeds 42,43,44 --scenario-limit 3 --eval-scenario-offset 3 --eval-scenario-limit 2 --tuning-trials 4 --cem-iterations 2 --cem-population 4 --neural-episodes 8 --neural-batch-size 4 --neural-anchor-coef 0.1 --neural-anchor-final-coef 0.02 --neural-elite-bc-coef 0.02 --neural-deployment-gate --scale-low 0.8 --scale-high 1.2 --min-support-rate 0.8 --max-mean-policy-delta 0.0 --max-mean-unsafe-delta 0.0 --min-mean-policy-improvement 0.001 --device cpu
```

Result:

| Seed | Fixed score | Raw BP-NMPS score | Deployed score | Delta | Gate | Supported |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 42 | 2.65188 | 2.64737 | 2.64737 | -0.00452 | off | yes |
| 43 | 2.65188 | 2.64507 | 2.64507 | -0.00682 | off | yes |
| 44 | 2.65188 | 2.65840 | 2.65188 | 0.00000 | on | yes |

Claim gate:

- Support rate: `1.000`.
- Mean deployed policy delta: `-0.00377859`.
- Mean unsafe delta: `0.00000`.
- Non-inferiority status: `supported`.
- Improvement status: `supported`.

Interpretation:

- The deployment gate corrected the seed-44 failure without using held-out feedback. It triggered because seed 44 was already worse than fixed MPC-lite on the training/calibration scenarios.
- The current defensible claim is not that every raw neural schedule is better. The defensible claim is:

> BP-NMPS learns a context-conditioned MPC gain scheduler and uses a calibration-set baseline deployment gate to preserve fixed MPC-lite when the learned deterministic policy is not supported by training evidence. In the current held-out three-seed probe, the deployed BP-NMPS policy is non-inferior to fixed MPC-lite on all seeds and improves mean score without increasing unsafe steps.

Residual risk:

- This is still a small probe with 8 neural episodes and 3 seeds. A thesis-grade result should repeat the same protocol with longer training, more seeds, and a final fixed command before claiming broad robustness.
- The method should report both raw policy score and deployed policy score, because the deployment gate is part of the algorithm and must remain transparent.

Next action:

Run the formal version with a larger budget:

```text
python scripts/run_mpc_parameter_seed_sweep.py --config configs/mujoco_full_car_corner.yaml --out results/mpc_parameter_seed_sweep_formal_gate --seeds 42,43,44,45,46 --scenario-limit 3 --eval-scenario-offset 3 --eval-scenario-limit 2 --tuning-trials 8 --cem-iterations 4 --cem-population 8 --neural-episodes 32 --neural-batch-size 4 --neural-anchor-coef 0.1 --neural-anchor-final-coef 0.02 --neural-elite-bc-coef 0.02 --neural-deployment-gate --scale-low 0.8 --scale-high 1.2 --min-support-rate 0.8 --max-mean-policy-delta 0.0 --max-mean-unsafe-delta 0.0 --min-mean-policy-improvement 0.001 --device cpu
```

## 2026-06-05: Formal Gate 5-Seed Sweep

Command:

```text
python scripts/run_mpc_parameter_seed_sweep.py --config configs/mujoco_full_car_corner.yaml --out results/mpc_parameter_seed_sweep_formal_gate --seeds 42,43,44,45,46 --scenario-limit 3 --eval-scenario-offset 3 --eval-scenario-limit 2 --tuning-trials 8 --cem-iterations 4 --cem-population 8 --neural-episodes 32 --neural-batch-size 4 --neural-anchor-coef 0.1 --neural-anchor-final-coef 0.02 --neural-elite-bc-coef 0.02 --neural-deployment-gate --scale-low 0.8 --scale-high 1.2 --min-support-rate 0.8 --max-mean-policy-delta 0.0 --max-mean-unsafe-delta 0.0 --min-mean-policy-improvement 0.001 --device cpu
```

Scope:

- Training/calibration scenarios: `class_b_correlated`, `class_c_correlated`, `sine_roll`.
- Held-out scenarios: `pothole_roll`, `mixed_full`.
- Seeds: 42, 43, 44, 45, 46.
- Neural training budget: 32 one-step policy episodes per seed.
- Tuning baselines: 8 random-search trials, 4 CEM iterations with population 8.

Result:

| Seed | Fixed score | Raw BP-NMPS score | Deployed score | Delta | Gate | Supported |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 42 | 2.65188 | 2.64739 | 2.64739 | -0.00450 | off | yes |
| 43 | 2.65188 | 2.65238 | 2.65188 | 0.00000 | on | yes |
| 44 | 2.65188 | 2.65359 | 2.65188 | 0.00000 | on | yes |
| 45 | 2.65188 | 2.64981 | 2.64981 | -0.00207 | off | yes |
| 46 | 2.65188 | 2.65224 | 2.65188 | 0.00000 | on | yes |

Claim gate:

- Support rate: `1.000`.
- Mean deployed policy delta: `-0.00131430`.
- Mean raw policy delta: `-0.00080197`.
- Mean unsafe delta: `0.00000`.
- Gate rate: `0.600`.
- Non-inferiority status: `supported`.
- Improvement status: `supported`.

Interpretation:

- The 5-seed run supports the deployed BP-NMPS claim under the current held-out protocol.
- Two seeds deploy the learned gain scheduler and improve held-out score. Three seeds are rejected by the calibration gate and safely fall back to fixed MPC-lite.
- The improvement is modest but algorithmically meaningful because it is achieved without increasing unsafe steps and without allowing unsupported learned schedules to degrade deployment.
- The deployment gate is central to the claim. The raw neural policy alone is not robust enough to claim broad superiority.

Recommended thesis wording:

> In the held-out full-car active-suspension benchmark, BP-NMPS combines neural MPC gain scheduling with a calibration-set deployment gate. Across five random seeds, the deployed BP-NMPS policy remains non-inferior to fixed MPC-lite on every seed and achieves a negative mean score delta without increasing unsafe steps. The result supports BP-NMPS as a conservative adaptive MPC-parameter scheduler rather than an unconstrained replacement for MPC-lite.

Evidence table:

```text
python scripts/build_mpc_parameter_evidence_table.py --inputs results/mpc_parameter_seed_sweep_formal_probe results/mpc_parameter_seed_sweep_elite002_probe results/mpc_parameter_seed_sweep_deployment_gate_probe results/mpc_parameter_seed_sweep_formal_gate --labels anchor_only,elite_distill,elite_plus_gate_probe,formal_gate_5seed --out results/bp_nmps_evidence_table
```

## 2026-06-05: Preview-Robustness Probe

Command:

```text
python scripts/run_bp_nmps_preview_robustness.py --out results/bp_nmps_preview_robustness_probe --seed-limit 2 --scenario-limit 2 --case-limit 3 --episode-seconds 0.2
```

Scope:

- Seeds: 42 and 43.
- Held-out scenarios: `pothole_roll` and `mixed_full`.
- Preview cases: `clean_preview`, `nominal_preview_error`, `strong_preview_error`.
- Episode length: 0.2 s, used as a fast probe rather than a final thesis-grade robustness run.

Result:

| Case | Mean score delta | Mean unsafe delta | Mean body RMS delta | Mean pitch RMS delta |
| --- | ---: | ---: | ---: | ---: |
| clean_preview | -0.000377 | 0.000 | -0.000402 | 0.000440 |
| nominal_preview_error | -0.000582 | 0.000 | -0.000527 | 0.000243 |
| strong_preview_error | -0.001209 | 0.000 | -0.000856 | -0.000248 |

Interpretation:

- Seed 42 keeps the deployment gate off and the learned MPC gain schedule remains non-inferior under all three preview cases.
- Seed 43 keeps the deployment gate on, so BP-NMPS exactly matches fixed MPC-lite and contributes zero score/unsafe delta.
- No unsafe-step regression appears in this probe.
- This result should be treated as a pipeline and early robustness probe, not a final broad robustness claim. The formal version should run all five seeds, all held-out scenarios, full episode length, and the same preview-corruption cases.

## 2026-06-08: Preview-Robustness Full Matrix

Command:

```text
python scripts/run_bp_nmps_preview_robustness.py --summary results/mpc_parameter_seed_sweep_formal_gate/mpc_parameter_seed_sweep_summary.json --out results/bp_nmps_preview_robustness
```

Scope:

- Seeds: 42, 43, 44, 45, 46.
- Held-out scenarios: `pothole_roll` and `mixed_full`.
- Preview cases: `clean_preview`, `nominal_preview_error`, `strong_preview_error`.
- Controllers: fixed MPC-lite and deployed BP-NMPS.
- Rows: 60 total evaluation rows, including 30 BP-NMPS rows.

Result:

| Case | Mean score delta | Mean unsafe delta | Mean body RMS delta | Mean pitch RMS delta | Mean energy delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean_preview | -170.081463 | -1.700 | -0.036521 | -0.015179 | -167.299321 |
| nominal_preview_error | -220.063626 | -2.200 | -0.025656 | -0.012234 | -64.931117 |
| strong_preview_error | -230.067552 | -2.300 | -0.017520 | -0.013252 | -67.654083 |

Interpretation:

- The deployed BP-NMPS policy remains non-inferior to fixed MPC-lite across the full preview-corruption matrix used here.
- Gate-on seeds 43, 44, and 46 exactly match fixed MPC-lite, while gate-off seeds 42 and 45 provide the observed improvements.
- The large negative score deltas are mainly caused by reduced unsafe steps under the current tuning score, where unsafe steps have high penalty weight.
- This is now stronger than the earlier preview-robustness probe, but it should still be reported with its exact scenario/case scope rather than as a universal robustness guarantee.
