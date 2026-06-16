# Direct Road-Preview Active-Suspension Control Plan

## 1. Decision

Current thesis route: **directly use already perceived road unevenness for control**.

Do not treat PreScan, Chrono vision, camera reconstruction, or a SimplerEnv-like VLA platform as the core contribution. These can be future extensions, but the current control problem should assume that the road preview sequence is already available:

```text
road unevenness / preview height sequence
        -> controller observation
        -> BP-NMPS MPC parameter scheduling
        -> SI-RPPO safe-teacher-gated force residuals
        -> active suspension action
```

This keeps the thesis focused on active-suspension control rather than perception-system construction.

## 2. Road Preview Representation

Use a fixed-horizon road preview sequence:

```text
P_t = [z_r(t + dt), z_r(t + 2dt), ..., z_r(t + Np dt)]
```

For the four-corner full-car case, the preview token can be:

```text
P_t[k] = [z_fl, z_fr, z_rl, z_rr]_{t + k dt}
```

The controller may also use derived road-summary features:

- preview RMS;
- maximum bump/pothole amplitude;
- slope or finite-difference roughness;
- left-right asymmetry;
- front-rear correlation;
- preview confidence under noise/delay/dropout.

## 3. Control Algorithms

### SPDF

SPDF remains the completed traditional feedback baseline. It does not need road preview and should be reported as a classical expert controller.

### Fixed MPC-Lite

Fixed MPC-lite is the structured road-preview baseline. It uses the preview sequence directly and provides interpretable gains.

### BP-NMPS

BP-NMPS learns a 7D MPC gain residual from road/scenario context:

```text
a_theta = [a_kz, a_kdz, a_kpitch, a_kdpitch, a_kroll, a_kdroll, a_kpreview]
```

The deployed controller is:

```text
K_deploy = K_base * scale(a_theta)
u_t = MPC_lite(x_t, P_t; K_deploy)
```

The deployment gate remains part of the method. If the learned deterministic parameter schedule is not better than fixed MPC-lite on calibration scenarios, deployment falls back to fixed MPC-lite.

### SI-RPPO

SI-RPPO is the direct-force RL branch. It should consume road preview and vehicle state, but it should not be unconstrained end-to-end force output. The deployed action is:

```text
u_t = Projection_U(g_t * DeltaPPO(o_t, u_{t-1}) + (1 - g_t) * u_passive)
```

The gate `g_t` opens when road preview excitation and body attitude demand justify active force deviation.

## 4. Experiments To Prioritize

1. BP-NMPS formal seed sweep on held-out road classes.
2. BP-NMPS gain schedule evidence by seed and scenario.
3. BP-NMPS preview robustness under:
   - clean preview;
   - nominal delay/noise/dropout;
   - strong delay/noise/dropout.
4. SI-RPPO ablation against PPO scratch, TD3, and SAC.
5. SI-RPPO gate activation by road class and scenario.
6. Optional direct-force RL comparison only after safety projection and gate evidence remain stable.

## 5. Thesis Wording

Recommended wording:

> 本文不将道路视觉重建作为研究重点，而是假设上游感知模块已经输出沿车轮轨迹的道路不平度预瞄序列。研究重点在于如何将该预瞄序列用于主动悬架控制，并在执行器约束和安全约束下实现可解释、可部署的强化学习控制策略。

Avoid:

> 本文构建了完整的视觉-语言-动作主动悬架平台。

That claim is too broad for the current evidence.

## 6. Practical Next Step

Keep improving the existing MuJoCo full-car road-preview environment and its preview-error model. The immediate formal experiment should be:

```text
python scripts/run_bp_nmps_preview_robustness.py --summary results/mpc_parameter_seed_sweep_formal_gate/mpc_parameter_seed_sweep_summary.json --out results/bp_nmps_preview_robustness
```

Then update:

```text
python scripts/build_thesis_claim_matrix.py --bp-nmps-preview-robustness-report results/bp_nmps_preview_robustness/bp_nmps_preview_robustness.md
```
