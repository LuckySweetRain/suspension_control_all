# SI-RPPO Strategy-Network Innovation Framing

This note gives a conservative, thesis-ready framing for the RL strategy-network contribution. It is written for the direct road-preview active-suspension route: road unevenness preview is assumed to be available from an upstream perception module, and the contribution is the deployable control policy design.

## Core Claim

The SI-RPPO contribution is not "using PPO for active suspension." The contribution is a deployable policy class for safety-critical active-suspension force control:

```text
road preview + vehicle state
        -> offline safe-teacher initialization
        -> delta action PPO update
        -> projection-aware optimization
        -> state-dependent safe-teacher improvement gate
        -> actuator-feasible active-suspension force
```

The key research claim is:

> SI-RPPO converts online PPO from an unconstrained direct-force learner into a projection-aware, safe-teacher-gated policy class. The policy is allowed to deviate from the safe teacher only when the preview/state evidence supports active correction, and the deployed force remains filtered through actuator feasibility constraints.

## What Makes It a Strategy-Network Contribution

The strategy network is changed at the behavior level, not only at the reward level:

- **Offline teacher initialization** gives PPO a stable behavior prior before online updates.
- **Delta action parameterization** predicts a correction around the previous/safe action instead of high-frequency absolute forces.
- **Projection-aware loss terms** penalize policies that rely heavily on safety projection, pushing the actor toward naturally feasible actions.
- **Safe-teacher improvement gate** makes the final action state dependent: quiet or low-confidence road states stay near the safe teacher, while high-excitation states permit active correction.
- **Preview-robust evaluation** tests the deployed policy under clean and corrupted preview inputs, rather than only reporting training return.

This supports a stronger thesis framing than "reward-shaped PPO" because the deployable policy form itself encodes safety, smoothness, and teacher fallback behavior.

## Evidence That Supports the Claim

Use these evidence anchors in the thesis:

- `results/si_rppo_e20_improvement_gate_fair_baselines/si_rppo_claim_report.md`
- `results/si_rppo_gate_evidence/si_rppo_gate_evidence.md`
- `results/si_rppo_preview_robustness/si_rppo_preview_robustness.md`
- `results/si_rppo_strategy_evidence/si_rppo_strategy_evidence_table.md`
- `docs/thesis_latex_snippets.tex`

Current supported statements:

- Against PPO scratch, SI-RPPO improves return, unsafe steps, comfort metrics, action smoothness, actuator tracking, and projection error.
- Against TD3 in the current preview-robustness matrix, SI-RPPO improves return, unsafe steps, body RMS, action smoothness, and projection-related behavior.
- Against SAC, SI-RPPO is a tradeoff: return is lower in the current matrix, while action smoothness and projection error improve. This should be presented as a deployment-quality tradeoff, not as broad superiority.
- Gate evidence is state dependent: low-excitation road classes have near-zero gate activation, while high-excitation cases such as sine-roll and mixed-full open the gate substantially.

## CCF-A-Level Positioning

The defensible high-level positioning is:

> Existing RL baselines for active suspension often optimize scalar return under continuous-force action spaces, but safety-critical deployment also requires smooth commands, low reliance on hard projection, and a mechanism for deciding when learned actions should override a conservative teacher. SI-RPPO addresses this gap by coupling safe-teacher initialization, projection-aware PPO, and state-dependent deployment gating under road-preview observations.

This is a method-level claim. It does not require saying that SI-RPPO is universally better than every off-policy algorithm. The current evidence is strongest for:

- online PPO deployment quality,
- smooth actuator-feasible control,
- state-dependent deviation from a safe teacher,
- robustness to preview corruption under the tested matrix.

## Boundaries to Keep

Do not claim:

- SI-RPPO is universally superior to SAC.
- The residual prior branch is a final supported contribution.
- The method solves visual road perception or builds a VLA active-suspension platform.
- The current short-budget evidence proves all-road or real-vehicle robustness.

Safe wording:

> In the current matrix, SI-RPPO gives a supported improvement over PPO scratch and TD3, and shows a deployment-oriented tradeoff against SAC: lower return in the tested setting but smoother actions and lower projection reliance. Therefore, the thesis positions SI-RPPO as a projection-aware safe-teacher-gated PPO policy class, not as a universal RL optimizer.

## Recommended Thesis Paragraph

本文的强化学习贡献不是简单地将 PPO 应用于主动悬架，而是设计了一类面向部署约束的策略网络。SI-RPPO 首先通过安全教师策略进行离线初始化，使策略具有稳定的初始行为；随后采用增量动作参数化与投影感知损失，降低高频力指令和执行器投影依赖；最终通过状态相关的安全教师门控决定 PPO 主动修正与安全教师动作之间的插值比例。实验结果表明，在当前路面预瞄鲁棒性矩阵中，SI-RPPO 相比 PPO scratch 和 TD3 在安全步数、动作平滑性和投影误差方面具有明显优势；相比 SAC 则表现为部署质量与回报之间的 tradeoff。因此，本文将 SI-RPPO 定位为一种安全教师门控、投影感知的在线 PPO 策略类，而不是声称其在所有道路和所有强化学习基线下均全面最优。
