# Thesis Defense Q&A Notes

This note prepares concise answers for thesis review or defense questions. It follows the current route: direct use of already perceived road unevenness / road-preview sequences for active-suspension control.

## 1. Why Remove PreScan / Visual Road Extraction?

**Likely question:** The proposal mentioned PreScan visual road contour extraction. Why is it not the main route now?

**Answer:** The current thesis focuses on control algorithm design under available road-preview information. Visual reconstruction from camera or PreScan is treated as an upstream perception problem. For active suspension, the controller only needs a road unevenness preview sequence or compact road features; making visual reconstruction the core would shift the thesis from control to perception-platform engineering. The revised route is therefore more focused and experimentally verifiable:

```text
perceived road unevenness / preview sequence -> active suspension controller
```

**Boundary:** Do not claim the thesis solves end-to-end visual perception. State that road-preview acquisition can be added as future work.

## 2. Why Is BP-NMPS an MPC Contribution?

**Likely question:** If BP-NMPS uses reinforcement learning, why call it an MPC method?

**Answer:** BP-NMPS does not replace MPC with a black-box force policy. The learned policy outputs a low-dimensional scaling of MPC-lite gains, and the final actuator command is still computed by the structured MPC-lite controller. This keeps the controller interpretable and reduces high-frequency unsafe force exploration. The deployment gate further preserves fixed MPC-lite when the learned schedule is not supported by calibration evidence.

**Evidence:** The formal 5-seed held-out sweep supports non-regression and small mean improvement: support rate 1.000, mean deployed score delta -0.001314, unsafe delta 0.

**Boundary:** Claim conservative adaptive MPC parameter scheduling, not full MPC replacement.

## 3. Why Is SI-RPPO More Than PPO?

**Likely question:** Is SI-RPPO just PPO with reward shaping?

**Answer:** No. SI-RPPO changes the deployed policy class:

- offline safe-teacher initialization,
- delta action parameterization,
- projection-aware actor optimization,
- state-dependent safe-teacher improvement gate,
- actuator-feasible final action projection.

The final policy is not an unconstrained PPO force output. It is a gated policy that decides when learned correction should override a conservative teacher.

**Evidence:** The strategy evidence table marks projection-aware imitation PPO, offline initialization, safe-teacher gate, and preview-robust deployment versus PPO scratch as supported.

## 4. What If SAC Has Better Return?

**Likely question:** SAC is a strong baseline and may have better return. Does that weaken the SI-RPPO claim?

**Answer:** The thesis claim is not universal return superiority over SAC. In the current matrix, SI-RPPO is a tradeoff against SAC: SAC has stronger return in some settings, while SI-RPPO has smoother actions and lower projection reliance. For a safety-critical active-suspension actuator, deployment quality includes smoothness, feasibility, and reduced hard-projection dependence, not only scalar return.

**Boundary:** Say "deployment-oriented tradeoff against SAC," not "SI-RPPO beats SAC."

## 5. Why Is the Residual Prior Branch Not the Main Contribution?

**Likely question:** The project explored residual prior control. Why is it not claimed as a contribution?

**Answer:** Current ablation evidence is weak or contradictory for the residual prior branch. Keeping it visible as a negative ablation is stronger than forcing it into the final claim. The supported SI-RPPO contribution is the safe-teacher-gated, projection-aware direct policy class.

**Boundary:** Residual prior is useful context and future work, not the final supported innovation.

## 6. Does the Current Evidence Reach a CCF-A-Level Claim?

**Likely question:** Is this enough for a CCF-A-level paper?

**Answer:** The method direction can be framed at CCF-A level because it identifies a real deployment gap in active-suspension RL: online policies must be smooth, actuator-feasible, and safe-teacher aware under preview uncertainty. The current thesis evidence supports a defensible prototype and thesis contribution. A full CCF-A submission would still need broader road distributions, longer training budgets, more seeds, stronger real-road or high-fidelity validation, and deeper comparison against SAC/TD3 variants.

**Boundary:** For the thesis, say "CCF-A-style method framing and innovation direction"; for publication, list the remaining experiments honestly.

## 7. Why Not Use Chrono or VLA Now?

**Likely question:** Could Chrono or a SimplerEnv-like VLA platform make the work stronger?

**Answer:** Chrono can be valuable as a future high-fidelity dynamics/sensor backend, and VLA-style policy training could be explored after a stable control API exists. But building a vision-language-action benchmark would be a separate platform project. The current thesis has stronger evidence by focusing on the control side: road-preview input, MPC parameter learning, and safe-teacher-gated RL.

**Boundary:** Chrono/VLA belongs in future work, not as the main thesis contribution.

## 8. Final Short Defense Statement

本文的核心不是从视觉中重建道路，而是在已获得路面预瞄信息的条件下，研究主动悬架控制器如何安全、可解释、可部署地利用该信息。SPDF 提供传统控制基线；BP-NMPS 将强化学习限制在 MPC 参数调度空间，并通过部署门控保持 fixed MPC-lite 的安全下界；SI-RPPO 则面向直接力控制，提出安全教师初始化、投影感知优化和状态相关门控的策略网络。当前证据支持这一路线的可行性和保守改进，但不声称对所有道路、所有预瞄误差或所有强化学习基线全面最优。
