# 毕业论文方法与实验章节草稿

本文档用于把当前工程中的 SPDF、BP-NMPS 与 SI-RPPO 证据整理成可迁移到毕业论文的方法和实验叙事。它不是最终定稿，所有强结论仍应以完整实验矩阵为准。

## 1. 研究路线定位

本课题不再把 PreScan 或具体视觉算法中的道路轮廓重建作为核心研究内容，而是假设上游感知模块已经输出可用于控制的道路预瞻序列或路况摘要。论文重点放在两个问题上：

1. 已感知道路预瞻信息如何进入主动悬架控制器。
2. 强化学习策略如何在执行器约束、安全约束和专家基线存在时稳定改进控制性能。

因此，论文中的算法层次可以组织为：

- SPDF：已完成的传统专家控制基线，用于给出无预瞻反馈控制参考。
- fixed MPC-lite：结构化、可解释的预瞻控制基线。
- BP-NMPS：强化学习学习 MPC 参数的保守自适应控制器。
- SI-RPPO：面向直接力输出策略网络的安全教师门控在线 PPO 方法。

## 2. BP-NMPS：基线保持神经 MPC 参数调度

BP-NMPS 的核心思想是让策略网络输出 MPC-lite 参数缩放，而不是直接输出主动悬架力。给定路况摘要、基准 MPC-lite 参数和车辆配置，策略网络输出归一化动作：

```text
a_theta = [a_kz, a_kdz, a_kpitch, a_kdpitch, a_kroll, a_kdroll, a_kpreview]
```

该动作被映射为 MPC-lite gain scale，并形成部署控制器：

```text
K_deploy = K_base * scale(a_theta)
u_t = MPC_lite(x_t, preview_t; K_deploy)
```

该设计有三个优点：

- 将强化学习搜索空间从执行器力空间转为控制器参数空间，降低高频力指令风险。
- 保留 MPC-lite 的物理结构和可解释性，便于报告实际调度的 gains。
- 可通过部署门控保证未被校准集支持的策略不会劣化 fixed MPC-lite。

训练中采用基线保持机制：

- 策略均值头零初始化，使 deterministic policy 初始严格等价于 fixed MPC-lite。
- anchor regularization 约束早期策略偏离基线。
- positive-advantage sampled action distillation 将探索阶段发现的优良参数动作蒸馏到 deterministic policy。
- calibration deployment gate 在训练/校准场景上验证 deterministic policy；若没有优于 fixed MPC-lite，则部署时回退 fixed MPC-lite。

当前 5 seed held-out 结果显示，部署后的 BP-NMPS 在所有 seed 上不劣于 fixed MPC-lite，support rate 为 1.000，mean deployed score delta 为 -0.001314，unsafe delta 为 0，gate rate 为 0.600。该结果支持的表述是：

> BP-NMPS 是一种保守的、自适应 MPC 参数调度器。它可以在校准证据支持时小幅调整 MPC-lite gains，并在证据不足时回退 fixed MPC-lite，从而避免将不稳定神经策略直接部署。

不能过度声称：

- 不能说 raw neural policy 在所有 seed 上均优于 fixed MPC-lite。
- 不能说 BP-NMPS 已经全面替代 MPC。
- 当前 preview robustness 已完成 5 seed、2 held-out 场景、3 preview case 的 full matrix，但仍应限定在该场景范围内表述。

## 3. SI-RPPO：安全教师门控的投影感知在线 PPO

SI-RPPO 用于直接力输出强化学习控制。其创新点不在于“使用 PPO”，而在于改变策略的可部署形式，使在线 PPO 在执行器约束下更稳。

最终执行动作写为：

```text
u_t = Projection_U(g_t * DeltaPPO(o_t, u_{t-1}) + (1 - g_t) * u_passive)
```

其中：

- `DeltaPPO(o_t, u_{t-1})` 是增量参数化的 PPO 主动修正。
- `u_passive` 是安全教师动作。
- `Projection_U` 表示执行器和安全约束投影。
- `g_t` 是 safe-teacher improvement gate。

门控系数 `g_t` 由道路预瞻激励、车身姿态加速度需求和安全裕度共同决定。低激励或低置信场景下，`g_t` 接近 0，策略保持接近安全教师；高激励场景下，`g_t` 增大，允许策略输出主动悬架修正力。

当前 gate evidence 显示：

| Scenario | Mean gate | Active >0.1 | Gate p95 |
| --- | ---: | ---: | ---: |
| class_b_correlated | 0.000 | 0.000 | 0.000 |
| class_c_correlated | 0.000 | 0.000 | 0.000 |
| pothole_roll | 0.036 | 0.080 | 0.675 |
| mixed_full | 0.400 | 0.958 | 0.570 |
| sine_roll | 0.907 | 0.978 | 1.000 |

这说明策略并非始终施加主动力，而是根据场景需求打开。该结果可用于支持 CCF-A 级论文叙事中的“策略网络结构创新”：

> SI-RPPO 将离线模仿初始化、增量动作参数化、投影感知约束和安全教师门控组合为一个可部署策略类，使 PPO 的主动修正必须在状态相关门控下获得偏离安全教师的机会。

当前 fair full-matrix 证据中，gated BC-PPO 相比 PPO scratch 改善 return、unsafe steps、body/pitch/roll comfort、action smoothness、actuator tracking 和 projection error。需要谨慎的是，SAC 仍是强基线，因此论文不应声称 SI-RPPO 全面优于所有 off-policy 方法；更稳妥的主张是 SI-RPPO 解决了在线 PPO 在主动悬架执行器约束下的安全性和可部署性问题。

## 4. 实验章节建议结构

实验章节可以按以下顺序组织：

1. 仿真环境与评价指标：full-car dynamics、四角路面预瞻、actuator lag/rate limit、安全约束、score 组成。
2. SPDF/PID/MPC-lite 基线结果：说明传统专家控制与预瞻结构化控制的参考地位。
3. BP-NMPS 参数学习实验：fixed MPC-lite、random search、CEM、neural scheduler、deployment gate。
4. BP-NMPS 可解释性：gain schedule evidence，展示 gate-on 回退与 gate-off 小幅 gain 调整。
5. BP-NMPS 预瞻误差矩阵：clean、nominal error、strong error 下的 5 seed held-out 结果。
6. SI-RPPO 消融实验：PPO scratch、BC-PPO、residual variants、TD3/SAC baselines。
7. SI-RPPO gate evidence：按场景报告 gate mean、active ratio、projection error 和 action smoothness。
8. 局限性：正式 full robustness、更多随机种子、更长训练周期、preview corruption 和硬件在环仍需补充。

## 5. 当前可引用证据文件

- `docs/thesis_claim_matrix.md`
- `docs/bp_nmps_experiment_log.md`
- `results/bp_nmps_evidence_table`
- `results/bp_nmps_schedule_evidence/bp_nmps_schedule_evidence.md`
- `results/bp_nmps_preview_robustness/bp_nmps_preview_robustness.md`
- `results/si_rppo_gate_evidence/si_rppo_gate_evidence.md`
- `results/si_rppo_gate_evidence/si_rppo_gate_evidence.tex`
- `results/si_rppo_gate_evidence/si_rppo_gate_method_summary.md`

## 6. 下一步必须补齐的正式实验

1. BP-NMPS 更长训练预算下的 seed sweep，确认 deployment gate rate 与 mean improvement 是否稳定。
2. BP-NMPS 更强预瞄误差和更多 held-out road classes 下的扩展矩阵。
3. SI-RPPO learned/uncertainty-aware gate，替代手工门控或作为对照。
4. SI-RPPO 对 SAC/TD3 的更长 horizon 与 harder-road 对照，避免 passive-like 策略成为过强短期基线。
5. 将最终 evidence table、gate table 和 schedule table 写入毕业论文 LaTeX。
