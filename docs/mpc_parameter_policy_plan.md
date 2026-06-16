# 基线保持的神经 MPC 参数调度方案

日期：2026-06-05

本文档记录当前工程中“强化学习学习 MPC 参数”的毕业论文路线。它和已有 SPDF、MPC-lite、SI-RPPO 线并不冲突：SPDF 是已完成的传统控制基线，MPC-lite 是结构化先验和可解释控制器，神经参数策略负责在已感知路面信息条件下学习 MPC 参数调度，而不是直接替代全部控制律。

## 1. 问题定位

原始设想是通过 PreScan 或类似软件提取视觉/道路轮廓，再做端到端控制。但当前工程已经转向“假设路面信息已被感知”，重点不再放在视觉提取，而是放在：

- 路面/场景上下文如何进入控制器；
- 传统 MPC-like 控制器参数如何随路况自适应；
- 强化学习策略网络如何在不破坏安全基线的前提下改进控制性能。

因此，本文路线把控制问题改写为：

给定路况摘要特征、基准 MPC-lite 参数和车辆配置，策略网络输出归一化参数缩放动作，用于调度 reduced full-car MPC-lite gains。动作不是悬架力本身，而是结构化控制器的参数残差。

## 2. 方法名称与核心思想

建议论文中使用的工作名：

**Baseline-Preserving Neural MPC Parameter Scheduling (BP-NMPS)**

中文可写为：

**基线保持的神经 MPC 参数调度方法**

核心思想：

1. 使用 reduced full-car MPC-lite 作为可解释、安全的结构化基线。
2. 将 RL 动作定义为 MPC gain 的归一化缩放，而不是直接输出四轮主动悬架力。
3. 策略网络输入基准 gains 与路况/场景特征，输出上下文相关的 gain residual。
4. 策略均值头零初始化，使 deterministic policy 初始严格等价于 fixed MPC-lite。
5. 训练中加入 anchor regularization 和 baseline-aware penalties，抑制策略在证据不足时偏离基线。
6. 实验报告区分 sampled best action 与 final deterministic policy，避免把探索采样误写成最终策略性能。

## 3. 当前工程实现

关键文件：

- `envs/mpc_parameter_env.py`
  - 将 MPC gain scaling 封装成 one-step Gym 风格环境。
  - observation 包含 7 维基准 gain、7 维路况/场景特征、3 维 meta 信息。
  - action 为 7 维 `[-1, 1]` 归一化 gain scale。
  - `scenario_limit` 已同步限制环境采样与评估场景，避免 baseline/rollout 不一致。

- `rl/mpc_parameter_policy.py`
  - `MPCParameterGaussianPolicy` 是小型高斯策略网络。
  - mean head 默认零初始化，deterministic action 初始为 0。
  - action=0 表示不改变基准 MPC-lite gains。

- `scripts/train_mpc_parameter_neural_policy.py`
  - 使用 REINFORCE 风格 one-step policy gradient 训练参数策略。
  - 支持 `anchor_coef`、`anchor_final_coef`、`elite_bc_coef`、`deployment_gate`、`baseline_penalty_weight`、`unsafe_penalty_weight`、`suspension_penalty_weight`、`actuator_penalty_weight`。
  - 输出 sampled best、final deterministic policy score、final policy actions CSV。

- `scripts/run_mpc_parameter_learning_benchmark.py`
  - 对比 fixed MPC-lite、random gain search、CEM parameter policy、neural parameter policy。
  - 报告中同时显示 best sampled score 与 policy score。
  - neural 的 Mean Metrics 已与 final deterministic policy 对齐。

- `scripts/run_mpc_parameter_seed_sweep.py`
  - 多 seed 运行 benchmark。
  - 输出 support rate、mean policy delta、unsafe delta、claim gate 与论文表格 `mpc_parameter_seed_sweep_table.tex`。

- `controllers/mpc_parameter_policy.py`
  - `NeuralMPCParameterController` 将训练好的神经参数策略封装为可部署控制器。
  - 该控制器根据当前场景生成 deterministic gain scale，再调用 `ReducedFullCarPreviewController` 计算四角主动悬架力。
  - 可通过 `make_controller("BP_NMPS", ..., checkpoint=...)` 接入统一 evaluation 流程。
  - 统一评估生成的轨迹 CSV 会记录 `mpc_param_action_*` 和 `mpc_param_gain_*`，便于解释不同路况下策略实际调度了哪些 MPC-lite 参数。

- `scripts/summarize_mpc_parameter_schedule.py`
  - 从 BP-NMPS evaluation 轨迹中提取每个场景的参数动作和 scheduled gains。
  - 输出参数调度 CSV、Markdown 报告、gain schedule 图和相对基准 gains 的百分比变化图，用于论文解释性分析。

## 4. 算法流程

1. 载入 full-car MuJoCo 场景配置。
2. 对每个训练 episode，选取一个 road/scenario context。
3. 构造 observation：
   - normalized base MPC gains；
   - scenario type、ISO level、speed、amplitude、wavelength、start time、component count；
   - scenario count、scenario limit、last score。
4. 策略网络输出 normalized action。
5. 环境将 action 映射为 MPC-lite gain scales。
6. 使用更新后的 MPC-lite gains 运行一次 rollout，得到 score 与 comfort/safety/actuator metrics。
7. reward 为负 score，并叠加相对 fixed MPC-lite 的退化惩罚。
8. policy loss 使用 one-step policy gradient，附加 entropy term 与 anchor loss。
9. 训练结束后，用 deterministic policy 单独评估，作为最终策略性能。

## 5. 可写入论文的创新点

当前最稳妥的创新点表述：

### 创新点 1：结构化动作空间

不同于直接输出主动悬架力，本方法让策略网络输出 MPC 参数缩放，使 RL 搜索空间从执行器力空间转为控制器参数空间。这样保留 MPC-lite 的可解释性，并降低策略产生危险高频力指令的风险。

### 创新点 2：基线保持的策略初始化

策略均值头零初始化，使 deterministic policy 初始严格等价于 fixed MPC-lite。该设计保证学习系统在训练初期不会因随机网络输出破坏已有基线控制器。

### 创新点 3：基线感知约束训练

训练 reward 不仅最小化综合 score，还显式惩罚相对 fixed MPC-lite 的 score、unsafe steps、悬架行程和执行器能耗退化。这使参数调度策略遵循“不劣于基线优先”的安全学习原则。

### 创新点 4：采样最优与最终策略分离评估

实验报告同时记录 sampled best action 和 final deterministic policy，避免将训练探索中的偶然最优样本误认为最终可部署策略。

### 创新点 4.5：精英采样动作蒸馏

REINFORCE 训练中，positive-advantage 的采样 MPC 参数动作会通过 `elite_bc_coef` 蒸馏到策略均值。这样可以把探索阶段发现的优良参数调度转化为 deterministic policy 的可部署行为，而不是只停留在 sampled best 记录中。

### 创新点 4.6：基线安全部署门控

训练结束后，deterministic BP-NMPS 会先在训练/校准场景上与 fixed MPC-lite 比较。如果校准得分没有优于基线，checkpoint 会记录 `deployment_gate_active=true`，并在 held-out 评估和控制器部署时回退到 fixed MPC-lite。该门控避免将泛化不稳定的参数调度强行部署，同时保留通过校准的策略改进。

### 创新点 5：多 seed claim gate

使用 seed sweep 检查 deterministic policy 是否稳定不劣于 fixed MPC-lite。claim gate 只有在 support rate、mean policy delta、unsafe delta 同时满足阈值时才标记 supported。

## 6. 实验矩阵

建议正式实验分三层：

### 实验 A：MPC 参数学习基础对照

命令：

```bash
python scripts/run_mpc_parameter_learning_benchmark.py --config configs/mujoco_full_car_corner.yaml --out results/mpc_parameter_learning_benchmark --tuning-trials 16 --cem-iterations 8 --cem-population 16 --neural-episodes 32 --neural-batch-size 4 --neural-anchor-coef 0.1 --neural-anchor-final-coef 0.02 --neural-elite-bc-coef 0.05 --neural-deployment-gate --scenario-limit 3 --eval-scenario-offset 3 --eval-scenario-limit 2
```

对照项：

- fixed MPC-lite；
- random gain search；
- CEM parameter policy；
- neural parameter policy。

主要指标：

- policy score；
- policy delta vs fixed；
- BodyAccRMS；
- Pitch/RollAccRMS；
- MaxSusp；
- ControlAbsEnergy；
- UnsafeSteps；
- EpisodeReturn。

### 实验 B：多 seed 稳定性

命令：

```bash
python scripts/run_mpc_parameter_seed_sweep.py --config configs/mujoco_full_car_corner.yaml --out results/mpc_parameter_seed_sweep --seeds 42,43,44 --tuning-trials 8 --cem-iterations 4 --cem-population 8 --neural-episodes 16 --neural-batch-size 4 --neural-anchor-coef 0.1 --neural-anchor-final-coef 0.02 --neural-elite-bc-coef 0.05 --neural-deployment-gate --scenario-limit 3 --eval-scenario-offset 3 --eval-scenario-limit 2 --min-support-rate 0.8 --max-mean-policy-delta 0.0 --max-mean-unsafe-delta 0.0
```

判定：

- `supported`：deterministic neural policy 至少在 80% seeds 下不劣于 fixed MPC-lite，平均 policy delta 不大于 0，unsafe delta 不大于 0。
- `needs_more_evidence`：不能写“稳定不劣于基线”，只能写“方法可运行，仍需训练或约束改进”。
- seed sweep 同时报告 `noninferiority_status` 和 `improvement_status`。前者表示不劣于 fixed MPC-lite，后者要求平均 policy delta 进一步满足提升阈值，不能把“贴住基线”误写成“性能提升”。
- 使用 `--eval-scenario-offset 3 --eval-scenario-limit 2` 时，前 3 个场景用于训练和参数搜索，后 2 个场景用于 final deterministic policy 与基线的 held-out 评估。

### 实验 C：和 SI-RPPO 主线关系

若参数调度表现稳定，可作为 residual PPO 的结构化 prior：

- fixed MPC-lite prior；
- BP-NMPS adaptive prior；
- safe residual PPO on fixed prior；
- safe residual PPO on adaptive prior。

这一步暂不应作为当前最小毕业目标，除非实验 A/B 已经支持 adaptive prior。

## 7. 当前证据状态

当前 smoke 证据显示：

- one-step smoke 下 anchor policy 的 deterministic output 能严格贴住 fixed MPC-lite；
- claim gate 可正确输出 `supported` 或 `needs_more_evidence`；
- 但当前 smoke 不证明性能超越，只证明基线保持机制和实验链路可运行。

因此，论文中暂时可以写：

> 本研究实现了一种基线保持的神经 MPC 参数调度框架，并建立了固定 MPC-lite、随机参数搜索、CEM 参数策略和神经参数策略之间的可重复评估流程。初步 smoke 实验证明该框架可以在不劣化 fixed MPC-lite 的情况下保持可部署策略输出，后续将通过多场景、多随机种子的正式训练评估其自适应增益。

暂时不应写：

> 神经参数调度已经显著优于 MPC。

除非正式 seed sweep 的 `policy_delta_vs_fixed` 稳定为负，并通过 claim gate。

## 8. 下一步实现优先级

1. 拉长 neural policy training：`neural_episodes >= 32`，`scenario_limit >= 3`。
2. 调整 reward shaping 权重，避免策略只停留在 zero action。
3. 增加 train/eval scenario split，检查泛化而非只记忆场景。
4. 将 BP-NMPS 作为 safe residual PPO 的 adaptive prior 进行对照。
5. 生成毕业论文表格和图：score delta、support rate、comfort/safety/actuator metrics。
