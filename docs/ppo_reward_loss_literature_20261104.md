# PPO 奖励与损失改进：依据、实现和验证

新实验配置为 `configs/cosim_ppo_literature.yaml`，通过 `train_cosim_ppo.py --config` 启用。
旧配置与旧奖励公式保留。新方案是待验证的研究假设，不代表已证明减振性能提升。

## 已核对的来源

1. Dridi, Hamza & Ben Yahia (2023), *A new approach to controlling an active suspension
   system based on reinforcement learning*, DOI [10.1177/16878132231180480](https://journals.sagepub.com/doi/full/10.1177/16878132231180480)。
   已读取全文：式 (18)、表 5、表 8。该文将加速度、车身/车轮对路面的位移误差与
   控制力纳入奖励，并做多种奖励消融。使用四分之一车模型，控制力范围与本项目不同。
   参考其多目标设计及 PPO clip=0.2、critic_lr=1e-4；不直接复制有量纲权重或性能百分比。
2. Schulman et al. (2017), *Proximal Policy Optimization Algorithms*,
   [arXiv:1707.06347](https://arxiv.org/pdf/1707.06347)，已读取原论文；采用裁剪代理目标、
   GAE 与价值回归/熵正则的基本结构。本项目原先已实现这些基本项，不把已有部分称为新增算法。
3. [Stable-Baselines3 官方 PPO 实现](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/ppo/ppo.html)：
   参考近似 KL 的计算和超过 1.5 倍目标的早停；优势归一化本项目原已具备。
4. [OpenAI Baselines PPO2 原始实现](https://github.com/openai/baselines/blob/master/baselines/ppo2/model.py)：
   参考原始/裁剪价值误差取较大值的 value loss。这里采用 PPO2 形式，
   不是声称与 SB3 当前 value loss 实现完全相同。

以下全文访问受限，未将其未读公式当作实现依据；用户可补充 PDF：

- *Reinforcement-Learning-Based Vibration Control for a Vehicle Semi-Active Suspension System via the PPO Approach*，
  [DOI 10.3390/app12063078](https://doi.org/10.3390/app12063078)，MDPI 返回 429。
- *Prior-Guided Residual Reinforcement Learning for Active Suspension Control*，
  [DOI 10.3390/machines13110983](https://doi.org/10.3390/machines13110983)，MDPI 返回 429。
- *Delay-resilient robust control of automobile active suspensions via deep reinforcement learning*，
  [DOI 10.1016/j.rineng.2026.109381](https://doi.org/10.1016/j.rineng.2026.109381)，全文入口返回 403。

## 本项目奖励（自主适配，非论文原系数）

每步奖励为 `r = -0.1 * sum(w_i * cost_i)`。各项先用固定物理尺度无量纲化。
所有均值在四轮或两轴之间取，奖励在 10 ms 控制周期末计算。

| 项目 | cost 定义 | 权重 |
|---|---|---:|
| 垂向加速度 | `(az / 2 m/s²)²` | 1 |
| 俯仰加速度 | `(pitch_accel / 1 rad/s²)²` | 0.05 |
| 悬架行程 | `mean(((jounce-jounce0)/80 mm)²)` | 0.1 |
| 动态轮胎载荷 | `mean(((Fz-Fz0)/abs(Fz0))²)` | 0.2 |
| 主动力 | `mean((u_executed/2000 N)²)` | 0.005 |
| 力变化 | `mean(((u_executed-u_previous)/150 N)²)` | 0.01 |
| 动作投影 | `mean(((u_raw-u_executed)/2000 N)²)` | 0.05 |
| 行程越限 | `mean(relu(abs(normalized_travel)-1)²)` | 10 |
| 轮胎接近卸载 | `mean(relu(0.1-Fz/abs(Fz0))²)` | 10 |

初始化/接触监督强制抑制动作时，投影惩罚为零，避免惩罚策略无法改变的初始化过程。
安全相关惩罚不能替代执行约束或稳定性证明。初始载荷和行程用原环境的平衡参考，
并非制造商提供的机械行程端点；轮胎载荷指标是本项目采用的路面附着代理量，
不等同于论文的轮胎位移量。没有宣称该奖励是 ISO 加权舒适性指标。

旧奖励的俯仰项约占总代价 48%，此次降低其相对权重；加速度仍为主要目标，
行程/附着和执行代价防止只优化车身加速度。0.1 的整体缩放用于控制 critic 回归尺度，
不能跨新旧奖励直接比较回合 return。权重和归一化常数均需消融验证。

## PPO 损失和探索

`L_actor = -mean(min(ratio*A, clip(ratio,1-eps,1+eps)*A)) - c_entropy*H`

`L_value = c_value * mean(max((V-R)², (V_old+clip(V-V_old,-vclip,vclip)-R)²))`

Actor 和 critic 使用独立优化器；返回值分别记录两者损失。新增配置：

- `clip_ratio=0.2`、`target_kl=0.015`；批次近似 KL 超过 `0.0225` 时停止后续更新，
  并记录实际步数。KL 早停是更新保护，不能保证每次更新后的 KL 必然小于阈值。
- `value_clip=0.2`，相对 rollout 保存的旧价值裁剪，单位是缩放后 return。
  `critic_lr=1e-4`；`value_coef=0.5` 保持。价值裁剪可设为 0 做消融。
- `entropy_coef=1e-4`；使用当前策略重新采样估计 tanh 后归一化动作分布的熵，
  不用旧 rollout 动作的负对数概率冒充当前策略熵，也不把未压缩高斯熵当作实际动作熵。
- 初始 `log_std=-3`，范围 `[-5,-2]`。在均值接近零时，2000 N 动作尺度对应约 100 N
  初始标准差。原来约 1200 N 的探索与每 10 ms 最多 150 N 的变化限制不匹配。
  本次保留绝对力动作定义和执行限幅，先收窄探索；投影仍存在，后续可另行比较增量动作。
- 新增近似 KL、clip fraction、explained variance、优化步数和早停标记。
- `gamma=0.99`、`lambda=0.95` 保持本项目值；不照抄不同时钟下的折扣因子。

新增选项默认关闭或未设置，旧权重通过默认值补全检查配置；新 checkpoint 额外保存
reward 配置并在加载时核对，防止评估时误用旧奖励。`--checkpoint` 仍只是权重加载，
不承诺恢复优化器状态。

## 验证与实验边界

运行 `python scripts/check_cosim_reward_loss.py`：检查平衡点零代价、加速度二次响应、
行程/卸载惩罚、监督豁免、真实 Episode 桥中的奖励序列化、有限 PPO 更新及 KL 超限时
不修改 actor。另运行全库 Ruff 和 `scripts/check_project.py`，七组通过，未运行 pytest。

第一轮真实 CARLA/CarSim 冒烟为三个 5 s 回合（`results/cosim_ppo_literature_smoke_20261104_v1`），
全部通过训练数据/执行动作审计。非监督步中动作被投影的比例约 35.8%、44.0%、42.9%，
明显低于旧试验约 99%；这证明探索幅度更匹配执行约束，尚不证明减振效果改善。
5 s 不能覆盖完整减速带/坑洼，因此另执行三个 10 s 回合，结果见 v2 目录。
v2 共 3,000 步，三次 actor 更新和全部物理执行记录审计通过；投影比例为
39.6%、42.9%、42.5%，探索标准差约 96–102 N。加速度项占代价约 68–80%，
俯仰项约 18–23%。最终近似 KL 为 0.0027–0.0045，critic explained variance
仍接近零，说明短回合检查尚不能证明价值估计或控制性能收敛。
旧 12 回合训练的审计也重新通过，验证新增默认字段没有破坏旧记录的解析与策略一致性。

本次只使用车辆状态与原深度预瞄进行短回合验证；Qwen 延迟不作为本轮奖励调整的混杂因素。
此前组合路面已参与诊断与设计判断，下一轮正式泛化评估应增加全新路面和随机种子，
并做旧奖励/新奖励、新 loss/旧 loss、探索幅度的独立消融。

完整短回合命令：

```powershell
conda run --no-capture-output -n suspension-control python scripts/train_cosim_ppo.py --config configs/cosim_ppo_literature.yaml --carla-python E:/anaconda3/envs/carla0911/python.exe --episodes 3 --seconds 10 --road-cycle wave bump pothole --output results/cosim_ppo_literature_new
```

修改仅为本轮联合仿真研究提供新的奖励/更新配置，未替换 Orin 上此前部署的权重，
未将短回合模型声称为收敛模型。新配置可直接继续开展受控训练实验。
