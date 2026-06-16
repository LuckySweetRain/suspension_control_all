# Reward 归一化 + Clip 诊断与修复

> 整理：2026-06-16 | 依据：源码行号 + 本地实测 | 工具：Alma

---

## 一、问题：reward 权重与实际贡献严重脱节（依据本地实测）

原 config（mujoco_full_car_corner.yaml）reward 权重：
```
body_acc: 1.0, pitch_acc: 0.4, roll_acc: 0.25, suspension: 18.0, tire_load: 5e-9, control: 0.05
```

实测各 component 占比（依据 results/reward_norm_ablation/）：

| Component | 旧权重 | passive 占比 | random 占比 | 问题 |
|---|---|---|---|---|
| body_acc | 1.0 | 16.2% | 21.3% | 以为是主目标，实际不是 |
| pitch_acc | 0.4 | 30.0% | 59.9% | 权重小却主导（random） |
| roll_acc | 0.25 | 6.7% | 17.5% | |
| suspension | **18.0** | 16.8% | **0.8%** | 权重全场最大却几乎不起作用 |
| tire_load | **5e-9** | **30.4%** | 0.2% | 权重全场最小（passive 却占 30%，纯靠物理量级 ~1e5 撑） |

**根因**：reward 各项物理量量级差异巨大（车身加速度² ~O(1e-3)，轮胎力² ~O(1e5)），权重数值完全被物理量级淹没，失去物理意义。

---

## 二、修复：硬编码归一化 + 语义权重（依据源码改动）

### 思路（两层结构）
```
component = semantic_weight × normalization_const × physical_quantity²
```
- normalization_const = 1 / passive 跨场景 mean（让每项归一到 ~O(1)）
- semantic_weight = 真实优先级占比

### 归一化常数（passive 跨 5 场景标定，无随机化）
注入到 `envs/mujoco_full_car_env.py` _reward 的 `_RNORM`：
```
body_acc:  1.9449e3   (= 1/5.142e-4)
pitch_acc: 4.1925e2   (= 1/2.385e-3)
roll_acc:  1.1738e3   (= 1/8.519e-4)
suspension:3.3782e4   (= 1/2.960e-5)
tire_load: 5.173e-6   (= 1/1.933e5)
```

### 语义权重（config mujoco_full_car.yaml）
```
body_acc: 0.40, pitch_acc: 0.15, roll_acc: 0.10, suspension: 0.20, tire_load: 0.10, control: 0.05
```
目标占比：舒适 65%（body40+pitch15+roll10）+ 行程 20% + 操稳 10% + 能耗 5%

### 验证（依据 results/reward_norm_ablation/）
归一化后 passive 占比：body 42.1% / pitch 15.8% / roll 10.5% / susp 21.1% / tire 10.5%，与目标误差 <2.5%。

---

## 三、训练发现：偶发 reward 爆炸（依据本地训练 log）

归一化后重跑 4 个 RL baseline（100ep）：
- 稳态 return 从原来几十万降到 -几千（off-policy）— 归一化有效
- 但偶发单 episode return 爆到 -180万~-230万

**根因**：warmup/探索阶段随机施 ±5000N 暴力力 → body_acc 单步爆到 25516（无上界）→ reward 尖峰。

---

## 四、PPO 崩溃专项诊断（依据 rl/ppo.py 源码）

| 检查项 | 状态 | 依据 |
|---|---|---|
| ratio clip（PPO 自带） | ✅ 有 0.2 | ppo.py 训练段 torch.clamp(ratio, 1-ε, 1+ε) |
| advantage 归一化 | ✅ 有 | ppo.py:219 (adv-mean)/(std) |
| GAE 实现 | ✅ 标准正确 | ppo.py:337 finish_ppo_trajectory |
| **value/return 归一化** | ❌ **没有** | ppo.py:202 returns 直接用；value_loss = MSE(value(obs), returns) |
| reward clip | ❌ 没有 | — |

**链条**：偶发暴力 → 单步 reward 爆 → GAE 累加出 -190万 returns → value MSE loss 被极端样本主导 → value 拟合跨 3 个数量级失败 → advantage 全错（虽归一化但符号/相对大小已错）→ PPO 学瞎。
off-policy (sac/td3/ddpg) 有 replay buffer 缓冲，稳态到几千；PPO on-policy 最敏感，稳态还在 -30万。

**关键认知**：PPO 自带 ratio clip 是信任域机制，**不管 reward 数值大小**。reward 尺度问题要靠 reward clip 或 value normalization 解决。

---

## 五、方案 A：reward clip（已落地）

### 标定（依据本地实测单步 reward 分布）
| 工况 | mean | p95 | p99 | max |
|---|---|---|---|---|
| passive | 9.3 | 46 | 145 | 372 |
| mild(±20%力) | 358 | 1072 | 1892 | 5521 |
| warmup暴力 | — | — | — | 25000+ |

选 clip = [-200, 0]：env _reward 返回处加 np.clip，config reward.clip 开关控制。

### 验证（依据本地实测）
- random 300步 return：196万 → 5.97万（降 30 倍）
- passive 几乎不受影响

### clip 后重跑 4 算法结果（依据 /tmp/rltrain_clip/ log）
| 算法 | clip 前 | clip 后稳态 | 状态 |
|---|---|---|---|
| ddpg | -211万爆炸 | -2852~-3205 | ✅ 学到温柔控制 |
| td3 | -189万爆炸 | -3873~-7561 | ✅ 学到 |
| sac | -186万爆炸 | -7.7~8万卡死 | ⚠️ 没学好 |
| ppo | -30万崩 | -80000（撞死 clip 上限 400×200） | ❌ 根本没学 |

---

## 六、结论与后续

- **归一化** ✅ 必要且正确，把 reward 量级从几十万降到几千。
- **reward clip** ✅ 灭了爆炸，救活 off-policy 的 ddpg/td3。
- **PPO（+部分 sac）** ❌ 仍不行 —— 根因是 value target 没归一化，clip 反把暴力动作梯度信号压平，更学不动。

### 待办（治本，未做）
- 方案 B：PPO value/return normalization（PPO 标配，SB3/OpenAI baselines 都有），让 value 预测归一化后的 return。
- 方案 C：打开 reward.action_delta 权重（现成项，权重却是 0），抑制策略输出暴力抖动力。

### 备份文件
- env: mujoco_full_car_env.py.bak_rewardnorm_* / .bak_clip_*
- config: mujoco_full_car.yaml.bak_* / .bak_clip_*
