# 仿真环境分析 + 半车→整车控制方法迁移

> 整理：2026-06-16 | 依据：源码行号 + 本地实测数据 | 工具：Alma

---

## 一、三套仿真环境的本质区别（依据源码）

| 环境 | 动力学由谁计算 | 本质 | 自由度 | 关键代码 |
|---|---|---|---|---|
| `HalfCarEnv` | 自写方程 + RK4 积分 | **纯数学建模** | 4（半车2D） | `models/half_car.py:165 rk4_step` |
| `MuJoCoHalfCarEnv` | **仍是 HalfCarEnv 的 RK4**，MuJoCo 只 `mj_forward` 渲染 | **数学建模 + 可视化壳** | 4 | `envs/mujoco_half_car_env.py:60 step`（继承父类物理，只调 mj_forward 更新渲染） |
| `MuJoCoFullCarEnv` | **MuJoCo 引擎 `mj_step` 真积分** | **真·多体物理仿真** | 7+（整车） | `envs/mujoco_full_car_env.py:228 mujoco.mj_step` |

### 1.1 关键区分（避免误判）

- `MuJoCoHalfCarEnv` 是"披着 MuJoCo 皮的数学建模"——物理由父类 RK4 算，MuJoCo 只负责画图。
- `MuJoCoFullCarEnv` **不是**"数学建模 + 侧倾"，而是真正调用 MuJoCo 物理引擎 `mj_step`（`envs/mujoco_full_car_env.py:228`），读广义坐标 `qpos/qvel`。车身侧倾、四轮独立运动及其耦合由引擎从多体约束动力学中**求解涌现**，而非人工写方程。

这与"数学半车加一个 roll 方程"有本质差别：
- 数学半车加侧倾 = 手写 roll 方程进 RK4，仍是解析模型，所有交叉耦合项要自己推。
- MuJoCo 整车的侧倾 = 引擎通过质量矩阵 / 约束求解器自动算出，含真实接触动力学、关节限位、半隐式积分。

### 1.2 数学半车动力学方程（依据 `models/half_car.py:140-141`）

车身垂向：
```
ddz_b = [-(F_sf + F_df + F_sr + F_dr) + u_b] / m_b
```
俯仰：
```
ddθ = [-a(F_sf+F_df) + b(F_sr+F_dr) + u_θ] / I_p
```
其中 F_s = k·Δy（弹簧）、F_d = c·Δẏ（阻尼）均线性。RK4 4 阶积分（`models/half_car.py:165`）。

---

## 二、量化一致性验证（本地实测，依据 results/env_comparison/）

工况：ISO Class B 随机路面，passive（零主动力），5 episodes × 400 steps，固定路面 seed。

| 物理量 (RMS) | 数学半车 HalfCarEnv | MuJoCo 整车 FullCar | 比值 |
|---|---|---|---|
| 车身垂向加速度 (m/s²) | 0.0178 | 0.0227 | 1.27× |
| 俯仰角加速度 (rad/s²) | 0.0328 | 0.0488 | 1.49× |
| 侧倾角加速度 (rad/s²) | N/A（2D 无侧倾） | 0.0292 | 仅整车 |
| 平均悬架行程 (m) | 0.00054 | 0.00232 | 4.3× |

**结论**：
- 车身/俯仰加速度处于同一数量级（1.3~1.5×），验证 MuJoCo 整车动力学与经典半车模型的一致性。
- 整车响应普遍偏大，源于多体耦合 + 四角独立 + 侧倾自由度（引擎求解涌现，非人工建模）。
- 悬架行程差 4.3× 是因为整车单角行程受侧倾影响，半车为前后等效——建模维度差异的合理体现。

---

## 三、半车 → 整车 控制方法迁移（依据源码）

经典控制器（SPDF / MPC-lite）从半车迁移到整车，通过**三层桥接**，结构不变、维度扩展。

### 桥接 1：动作空间——axle 模式当翻译官（依据 `mujoco_full_car_env.py:103-109`）

整车两种动作模式：
- **axle 模式**：2 维 = [前轴力, 后轴力] ← 完全对应半车的前/后作动器
- **corner 模式**：4 维 = [左前, 右前, 左后, 右后]

互转（依据 `mujoco_full_car_env.py` 末尾 `_action_from_corner` + `reduced_full_car.py` 返回段）：
```
corner → axle: axle = [corner_fl + corner_fr, corner_rl + corner_rr]
axle → corner: 前轴力分配给左前/右前，后轴力分配给左后/右后
```
半车设计的"前后控制律"零改动即可经 axle 模式对接整车。

### 桥接 2：状态空间——几何投影（依据 `controllers/reduced_full_car.py:50-62`）

半车只有 z（垂向）和 θ（俯仰）。整车多了 roll（侧倾）。用刚体几何关系把车身三模态投影到四角位移：
```
z_fl = z + a·pitch + (track/2)·roll
z_fr = z + a·pitch − (track/2)·roll
z_rl = z − b·pitch + (track/2)·roll
z_rr = z − b·pitch − (track/2)·roll
```
（速度同理，reduced_full_car.py:59-62）

- a/b = 质心到前/后轴距离（半车原有参数）
- track/2 = 半轮距（整车新增——侧倾靠它进来）

**本质**：半车"垂向-俯仰"二自由度反馈律，加一项 ±(track/2)·roll 即扩展成整车"垂向-俯仰-侧倾"三自由度律。

### 桥接 3：控制律——模态反馈 + 预瞄前馈（依据 `reduced_full_car.py:22-28, 36-62`）

整车 MPC-lite 控制律（`ReducedFullCarPreviewController`）：
```
F = k_z·z + k_dz·ż            (垂向模态)
  + k_pitch·θ + k_dpitch·θ̇     (俯仰模态)
  + k_roll·roll + k_droll·droll  (侧倾模态，整车新增)
  + k_preview·road_preview     (预瞄前馈)
```
默认增益（default.yaml `reduced_full_car_mpc`）：
k_z=1200, k_dz=1800, k_pitch=950, k_dpitch=700, k_roll=850, k_droll=650, k_preview=18000

半车版只有前 2 组 + 预瞄；整车版多了 k_roll/k_droll 侧倾组，其余同构。

### 三条迁移路线总览

| 方法 | 迁移方式 |
|---|---|
| 经典控制器（SPDF/MPC-lite） | 手写几何投影 + 模态反馈扩展（上述三层） |
| BP-NMPS | 调度 MPC-lite 增益（基于上述结构） |
| **SI-RPPO（核心创新）** | **端到端 RL 直接学** obs→4角力，不用几何投影，但用整车 obs 训练 |

---

## 四、论文措辞边界（避坑）

- MuJoCoFullCarEnv 虽更真实，**仍是仿真**，不可宣称"验证了实车性能"。
- 正确表述："MuJoCo 高保真整车仿真，通过多体约束求解验证策略在更复杂动力学下的泛化性"。
- Chrono / PreScan / VLA 为未来扩展，非论文核心。
- 论文务必区分 `MuJoCoHalfCarEnv`（数学建模+渲染）与 `MuJoCoFullCarEnv`（真引擎求解），勿混淆。
