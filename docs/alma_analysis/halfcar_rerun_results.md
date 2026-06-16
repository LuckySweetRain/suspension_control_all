# 半车全面重跑结果（reward归一化+clip+BC初始化+修复路面）

> 2026-06-16 | 依据：本地实测 results/halfcar_eval_norm/metrics.csv | 工具 Alma

## 背景
MuJoCo 整车作动器力学有结构性问题（改be vs 主动力不等效差15%，skyhook两方向都减不了振），
决策：所有方法收缩到数学半车 HalfCarEnv（白盒、力学已验证正确：skyhook c=5000 减振37%）。

## 四件套修复
1. 路面 SDE bug：road_profiles.py:55 b·w·dt → b·w·√dt（RMS 0.62→19.6mm）
2. reward 归一化：half_car_env.py 注入 _RNORM（passive跨场景mean倒数）
3. reward clip [-200,0] 防极端动作污染value
4. BC 初始化：default.yaml expert.enabled=True（半车天然支持，MuJoCo被skip）

## reward 权重（语义优先级，无roll）
body0.45 / pitch0.20 / susp0.25 / tire0.10 / control0.05
验证 passive 占比精准命中 45/20/25/10 ✅

## 训练收敛（100ep，BC初始化）
| 算法 | warmup起步 | 收敛末态 | 对比MuJoCo |
|---|---|---|---|
| sac | -20965 | -305~-1414 | MuJoCo卡-8万 → 半车正常 |
| td3 | -11799 | -452~-2536 | |
| ddpg | -9982 | -97~-1939(最低-97) | |
| ppo | -28977 | -362~-1731 | MuJoCo撞死-80000完全没学 → 半车救活 |

## 评估：车身加速度RMS减振率 vs passive（7场景平均）
| 控制器 | 平均减振率 |
|---|---|
| SPDF | +97.0% |
| PID | +89.0% |
| SAC | +65.7%（RL最佳）|
| PPO | +37.7%（救活）|
| TD3 | +25.7% |
| MPC | +20.8% |
| DDPG | +20.8% |

## 结论
- 切到半车这条路彻底走通：所有控制器正常减振（MuJoCo上skyhook/MPC都减不了振）
- PPO 从完全学不动 → 减振37.7%，证明 BC初始化+正确力学 的价值
- SPDF/PID 经典强 baseline（97/89%），符合 claim boundary（RL不普遍优于经典）

## 备份
half_car_env.py.bak_rnorm_* / default.yaml.bak_rnorm_* / road_profiles.py.bak_roadfix_*

## 待办（下一步）
- SI-RPPO / BP-NMPS 切到半车（脚本可能写死 MuJoCo，需改）
- 论文数据更新
