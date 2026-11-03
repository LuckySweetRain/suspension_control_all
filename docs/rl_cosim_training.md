# Qwen 条件 PPO 联合仿真训练进度

## 目标与完成标准

目标是低频 Qwen 条件与高频 PPO 主动悬架控制，在真实 CARLA 路面和 CarSim
动力学中训练，并将策略部署到 Orin 做推理与闭环评估。半车环境冒烟、短回合更新、
模型文件存在都不能替代最终验收。

尚需完成：多路面训练与独立测试、Qwen 条件消融、延迟感知的多速率异步闭环、
Orin 策略部署、端侧数值一致性与推理延迟测量、真实联合仿真对照评估。
不要求在没有证据时宣称 PPO 优于 MPC；应保存真实的负面结果。

## 已实现的训练接口

`scripts/train_cosim_ppo.py` 在 suspension-control 环境使用项目原 PPO 更新器。
CARLA 0.9.11 客户端在独立 Python 3.7 进程运行，localhost JSON 行协议传递
每个动作前的观测与动作后的 CarSim 状态。CarSim 以 1 ms 积分，每 10 ms 更新控制，
CARLA 相机每 50 ms 更新。当前是离线同步仿真，不是硬实时控制。

原 Orin 服务仍负责深度预瞄与驾驶控制，MPC 力仅保留为参考；连接 PPO 时实际悬架力
由采样的 PPO 两轴主动增量力生成。保持原被动阻尼和轴力到四角的映射。
覆盖不足或碰撞监督会抑制主动增量。控制变化量限制为每步 150 N，轴力上限 2000 N，
角点总力限制为 5000 N。原始采样动作/logp 与投影后的命令分别保存。

观测 v1 为 24 个基础量及 20×4 个路面预瞄量，共 104 维。
基础量包括垂向/俯仰运动、相对初始平衡的悬架行程、行程速度、轮胎载荷、上次命令、
车速和 Qwen 条件。Qwen 条件目前只使用真实预测轨迹推导的速度、方向、信息年龄及
有效位；预留的舒适性建议通道为零，不冒充模型已经学会悬架建议。

奖励是 10 ms 端点采样的加速度、俯仰、行程、轮胎载荷、力及变化量代价，尚未聚合
全部 1 ms 子步。未来最终评估需检查子步峰值。训练使用仿真状态，不能据此宣称已解决
真实传感器状态估计问题。有限时域截断使用末状态 critic 引导；断连、崩溃或无效状态
不产生伪造转移，不进行该回合的参数更新。

## 命令示例

从仓库根目录执行，输出目录必须尚不存在，避免覆盖失败证据：

```powershell
conda run --no-capture-output -n suspension-control python scripts/train_cosim_ppo.py --carla-python E:/anaconda3/envs/carla0911/python.exe --episodes 1 --seconds 3 --road-profile wave --output results/cosim_ppo_new_run
```

加入 `--planner qwen --planner-port 55013` 使用此次独立启动的 Qwen 端点。
该端点运行在 Orin 的临时用户服务 `suspension-qwen-rl.service`；原 TransFuser
55003 服务未替换。临时服务不等同于可复现的端侧交付包。

使用 `--mode passive` 获取被动对照；使用 `--mode evaluate --checkpoint <明确路径>`
进行确定性策略评估，不更新参数。`--checkpoint` 用于训练时只加载网络权重，
当前不恢复优化器、随机数状态，不能称为无缝断点续训。

## 2026-11-03 已取得的证据

- CARLA 0.9.11 初始空闲、CarSim DLL/参数模板/数据库存在；Orin SSH 与 CUDA 可用。
- `results/cosim_ppo_bridge_20261103_v1`：首轮因跨 Conda 环境 DLL 搜索冲突失败，
  无参数更新，证据保留。已给客户端子进程设置自身 DLL 路径优先级。
- `results/cosim_ppo_bridge_20261103_v2`：300 个真实转移、一次 PPO 更新，保存 final.pt。
- 5 s 波浪路面被动对照 `results/cosim_ppo_passive_20261103_v1`：500 步，
  垂向加速度 RMS 2.03904 m/s²。
- 同路面确定性评估 `results/cosim_ppo_eval_20261103_v1`：500 步，
  垂向加速度 RMS 2.04382 m/s²，尚无改善。以上均是单次、含初始化阶段的短回合指标。
- `results/cosim_ppo_qwen_20261103_v1`：真实 Qwen 条件集成试验，结果需检查该目录
  summary/failure 及转移记录。已完成 300 步和 7 次真实推理，但模型驾驶使车辆停下，
  仅行驶 6.5 m，不能将较低振动归因于悬架策略。首轮 Qwen 推理约 76.2 s，
  后续约 3.75 s，进一步证明必须进行多速率延迟处理。
- `results/cosim_ppo_qwen_context_20261103_v1`：使用 `--qwen-context-only`，
  保持 CarSim 原驾驶器，Qwen 只为策略提供条件。完成 500 步、17 次真实 Qwen 推理，
  行驶约 48.2 m，末速约 35 km/h。该回合随机探索，不是确定性性能评估。
- `results/cosim_ppo_qwen_train_20261103_v1`：已启动 12 回合，每回合 10 s，
  波浪/减速带/坑洼轮换，组合路面留待评估。完成状态以逐回合 summary/failure、
  进程状态和 final.pt 为准，不依据本记录推断训练已经完成。
- `results/cosim_ppo_orin_probe_20261103_v1`：初期策略 TorchScript 导出，本机 500 条
  真实观测误差为零。部署到 Orin 的
  `/home/nvidia/suspension_control/ppo_candidates/probe_20261103_v1`，CPU 单线程回放
  500 条观测最大力误差 0.000106812 N；预热后推理中位数 1.542 ms、P95 6.195 ms、
  最大 8.224 ms，没有超过 10 ms 的样本。测试时 Qwen 训练服务并行运行。
  这是初期策略的推理检查，不是最终权重的车辆闭环验收，也不是最坏时延保证。
- Ruff 全库检查通过；`scripts/check_project.py` 七项通过，报告
  `results/project_checks/20261103_200952_193697/report.json`。未运行 pytest。

上述权重仅是链路验证产物，尚不是已收敛或经过端侧验收的最终策略。
无关的 `.qoder` 删除保持原状；本轮未使用 4090，也未停止其他用户任务。

## 导出和核验

`scripts/summarize_cosim_ppo.py <run目录>` 生成实际转移指标，显式区分完成/失败回合。
`scripts/export_cosim_policy.py --checkpoint <权重> --transitions <真实转移JSONL> --output <新目录>`
导出 104 维观测到两轴原始主动增量力的确定性 TorchScript，附 SHA256 和回放样本。
`scripts/benchmark_cosim_policy.py <导出目录>` 检查 batch=1 的 CPU 数值一致性与时延。
导出网络不含观测预处理、限幅、变化率限制和被动阻尼映射；最终闭环必须复用同一契约。

## 端侧闭环接口与异步准备（2026-11-03 后续）

新增 `scripts/serve_cosim_policy.py`：CPU 单线程 TorchScript 端点，校验 schema、权重
SHA256、观测维度和连续序号。单次请求只返回两轴原始力，不直接驱动真实执行器。
训练入口在 `--mode evaluate` 时可设置 `--inference-host`、`--inference-port` 和
`--policy-bundle`。网络失败不会回退本机策略；bundle 必须匹配指定 checkpoint。

Orin 初期策略端点运行在 `suspension-ppo-probe.service`、55111 端口，部署路径同前。
500 条真实观测 TCP 回放的最大力误差为 0.000106812 N；往返时间中位数 5.046 ms、
P95 7.399 ms、P99 10.541 ms、最大 13.168 ms，1.4% 超过 10 ms。
报告为 `results/cosim_ppo_orin_probe_20261103_v1/orin_rpc_benchmark.json`。
这已经包含推理和 JSON/TCP，但不包含传感器、预瞄和车辆执行，不是完整实时达标证据。

新增 `runtime/async_planner.py`，提供单请求在途、非阻塞 poll、捕获时间与坐标原点
不可变校验；不把旧结果重新标成当前结果，也不伪造模型输出。
`python scripts/check_async_planner.py` 已通过非阻塞、单请求、元数据冻结和错误时间戳
拒绝检查。此模块尚未接入正在运行的仿真批次；当前 12 回合继续使用原同步模型。

本轮 `python -m ruff check .` 通过；`python scripts/check_project.py` 七项通过，报告
`results/project_checks/20261103_202736_423888/report.json`。
下一阶段在 12 回合结束后检查逐回合完成证据，导出最终权重，再进行未见组合路面的
被动/PPO 对照、Qwen 条件消融、Orin 实际动作闭环和异步延迟评估。
