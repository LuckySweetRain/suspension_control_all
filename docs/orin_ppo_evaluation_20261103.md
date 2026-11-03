# CARLA + CarSim PPO 训练与 Orin 评估（2026-11-03）

已完成一次真实联合仿真训练、最终权重导出、Orin 部署与闭环推理评估。
这是初始研究基线，未证明收敛、泛化或硬实时能力。Qwen 使用已有驾驶轨迹模型，
没有训练新的悬架建议头；保留的舒适性建议观测通道为零。

## 训练与部署

- 12 回合 × 1,000 步；控制采样间隔 10 ms，波浪/减速带/坑洼轮换，组合路面留作评估。
- 最终权重：`results/cosim_ppo_qwen_train_20261103_v1/final.pt`。
- 全部回合审计通过：`results/cosim_ppo_qwen_train_20261103_v1/audit.json`。
- 导出包：`results/cosim_ppo_acceptance_20261103_v1/policy_bundle`。
- Orin 目录：`/home/nvidia/suspension_control/ppo_candidates/cosim_ppo_acceptance_20261103_v1`。
- 已运行用户服务：`suspension-ppo-cosim-ppo-acceptance-20261103-v1.service`，端口 `55112`。
- 模型 SHA256：`24af3925edee81a68999cc3b1dffbe7d618e9204a57b9aaca8e2b91ba298875d`。
- 远端文件与监听服务身份均已核验，记录在验收目录的 `deployment_audit.json`。

输入为 104 维归一化观测，输出为前/后轴两路原始主动力（N）。限幅、变化率限制、
被动阻尼叠加和四角映射由控制桥处理；该网络服务本身不直接操作执行器。
详细观测与控制边界见 [训练说明](rl_cosim_training.md) 和 `scripts/train_cosim_ppo.py`。

## 本次对照结果

每项为同一组合路面 10 s、1,000 步、约 96.81 m、35 km/h；指标包含初始化阶段。

| 模式 | 垂向加速度 RMS（m/s²） | 相比被动变化 |
|---|---:|---:|
| 被动 | 1.609573 | — |
| 本机 PPO + 同步 Qwen | 1.598125 | -0.7113% |
| 本机 PPO，无 Qwen 条件 | 1.599185 | -0.6454% |
| Orin PPO + 同步 Qwen | 1.598123 | -0.7114% |
| Orin PPO + 异步 Qwen | 1.599112 | 约 -0.6500% |

所有策略评估的记录动作均与原始 CarSim trace 一致；两个 Orin 回合分别有 1,000 次
真实端侧推理，无本机替代。对导出模型重放全部观测，最大动作误差小于 0.000107 N。
单条短路面的微小差异不能证明 Qwen 的稳定增益，更不能证明已达到最终控制性能。

## 实时性结论

- 最终模型独立回放：CPU 单线程，中位 1.554 ms、P95 1.723 ms、最大 8.499 ms；
  1,000 条观测均低于 10 ms，但不包含通信、感知、控制映射和仿真。
- 同步闭环 PPO 请求（含通信）：P95 12.496 ms，8.6% 超过 10 ms。
- 异步闭环 PPO 请求：P95 14.076 ms，16.1% 超过 10 ms。
- 异步完整循环：中位 21.921 ms、P95 76.695 ms、最大 128.673 ms，全部超过 10 ms。
- 10 s 仿真的总墙钟时间：同步约 193.5 s，异步约 35.1 s。
- 异步 Qwen 完成 7 次推理，控制循环持续推进；结果到达时的仿真年龄中位为 1.11 s，
  实际墙钟等待中位为 3.77 s。全部超过当前 1 s 条件有效期，有效条件步数为零。

因此，异步传输已验证可用，但当前整条链路不满足稳定 100 Hz。不能直接把此策略
视为实车可用控制器，也不能通过放宽时效阈值来宣称解决了 Qwen 延迟。后续需要针对
延迟/缺失条件训练与评估，缩短感知和传输路径，并在目标实时运行时验证完整截止时间。

## 复现与使用

Orin 上现有服务正在运行，不要重复占用 `55112`。查看状态：

```bash
systemctl --user status suspension-ppo-cosim-ppo-acceptance-20261103-v1.service
```

若服务已停止，在 Orin 使用现有 Python 环境启动（前台命令）：

```bash
cd /home/nvidia/suspension_control/ppo_candidates/cosim_ppo_acceptance_20261103_v1
/mnt/nvme/qwen-drive/environments/qwen-drive-py312/bin/python serve_cosim_policy.py . --port 55112
```

该服务是本次创建的临时 systemd 用户服务，不承诺重启后自动恢复；模型包也不是完整
Docker 镜像。当前仍依赖已有 Orin Python 环境、独立 Qwen 服务及 Windows 上的
CARLA 0.9.11 / CarSim DLL 和外部数据库。软件环境限制见训练说明。

本机仓库根目录复现异步评估（输出目录须不存在）：

```powershell
conda run --no-capture-output -n suspension-control python scripts/train_cosim_ppo.py --carla-python E:/anaconda3/envs/carla0911/python.exe --episodes 1 --seconds 10 --road-profile combined --mode evaluate --checkpoint results/cosim_ppo_qwen_train_20261103_v1/final.pt --planner qwen --planner-port 55013 --qwen-context-only --async-qwen --inference-host 192.168.50.2 --inference-port 55112 --policy-bundle results/cosim_ppo_acceptance_20261103_v1/policy_bundle --output results/cosim_ppo_orin_async_new
```

需保持深度预瞄服务和 Qwen 服务可用；不要与其他车辆仿真同时运行。
同步验收证据在 `results/cosim_ppo_acceptance_20261103_v1/`，异步证据在
`results/cosim_ppo_orin_async_20261103_v1/`；各策略目录含 `evaluation_audit.json`。
权重、数据与原始日志属于本地工件，不纳入 Git。

验证命令：`python -m ruff check .`、`python scripts/check_async_planner.py`
（Python 3.12 与 CARLA 3.7 均通过）、`python scripts/check_project.py`（七项通过，
报告 `results/project_checks/20261103_205853_283821/report.json`）。没有运行 pytest。
