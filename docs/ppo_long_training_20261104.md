# PPO 延长训练计划与运行记录

采用 `configs/cosim_ppo_literature.yaml`，追加 300 回合 × 10 s × 100 Hz，合计 300,000 步。
从 `results/cosim_ppo_literature_smoke_20261104_v2/final.pt` 的 3,000 步权重开始；
该旧文件没有优化器状态，因此明确属于权重热启动，不是完整优化器续训。
本次保持奖励、网络和执行约束不变，集中检查增加训练量的影响。

训练路面按 wave/bump/pothole 轮换。组合路面用作固定验证集：训练前记录被动对照、
初始策略，再在首回合、每 25 回合和最后一回合运行确定性评估。组合路面已参与
之前的诊断，所以这里不能将它称为未见测试集。泛化需要后续新路面/速度/随机种子评估。
训练不请求 Qwen，保留车辆状态与 Orin 深度预瞄。

`best.pt` 按验证垂向加速度 RMS 选择，同时要求本次采样无 80 mm 行程越限、最小
轮胎载荷为正。这只是候选模型选择条件，不是稳定性证明；其他性能指标一起保留。
`initial.pt`、逐回合 checkpoint、`final.pt` 与原始日志均保留，禁止把随机探索回报
直接等同于确定性控制性能。初始策略也可以保持为最佳候选。

运行目录：`results/cosim_ppo_long_20261104_v1`。
启动日志和进程身份分别在该目录旁的 `_launcher.log`、`_launcher.json` 文件。
查看进度用 `status.json`、`evaluation_history.json`、`best_evaluation.json` 和实际进程。
完成全部回合后自动进行原始仿真/动作概率/模型更新审计，审计通过才标记 completed。
任何训练或评估失败会保存证据并退出，不自动重试。

```powershell
conda run --no-capture-output -n suspension-control python scripts/train_cosim_ppo.py --config configs/cosim_ppo_literature.yaml --carla-python E:/anaconda3/envs/carla0911/python.exe --episodes 300 --seconds 10 --road-cycle wave bump pothole --checkpoint results/cosim_ppo_literature_smoke_20261104_v2/final.pt --eval-every 25 --eval-road combined --output results/cosim_ppo_long_20261104_v1
```

预计约 2–3 小时，实际以测得的回合耗时为准。已有相同长度回合约产生 14 MB 原始工件，
本轮开始前 E 盘约 95 GB 可用。训练使用本机 CPU、Windows CARLA/CarSim 和现有 Orin
预瞄服务；没有启动 4090 服务器任务。

新 checkpoint 保存 CPU PyTorch/NumPy 随机数及两个 Adam 优化器状态。恢复时显式增加
`--resume-training`，指定含这些状态的 checkpoint，并使用新的输出目录。
回合编号/路面轮换会从新目录重新开始，需按剩余路面顺序指定 `--road-cycle`；该功能
不承诺恢复正在运行的半个 CarSim 回合，也不自动合并旧目录的评估历史。

验证：全库 Ruff、`scripts/check_project.py` 七组通过；
`scripts/check_ppo_resume.py` 验证恢复后的下一批观测、动作及 actor/critic 更新与不中断
运行逐元素一致，报告 `results/ppo_resume_checks/20261104_094841/report.json`。
