# W&B 与精简训练记录

W&B run：
https://forge.coreweave.com/wandb/yu2372432120-harbin-institute-of-technology/suspension-control/runs/zk0y9eji

项目为 `suspension-control`。本次由独立 `scripts/watch_cosim_wandb.py` 进程同步，
网络同步不占用仿真/学习线程。只提交汇总指标及 PPO/奖励超参数，关闭自动代码/Git收集，
不上传相机图像、原始轨迹、模型文件或凭据。

## 本次续训

旧目录 `results/cosim_ppo_long_20261104_v1` 完成 236 个回合后，在第 237 回合的
CARLA 初始图像等待阶段发生超时。失败记录保留。已从 `episode_0235/checkpoint.pt`
恢复优化器和随机数状态，在 `results/cosim_ppo_long_20261104_v2` 继续剩余 64 回合。
下一路面从 pothole 开始，保持原轮换顺序。W&B 的 `training_steps` 使用累计偏移
236,000，旧 236 回合和已有验证历史补传到同一 run。

## 保留策略

训练启动参数：`--artifact-policy compact --keep-every 25`。

- 每回合原子替换 `latest.pt`，供故障恢复；不再累积每回合权重副本。
- 保留首回合、每 25 回合、末回合的 checkpoint 和完整仿真记录。
- 保留 `initial.pt`、`best.pt`、`final.pt`，以及所有验证和失败工件。
- 普通成功回合写入单一 `history.jsonl`，含 reward、loss、KL、clip fraction、
  explained variance、动作力、投影率和奖励分项均值。
- 清理前逐步核对行为策略 log probability、CarSim 原始状态和实际施加力，保留
  校验摘要与原始数据哈希；仅删除带有本次回合所有权标记且路径严格位于本项目
  `results` 下的文件夹。旧目录不做批量删除。
- 精简模式最终审计验证保存的在线校验记录；删除后的原始轨迹不能再做离线重放审计，
  这一证据范围与完整保留模式不同。研究关键评估仍保留全量数据。

不指定该参数时仍为 `full`，以兼容原有复现实验。实际续训已显式启用 `compact`。

## 看板指标与恢复同步

横轴采用 `training_steps`。查看 `train/return`、`loss/actor`、`loss/value`、`ppo/*`、
`train/projection_fraction`、`train/reward/*` 与 `eval/vertical_accel_rms_mps2`。
训练曲线包含探索噪声，应结合确定性验证曲线判断改善。历史旧训练只有当时已记录的
summary 指标，不伪造缺失的投影率和奖励分项；这些指标从新精简记录开始提供。

同步进程的 run ID、URL、已入队事件和心跳在
`results/wandb_ppo_long_20261104/sync_state.json`。SDK 会异步上传；本地入队不等于
服务器确认，必要时通过 W&B API/网页核对。进程退出后可用同一输出目录重启镜像脚本，
从已记录事件继续同步；服务长时间断网时保留 SDK 本地日志以便恢复。

```powershell
conda run -n suspension-control python -m pip install -r requirements-tracking.txt
conda run --no-capture-output -n suspension-control python scripts/watch_cosim_wandb.py --run-dir results/cosim_ppo_long_20261104_v1 --step-offset 0 --run-dir results/cosim_ppo_long_20261104_v2 --step-offset 236000 --output results/wandb_ppo_long_20261104 --entity yu2372432120-harbin-institute-of-technology
```

脚本指定 `https://api.wandb.ai`，SDK 从现有用户级凭据读取认证。此机实测 API 需经过
现有本地代理，直连返回 403；脚本继承已有代理配置。密钥不放在命令行、仓库或文档。
先核对记录的 PID 是否仍在运行，避免重复启动同步或仿真进程。

验证：全库 Ruff、`scripts/check_project.py` 七组通过；
`results/cosim_compact_smoke_20261104_v1` 三个实际短回合验证首末保留、普通回合清理、
latest/final 保存、完整历史与在线物理审计。短回合只验证存储工作流，不用于控制性能结论。
