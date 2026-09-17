# 项目整理与验证记录

日期：2026-09-13。

## 项目边界

主程序是 Python 半车控制栈：路面 → 半车环境/动力学 → 经典控制器或 RL 策略 →
共享动作处理 → 轨迹与指标。`config.py` 负责父配置继承与算法参数覆盖。
`carsim_mpc_quick/` 是独立 MATLAB/Simulink/CarSim 项目；四分之一车 RK4/PyChrono
程序使用独立配置。论文材料和本地临时工件不属于主程序源码。

## 环境与贡献指南

- 统一使用中文根目录 `AGENTS.md`（仓库协作指南）；README 提供中文运行说明。
- 新增 `environment.yml`，默认环境名 `suspension-control`、Python 3.12。
  Conda 管理解释器，环境内的 pip 按 `requirements-dev.txt` 安装项目依赖。
- 运行依赖与 Ruff 开发依赖分开；移除主项目不再使用的 MuJoCo 和 pytest 依赖。
- 本机已创建 `E:\anaconda3\envs\suspension-control`，克隆自既有 `rl_env` 后补齐依赖。
  原环境未修改。此次验证使用 Python 3.12.13、PyTorch 2.10.0、NumPy 2.4.3、
  SciPy 1.18.1、Gymnasium 1.2.3、Ruff 0.16.7。
- 本环境报告 CUDA 不可用，本次训练验证在 CPU 上完成。环境定义没有固定 CUDA 构建。

## 已修复的问题

| 范围 | 修复及意义 |
| --- | --- |
| 配置 | 检查继承循环、YAML 根类型、正数与整数约束、采样权重；深合并不共享可变子对象；CSV 校验使用与运行时相同的相对路径 |
| 观测 | 环境与四种算法共用维度检查；明确拒绝尚未实现的历史、估计/噪声观测和预瞄误差模式，避免产生误导性实验 |
| 环境 | 重置重新计算安全状态，使用 Gymnasium 局部随机状态；非法动作提前报错；动力学限力与配置保持一致 |
| 路面 | 批量计算预瞄高度，移除无用导数计算；每个环境状态共用一次预瞄结果；拒绝未知路面和非有限表格数据 |
| 训练/评估 | 共用策略动作参数化、门控和安全投影；保留原始策略动作与其对数概率；专家转移使用相同控制周期；评估应用算法覆盖且要求 RL 权重 |
| RL 更新 | 时间上限截断保留 bootstrap，真正终止仍截断目标；TD3 的 BC 不再覆盖动作正则；单步 PPO 轨迹避免方差 NaN |
| MPC | 按预测时刻插值预瞄；支持短预瞄末值延拓和 horizon=1；修复初值重复移位，增量惩罚参考实际上一动作 |
| 频谱 | 从轨迹时间推导采样频率，修复论文图脚本中写死的 1000 Hz；单边 PSD 正确计入正负频率能量 |
| 开发流程 | 新增 `configs/smoke.yaml`、`scripts/check_project.py`；训练支持设备与输出目录参数；记录运行库版本；更新失效命令并统一 Ruff 格式 |

`configs/default.yaml` 的研究基线数值未修改。但 bootstrap、MPC、专家采样周期与
频谱计算的修复会影响新实验或新生成图表。既有权重仍可加载；比较研究指标时应记录
代码版本、配置和环境，并重新运行相应实验。没有据短检查更新任何论文性能结论。

## 验证结果

以下检查均已通过，未运行 pytest：

```powershell
conda run --name suspension-control python -m pip check
conda run --name suspension-control python -m ruff check .
conda run --name suspension-control python -m ruff format --check .
conda run --name suspension-control python scripts/check_project.py
```

- 10 个半车配置校验、两种观测维度模式、Gymnasium 官方环境检查。
- 7 种场景的路面标量/批量计算对照；CSV 数据生成后重新加载、校验。
- PID、SPDF、MPC、PASSIVE 短回合；三种预瞄采样间隔和单步 MPC 时域。
- 残差与绝对动作路径的一致性，PPO bootstrap 和单步更新，TD3 BC 与动作正则共存。
- DDPG、TD3、SAC、PPO 各一次短训练，四次权重保存/加载，8 行综合评估结果。
- 正弦信号的频率峰值与 PSD 积分校验。
- 独立 CLI 的 SAC 训练、评估与绘图，SPDF 数据采集、PPO BC/anchor，以及四分之一车 RK4 运行。

本次工件位于 `results/project_audit_20260913_104741/`；主要检查报告为
`conda_checks/20260913_122706_576376/report.json`，命令行绘图位于 `cli_evaluation/figures/`。
7 个场景各 50 步的固定动作对照中，状态、奖励及观测相对修改前的最大绝对差均为 0。
同一 Conda 环境内，450 点 ISO 路面预瞄的微基准中位耗时由约 2.18 ms 降至 0.020 ms；
该结果仅衡量预瞄函数，不代表端到端训练加速或控制性能提升。

## 尚未覆盖的范围

长时间训练收敛、GPU 运行、实际 PyChrono 后端和 MATLAB/CarSim 联合仿真未验证。
神经 MPC 参数调度及预瞄误差注入仍需独立实现；相关历史脚本不能作为已支持的依据。
论文图脚本保留本地历史结果路径，需要实际研究数据才能完整重建。
Gymnasium 对以牛顿为单位的动作范围和无界观测给出的提示仍保留。

## 文件清理（2026-09-13）

经授权检查临时目录和仓库内 Conda 残留后，删除 1,143 个文件（21,067,612 字节，
约 20.1 MiB）：旧 `.venv`、未完成的本地 Conda 安装缓存、Python/Ruff 缓存、
文档编译中间文件、PPT 渲染与提取结果、16 份逐文件 SHA256 相同的 SVG 备份、
空日志，以及两个失效实验入口。

- 移除 `scripts/run_bp_nmps_preview_robustness.py`：依赖已删除的
  `scripts.tune_mpc_parameters` 和当前工厂不支持的控制器。
- 移除 `scripts/run_si_rppo_preview_robustness.py`：预瞄误差注入和旧全车基线
  不适用于当前半车环境。保留其历史实验结果，以及仍可处理旧数据的
  `scripts/summarize_mpc_parameter_schedule.py`。
- 将临时目录中的报告生成源码迁至 `tools/build_carsim_mpc_report.py`，修正仓库路径，
  生成工件改写入 `results/carsim_report/`；两张独有 SVG 迁至
  `projects/active_suspension_diagrams/`。
- 保留两套答辩项目的源稿、素材、设计说明、分析记录、最终 SVG 和 PPTX。
  两个项目共用的 9 对 PNG 虽内容相同，仍保留各自引用，避免破坏独立编辑。
- 保留现有实验结果、数据集、模型权重、论文及 CarSim/Simulink 工程；
  独立 Conda 环境仍位于 `E:\anaconda3\envs\suspension-control`。

完整盘点与逐文件删除清单位于 `results/file_cleanup_20260913_134057/`：
`inventory.json`、`protected_inventory.json`、`deletion_plan.json`、`summary.json`。
`source_recovery.zip` 保留两个移除入口和迁移前报告脚本的原始字节；
已跟踪的 PPT 中间文件可从 Git 历史恢复。目录说明见 [文件导航](file_inventory.md)。

清理后执行并通过 Ruff 静态检查、60 个文件的格式检查、`git diff --check`，以及
`scripts/check_project.py` 的 7 组数值/短训练检查。新回归报告位于
`results/file_cleanup_20260913_134057/checks/20260913_135813_362505/report.json`。
另核对了 5,239 个保留文件的大小与修改时间、迁移 SVG 的哈希、恢复压缩包内容、
文档链接和 SVG 本地素材引用，未发现非计划缺失或改动；详见同目录 `verification.json`。
报告辅助脚本通过语法与静态检查，本次未重新生成 Word/PPT 或运行 CarSim。
