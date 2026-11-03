# 项目文件导航

## 代码与配置

| 位置 | 用途 |
| --- | --- |
| `config.py`、`configs/` | 配置读取、YAML 继承和各算法/场景参数；主入口为 `default.yaml`，快速校验用 `smoke.yaml` |
| `models/` | 半车动力学和车辆参数 |
| `envs/` | 半车 Gymnasium 环境与注册 |
| `roads/` | 路面生成、CSV 导入与预瞄采样 |
| `controllers/` | PID、SPDF、MPC、RL 策略适配和动作安全处理 |
| `rl/` | DDPG、TD3、SAC、PPO、网络与经验回放 |
| `experiments/` | 控制器构建、公共 rollout、指标与综合评估 |
| `visualization/` | 轨迹图、频谱和动画 |
| `environment.yml`、`requirements*.txt`、`pyproject.toml` | Conda 环境、依赖及 Ruff 规则 |
| `.agents/`、`.qoder/`、`AGENTS.md` | 本地协作工具设置与贡献约定 |

`configs/half_car_bp_nmps.yaml` 当前可用于普通 MPC 配置，文件名不代表已实现神经参数调度。
配置相互继承，不要只因某份配置未被默认命令使用就删除它。

## scripts/ 命令入口

从仓库根目录、在 `suspension-control` 环境中运行 Python 入口。

| 文件 | 用途 |
| --- | --- |
| `train_rl.py`、`evaluate_all.py` | 常规训练及多控制器评估 |
| `check_project.py`、`validate_training_config.py`、`check_gym_registration.py` | 数值/训练回归、配置校验、Gymnasium 冒烟检查 |
| `collect_expert_dataset.py`、`collect_env_statistics.py` | 专家转移数据与观测统计 |
| `generate_road_dataset.py`、`import_road_directory.py` | 生成路面或导入外部 CSV 目录 |
| `plot_results.py`、`animate_result.py`、`plot_preview_poly_trajectories.py` | 结果绘图、动画和预瞄拟合展示 |
| `evaluate_ch6_unified.py`、`rerun_report_experiments.py` | 论文特定配置的统一对比与复现实验 |
| `generate_ieee_figures.py`、`generate_eps_figures.py`、`ijrnc_figure_style.py` | 历史论文绘图入口及公共样式 |
| `generate_ch6_matlab_figures.m` | MATLAB 论文绘图，依赖源码中配置的外部报告目录 |
| `summarize_mpc_parameter_schedule.py` | 汇总已有 BP-NMPS 轨迹中的参数，不提供对应训练或控制器 |

论文脚本中的日期目录、权重和外部报告路径需要与实际数据核对。两个依赖旧全车接口及
预瞄误差模式的运行脚本已移除，原因和恢复位置见 [整理记录](project_maintenance.md)。

## 研究材料与本地工件

| 位置 | 内容与处理方式 |
| --- | --- |
| `docs/active_suspension_research_plan.md` | 研究计划 |
| `docs/model_parameter_audit.md`、`docs/alma_analysis/` | 历史模型与实验分析；其中旧路径仅作追溯依据 |
| `docs/carsim_qp_mpc_latex/` | CarSim 报告 TeX、插图及最终 PDF；辅助编译文件可再生成 |
| `docs/CarSim_QP-MPC联合仿真技术报告.docx` | 保留的 Word 技术报告 |
| `docs/projects/` | 答辩项目、源稿、独立图稿和导出成果，见 [材料导航](projects/README.md) |
| `tools/` | 报告与演示文稿辅助工具，见 [工具说明](../tools/README.md) |
| `carsim_mpc_quick/` | 独立 MATLAB/Simulink/CarSim 工程，包含源码、模型及验证脚本；依赖外部 CarSim 数据库 |
| `results/` | 实验配置、日志、轨迹、指标、权重和检查报告；常用成果及归档入口见 [结果导航](../results/README.md)，已有引用的路径保留 |
| `results/archive/` | 无已知仓库引用的历史训练与旧评估；训练按算法分类，原路径映射见 [完整索引](../results/results_cleanup_20260922/目录索引.md) |
| `datasets/` | 本地路面及专家数据；可能由配置引用 |

缓存、临时预览、PPT 提取结果和已确认重复的备份可以清理。模型权重、实验轨迹、
文稿源文件及导出成果有独立用途，不能仅根据文件大小、日期或扩展名判断是否无关。
内容相同的素材也可能被不同项目分别引用，删除前同时检查哈希与路径引用。

环境统一使用仓库外的 Conda 环境。新实验放在 `results/<实验名或时间戳>/`，新数据放在
`datasets/`；论文定稿放在 `docs/`，答辩定稿放在对应项目的 `exports/`。

2026-09-22 的结果整理保留全部独有实验与指标，清除了内容重复的阶段权重和可再生成的
PPT 预览，并将缓存、临时探针、运行日志与中间评估统一收纳。四类算法的 `latest` 选择保持不变；
归档权重使用索引中的新路径显式加载。
性能筛选口径和逐项操作见 [清理报告](../results/results_cleanup_20260922/清理报告.md)。
