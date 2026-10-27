# suspension_control_all

纯 Python 半车主动悬架控制研究项目，包含 PID、SPDF、预瞄 MPC，以及 DDPG、TD3、SAC、PPO 和 SI-RPPO 训练、评估流程。

## Conda 环境

从仓库根目录创建并激活独立环境：

```powershell
conda env create -f environment.yml
conda activate suspension-control
```

`environment.yml` 使用 Python 3.12，在 Conda 环境内安装 `requirements-dev.txt`。
运行依赖统一维护在 `requirements.txt`，开发工具维护在 `requirements-dev.txt`；
无需 `pip install -e .`。源码保留 Python 3.10 语法兼容性。

依赖文件更新后同步环境：

```powershell
conda env update --name suspension-control --file environment.yml
```

自动化任务可使用 `conda run --name suspension-control python scripts/check_project.py`。
PyTorch 根据当前安装和硬件自动选择 CUDA 或 CPU；训练支持 `--device cpu`、
`--device cuda` 和默认的 `--device auto`。PyChrono 与 MATLAB/CarSim 不包含在此环境中。

## 项目结构

| 路径 | 职责 |
| --- | --- |
| `config.py`、`configs/` | YAML 继承、算法覆盖、实验配置 |
| `models/`、`envs/` | 半车动力学、Gymnasium 接口；另有独立四分之一车模型 |
| `roads/` | ISO、凸包、正弦、阶跃、坑洼、组合及表格路面 |
| `controllers/` | 经典控制器、RL 适配器、共享动作处理与安全层 |
| `rl/` | 四种 RL 算法、Transformer 网络、经验回放 |
| `experiments/` | 公共评估、轨迹记录与指标计算 |
| `scripts/`、`visualization/` | 训练、校验、数据生成、绘图与动画 |
| `carsim_mpc_quick/` | 独立 MATLAB/Simulink/CarSim 子项目，见其 README |
| `chrono/` | 独立 Chrono 整车悬架工程与配置，见其 README |
| `docs/`、`projects/` | 研究记录与答辩材料 |
| `results/`、`datasets/` | 本地结果、权重与数据，不提交 Git |

各类文件的用途和保留规则见 [文件导航](docs/file_inventory.md)；
代码分层、当前启动入口和 CARLA/Orin 公共模块见 [代码结构](docs/code_structure.md)。
答辩源稿与最终成果见 [研究材料导航](projects/README.md)，文档辅助脚本见
[工具说明](tools/README.md)。历史上已跟踪的少量 CarSim 结果 CSV 仍保留在 Git 中。

## 验证与开发

所有命令在激活环境后，从仓库根目录运行：

```powershell
python -m ruff check .
python -m ruff format --check .
python scripts/validate_training_config.py --config configs/default.yaml --out results/validate.json
python scripts/check_gym_registration.py --config configs/default.yaml --smoke
python scripts/check_project.py
```

`check_project.py` 检查配置、路面数值一致性、Gymnasium、MPC 时间对齐、
动作处理一致性、四种算法短训练和 checkpoint 往返。
报告写入 `results/project_checks/<时间戳>/`。
本仓库没有 pytest 套件，也不设置覆盖率门槛。

代码使用 4 空格、双引号、100 字符行宽，遵循 Ruff E/F/I/B/UP 规则；
`python -m ruff format .` 用于格式化。临时目录和生成工件已从 Ruff 扫描中排除。

## 训练与评估

单回合开发检查：

```powershell
python scripts/train_rl.py --algorithm sac --config configs/smoke.yaml --episodes 1 --device cpu --out results/smoke/sac_001
```

`--out` 必须指向新目录。常规实验使用：

```powershell
python scripts/train_rl.py --algorithm sac --config configs/train_fast.yaml
python scripts/evaluate_all.py --config configs/default.yaml
python scripts/evaluate_all.py --config configs/default.yaml --checkpoints td3=latest,sac=latest
```

默认评估经典控制器；配置明确要求 RL 控制器时，必须提供对应权重。
`--checkpoints` 接受逗号分隔的 `算法=路径`，算法为 `ddpg/td3/sac/ppo`。
`latest` 按修改时间查找 `results/*_<算法>/checkpoints/final.pt`；
评估配置必须与训练时的观测维度、动作处理和车辆参数相符。
训练保存配置、清单、历史和权重。

SI-RPPO 的 BC 数据不随仓库分发，先生成再训练：

```powershell
python scripts/collect_expert_dataset.py --config configs/half_car_safe_ppo.yaml --expert SPDF --out datasets/expert/half_car_spdf.npz --episodes 7
python scripts/train_rl.py --algorithm ppo --config configs/half_car_safe_ppo.yaml
```

短训练仅验证流程；控制性能和收敛结论需要完整实验。

## 数据与子项目边界

```powershell
python scripts/generate_road_dataset.py --out datasets/half_car_roads --duration 8
python scripts/plot_results.py --result-dir results/<实际评估目录>
python scripts/run_chrono_active_suspension.py --config configs/chrono_quarter_car.yaml --backend rk4
```

CSV 路面使用 `time,height` 两列，可带表头；相对路径基于仓库工作目录。
`inherits:` 的父配置路径基于当前 YAML 所在目录。
修改 `default.yaml` 前运行 `rg "inherits:" configs` 检查影响范围，优先添加独立配置。

当前半车环境提供 privileged 观测及可选上一时刻动作。历史观测、估计/噪声观测、
`preview_error` 和神经 MPC 参数调度未实现，不能据旧脚本或配置名认定已支持。
历史 MuJoCo 文档、论文绘图脚本可能引用已删除配置或本地实验路径。
已移除两个依赖旧接口的预瞄误差实验入口，保留历史结果及可用的数据分析脚本。

贡献约定见 [AGENTS.md](AGENTS.md)，整理及验证记录见
[项目整理记录](docs/project_maintenance.md)。
