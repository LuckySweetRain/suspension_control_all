# 仓库协作指南

## 项目结构

本项目基于半车模型开展主动悬架控制研究，各目录职责如下：

- `models/`：车辆动力学模型；`envs/`：Gymnasium 环境；`roads/`：路面生成与导入。
- `controllers/`：PID、SPDF、MPC 控制器及策略适配器。
- `rl/`：DDPG、TD3、SAC、PPO 算法、神经网络和经验回放。
- `experiments/`：公共评估逻辑；`scripts/`：命令行入口。
- `configs/`：实验配置；`visualization/`：绘图代码；`docs/`：研究记录。

## 开发命令

所有命令均从仓库根目录运行，脚本会自行配置模块导入路径。

- 创建 Conda 环境：`conda env create -f environment.yml`。
- 激活环境：`conda activate suspension-control`。
- 训练：`python scripts/train_rl.py --algorithm sac --config configs/train_fast.yaml`。
- 评估：`python scripts/evaluate_all.py --config configs/default.yaml`。
- 配置校验：`python scripts/validate_training_config.py --config configs/default.yaml`。
- 环境冒烟检查：`python scripts/check_gym_registration.py --config configs/default.yaml --smoke`。
- 静态检查：`python -m ruff check .`；代码格式化：`python -m ruff format .`。

评估支持 `--checkpoints sac=latest,ppo=path/to/final.pt`；`latest` 按修改时间选择
匹配算法的最新 `final.pt` 权重文件。

## 代码风格

保持 Python 3.10 兼容性，使用 4 空格缩进、双引号和
`from __future__ import annotations`。函数与模块采用 `snake_case`，类采用
`PascalCase`。Ruff 执行 E/F/I/B/UP 规则，格式化行宽为 100 字符。
明确标注物理单位、观测维度和策略动作变换。

使用 Conda 管理环境；运行依赖维护在 `requirements.txt`，开发工具维护在
`requirements-dev.txt`，由 `environment.yml` 引入。

## 测试与验证

本仓库没有 pytest 测试套件或覆盖率门槛，不要运行 pytest，也不要声称通过了 pytest。
修改后运行 Ruff 及相关校验、冒烟检查脚本。修改控制器或训练代码时，运行
`python scripts/check_project.py`，检查观测、动作、奖励是否为有限值，以及训练与评估
是否一致。汇报实际执行的命令和验证局限；短时检查不能证明训练收敛或控制性能提升。

## 配置与数据

`config.py` 根据 YAML 的 `inherits:` 递归合并配置。修改 `default.yaml` 前，先运行
`rg "inherits:" configs` 检查影响范围，优先在独立实验配置中覆盖参数。
保留可复现的随机种子，避免无说明地改变研究基线。

路面文件的相对路径基于仓库工作目录解析。缺少模仿学习数据时，使用
`scripts/collect_expert_dataset.py` 生成；数据集和模型权重属于本地工件。
使用历史记录中的路径前，先确认文件仍然存在。

## 子项目边界与 Git 约定

`carsim_mpc_quick/` 是 Python/CarSim/CARLA/Orin 联合仿真项目，依赖外部数据库及求解器许可。MATLAB/Simulink 源码与模型已移除；历史数值资产和论文材料保留。
`docs/projects/` 保存答辩源稿、素材和最终成果；`tools/` 保存文档辅助脚本。
文件用途见 `docs/file_inventory.md`。整理时保留论文源稿、数据集、实验结果和权重；
重复文件须核对内容与引用，可再生成的预览、编译缓存与提取中间文件不纳入版本管理。
未经明确授权，不得读取、修改或删除 `tmp/`、`tmp_*`、`.conda_envs/`、`.conda_pkgs/`。
生成结果统一放在 `results/` 或 `datasets/`。

使用 `master` 分支，直接提交，不使用拉取请求（PR）。提交信息使用中文，例如
`修复配置校验路径`。保持改动便于审查，保留与当前任务无关的工作。
工作前读取适用的子目录 `AGENTS.md` 和本地 `AGENTS.local.md` 指令。
