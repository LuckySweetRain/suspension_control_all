# AGENTS.md

This file provides guidance to the AI agent when working with code in this repository.

## 项目概述

纯 Python 半车主动悬架控制研究项目。包含 RL 基线（DDPG/TD3/SAC/PPO）、经典控制器（PID/SPDF）、预览 MPC，以及一个独立的 MATLAB/CarSim 联合仿真子项目。

## 运行命令

所有脚本从仓库根目录直接运行，无 `pip install -e .`，无 Makefile：

- 训练 RL：`python scripts/train_rl.py --algorithm sac --config configs/train_fast.yaml`
- 评估全部控制器：`python scripts/evaluate_all.py --config configs/default.yaml`
- `evaluate_all.py --checkpoints` 用逗号分隔的 `算法=路径`，支持字面量 `latest`（按 mtime 解析最新 `final.pt`）

脚本自行把仓库根注入 `sys.path`，所以 `from config import ...`、`from controllers import ...` 只在根目录为 CWD 时可用。

## 配置系统

`config.py` 实现自定义 `inherits:` 链：YAML 顶部写 `inherits: default.yaml`，`load_config()` 递归深合并、子配置覆盖父配置。**修改 `configs/default.yaml` 会静默影响所有继承它的子配置**（half_car_bp_nmps.yaml、half_car_safe_ppo.yaml、ppo.yaml 等）。改基线前先 `grep -r "inherits" configs/` 看影响范围。

## 依赖注意事项

- `scipy` 已在 requirements.txt 中（MPC 和图脚本需要）。
- `torch` 自动选 CUDA（`cuda if available else cpu`），无环境变量强制 CPU；需 CPU 时编程传入 `device=`。
- `datasets/expert/half_car_spdf.npz` 和训练 checkpoint（`*.pt`）被 gitignore——新克隆缺 BC 预训练数据，需先用 `scripts/collect_expert_dataset.py` 重新生成才能跑 SI-RPPO 的 BC 预训练。

## 测试与验证

本仓库**没有 pytest 测试套件**（README 说明已移除）。不要声称或运行 `pytest`。最接近的验证手段：
- `ruff check`（lint，配置见 `pyproject.toml`：E/F/I/B/UP，忽略 E501/E741，行长 100，目标 py310）
- `ruff format`（格式化，同样读取 `pyproject.toml` 的 `line-length=100`）
- `python scripts/validate_training_config.py --config <cfg> --out results/validate.json`
- `python scripts/check_gym_registration.py --config <cfg> --smoke`

`.qoder/settings.json` 配置了 PostToolUse 钩子：每次 Write/Edit 后自动对 .py 文件运行 `ruff check --quiet`，语法错误和 lint 问题会立即反馈给 AI。

代码风格跟随文件现状：4 空格缩进、双引号、`from __future__ import annotations`。ruff 的 I 规则（导入排序）可能触发对现有文件的重新排序，AI 可按需 `ruff check --fix` 或保留原样。无 CI。

## 子项目边界

- `carsim_mpc_quick/` 是独立 MATLAB+CarSim 子项目，需 MATLAB、Simulink `.slx`、以及绝对路径 `E:\Desktop\workspace\CarSim_database\carsim_active_suspension` 的 CarSim 数据库。不从 Python 运行。
- `projects/` 和 `tools/create_metric_formula_slide.ps1` 是论文答辩 PPT 生成工件，非 Python 源码。

## 仓库根临时文件

`tmp/`、`tmp_*.pptx`、`.conda_envs/`、`.conda_pkgs/`、`tmp_docx_qa/`、`tmp_ppt_check/` 等是本地工件，**禁止读取、修改或删除**，除非用户明确要求。

## Git 约定

- 分支为 `master`（非 `main`）。
- 直接提交到 master，不使用 PR。
- commit message 用中文（参考现有："修复...", "半车..."）。
- 不要信任文档中引用的已删除配置（`docs/model_parameter_audit.md` 和 README 提到 `configs/mujoco_full_car_*.yaml` 等已删除的路径）。

---

如需私人、不提交的本地偏好，可创建 `AGENTS.local.md`（gitignore，优先级高于本文件）。模块特定指令可在子目录放 `AGENTS.md`，在工作于该目录时自动加载。
