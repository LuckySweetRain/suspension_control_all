# suspension_control_all

纯 Python 半车主动悬架控制实验项目。

本代码仓库围绕可运行的控制栈进行组织。

## 目录结构 (Structure)

- `config.py` - YAML 配置文件加载和算法配置覆盖。
- `configs/` - 实验和环境配置文件。
- `controllers/` - 经典控制、先验控制、MPC（模型预测控制）以及基于参数策略的控制器。
- `envs/` - 半车模型 (half-car) Gymnasium 环境以及环境注册代码。
- `experiments/` - 共享的实验评估程序。
- `models/` - 物理模型定义。
- `rl/` - DDPG, TD3, SAC, PPO 强化学习算法，及其相关的神经网络与经验回放池。
- `roads/` - 路面轮廓生成与导入辅助工具。
- `scripts/` - 可单独运行的训练、评估、验证及数据实用脚本。
- `visualization/` - 绘图与动画辅助工具。
- `docs/` - 保留的设计笔记和实验日志。

移除了项目中繁杂的冗余文件，包括 pytest 测试套件与缓存、临时 pytest 工作区、备份文件、日志、生成的论文/导出打包文件以及论文构建输出内容。

## 安装依赖 (Setup)

```bash
python -m pip install -r requirements.txt
```

## 常用命令 (Common Commands)

训练一个快速的强化学习基线模型：

```bash
python scripts/train_rl.py --algorithm sac --config configs/train_fast.yaml
```

评估各个控制器及可选的模型权重 (checkpoints)：

```bash
python scripts/evaluate_all.py --config configs/default.yaml
```

生成路面数据集：

```bash
python scripts/generate_road_dataset.py --out datasets/mujoco_roads --duration 8
```

验证 MuJoCo 训练配置：

```bash
python scripts/validate_training_config.py --config configs/mujoco_full_car_corner.yaml --out results/mujoco_config_validation.json
python scripts/validate_mujoco_env.py --config configs/mujoco_full_car_corner.yaml --max-steps 200 --action-mode random --out results/mujoco_env_validation.json
```

运行基准测试流水线：

```bash
python scripts/run_mujoco_benchmark.py --config configs/mujoco_full_car_corner.yaml --generate-roads --out results/mujoco_benchmark
```

绘制已有结果目录的图表：

```bash
python scripts/plot_results.py --result-dir results/<run>
```
