# Chrono 七自由度主动悬架

独立的 Project Chrono / Gymnasium 项目：七个机械自由度、四轮主动执行器、
Irrlicht 原生可视化、CPU 多进程采样和 Stable-Baselines3 PPO/SAC。
代码兼容 Python 3.10+，安装环境和参数来源见 `environment.yml` 与
`docs/parameter_sources.md`。所有命令从外层仓库根目录执行。

## 模型边界

车身保留升沉、侧倾、俯仰；四个簧下质量仅保留垂向运动。五个动态刚体
有 30 个速度坐标，通过 23 个独立约束得到 7 个机械自由度。
坐标系 x 向前、y 向左、z 向上；正侧倾使左侧上升，正俯仰使车头下降。
执行器正力使车身与车轮分离，车身与簧下质量受到大小相等方向相反的力。

车辆是公开参数派生的研究参考模型，不是经过实车试验标定的量产车数字孪生。
七自由度不含纵向/横向运动、横摆、轮胎侧偏、车轮旋转动力学、完整连杆运动和柔性车身。
原生画面中的车壳、轮胎外形是视觉资产，不会改变质量、惯量或增加隐含自由度。

模型包括重力静态预载、有限转角的悬架几何、可配置弹簧非线性、压缩/回弹阻尼、
渐进式压缩与伸张限位、单边轮胎接触，以及执行器一阶响应、纯时延、力/速率限幅和目标功率降额。
控制器下发四个归一化目标力，物理引擎不会把命令直接当作瞬时实际作动力。
机械功率记录不等于电池功率或液压泵功率；没有假设回馈效率。

## 入口

首次安装（Windows 参考环境）：

```powershell
conda env create -f chrono/environment.yml
conda activate chrono-rl
python -m pip install -e chrono --no-deps
```

环境单独安装，避免改变外层半车项目。官方 Chrono 包含本项目没有调用的
Sensor/CUDA 依赖，下载体积较大；物理求解由CPU执行。默认CPU版PyTorch支持全部训练功能。
需要GPU学习器时运行 `python -m pip install -r chrono/requirements-cuda.txt`，
该版本使用自身CUDA 12.6运行库；须用实际 `torch.cuda.is_available()` 与训练检查验收。
PyTorch构建选项见[官方安装记录](https://pytorch.org/get-started/previous-versions/)。

```powershell
conda activate chrono-rl
python chrono/scripts/check_model.py
python chrono/scripts/view.py --controller passive --scenario bump
python chrono/scripts/view.py --controller skyhook --scenario left_bump
python chrono/scripts/benchmark.py --help
python chrono/scripts/train.py --help
python chrono/scripts/evaluate.py --baseline passive
python chrono/scripts/run_experiments.py --help
```

`check_model.py` 是物理与环境检查脚本，不使用 pytest。
完整 CLI 参数见 `docs/training.md` 和 `docs/visualization.md`。
产物统一写入外层 `results/chrono/`，不写入源代码目录。

道路支持B/C级目标谱、正弦、有限上升沿阶跃、凹坑、凸包，以及随机背景叠加局部事件。
七类配置、四轮曲线导出、混合训练和28-case评估见 [道路建模](docs/roads.md)。
原有随机路面和20-case评估仍保留，旧结果不自动改用新道路。

## Python 接口

安装开发包后可直接调用：

```python
from chrono_suspension.env import ChronoSuspensionEnv

env = ChronoSuspensionEnv(observation_mode="sensors")
obs, info = env.reset(seed=42, options={"scenario": {"kind": "bump"}})
for _ in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.baseline_action("skyhook"))
    if terminated or truncated:
        break
env.close()
```

- 动作：`Box(-1, 1, shape=(4,))`，顺序 FL、FR、RL、RR，乘以配置中的 N 制力限值。
- `privileged`：7 个位置、7 个速度、4 个实际执行器力及延迟命令队列；可选路面预瞄。
- `sensors`：加噪的重力补偿垂向加速度、姿态角估计、角速度、4 个悬架行程及已知命令，
  加固定传感器延迟与历史窗口；不包含真实轮胎载荷、簧下位置或未来路面。
- 默认无预瞄。实际可观测信号与估计假设见 `docs/model.md`。
- `info` 提供 SI 单位诊断与约束指标，诊断不会自动加入策略观测。
- 时间到达返回 `truncated`；非有限状态或超出物理包络返回 `terminated` 并保留故障信息。

## 实验与可复现性

训练在没有渲染窗口的独立 CPU 进程中运行，不把 Chrono 描述为 GPU 向量物理引擎。
主进程负责网络学习。种子、完整配置、环境版本和归一化统计随 checkpoint 保存。
两种观测 × PPO/SAC × 3 个种子构成 12 组实验，单次总计算预算上限 24 小时。
统一测试路面上比较被动、主动 skyhook、全状态 LQR 和 RL；LQR/skyhook 为拥有状态信息的
参考控制器，不能把它们与传感器策略的差异全部归因为算法能力。

加速度指标是未加权 RMS，不宣称满足 ISO 舒适性认证。
短训与数值稳定检查不能证明训练收敛、实车安全或相对 CarSim 的精度提升。

本机已运行的检查、原生截图、并行吞吐和结果路径见 [验收记录](docs/validation.md)。
已完成12组各8192步的短训及300个测试回合，约17.3分钟；本轮RL舒适性指标未优于被动基线，
部分策略超过设计行程。这是训练流程验收，后续控制研究须继续训练、调优并验证约束。
参数证据见 [参数来源](docs/parameter_sources.md)，力学与观测定义见 [模型说明](docs/model.md)。
