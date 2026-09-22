# 批量强化学习与实验预算

训练入口使用 Stable-Baselines3 的 PPO/SAC。Chrono 刚体系统只在各自的环境进程内创建；模型、渲染器与 SWIG 对象不跨进程序列化。动作始终为四个归一化主动力命令，顺序为 FL、FR、RL、RR，范围为 `[-1, 1]`。

所有命令从仓库根目录执行，先激活独立的 `chrono-rl` Conda 环境。产物统一在仓库的 `results/chrono/`，不写入旧训练目录。

## 吞吐检查与单次训练

```powershell
conda activate chrono-rl
python chrono/scripts/benchmark.py
python chrono/scripts/train.py --algorithm ppo --observation-mode privileged --workers 4 --steps 65536 --name ppo_first
python chrono/scripts/train.py --algorithm sac --observation-mode sensors --workers 4 --steps 65536 --name sac_sensors_first
```

基准逐项测量 1/2/4/8 个 headless 环境，报告启动时间、aggregate transitions/s，以及模拟时间/实际时间。一个环境用 `DummyVecEnv`；多个环境用 `SubprocVecEnv(start_method="spawn")`。每个子进程限制 OpenMP/BLAS/PyTorch 为单线程，避免重复开线程造成过度占用。自动选择在峰值速度 95% 以内最少的 worker 数。这个数字仅反映物理步进，实际学习成本由后续 pilot 测量。

PPO 默认 CPU；SAC 默认在 `torch.cuda.is_available()` 为真时使用 CUDA，也可以明确 `--device cpu`。仿真仍是 CPU 多进程，并没有把整个 Chrono 系统搬到 GPU。Windows 的入口必须在 `if __name__ == "__main__"` 中调用；现有脚本已处理。

在自己的Python脚本中并行调用：

```python
import numpy as np
from chrono_suspension.config import load_config
from chrono_suspension.training import make_vector_env

if __name__ == "__main__":
    vector = make_vector_env(load_config(), observation_mode="sensors", seed=42, n_envs=4)
    try:
        observation = vector.reset()  # shape (4, 130), already normalized
        for _ in range(100):
            actions = np.zeros((4, 4), dtype=np.float32)  # replace with your policies
            observation, rewards, dones, infos = vector.step(actions)
    finally:
        vector.close()
```

保存为脚本运行；Windows不能把该spawn示例直接粘到没有主入口保护的交互控制台。
这里返回SB3 VecEnv的四元组，结束的环境会自动reset；该回合的终态观测在对应
`infos[i]["terminal_observation"]` 中，也经过VecNormalize归一化。单环境 `ChronoSuspensionEnv.step` 则遵循Gymnasium
五元组。`infos[i]["metrics"]` 保留未归一化的物理诊断量，不能误当策略输入。

`--steps` 表示所有 worker 加总的绝对环境 transition 目标，不是每个 worker 的步数。PPO 每个 worker 收集 256 个 transition 才更新，单次训练脚本将目标向上对齐至 `256 * workers`。例如 4 worker 的最小比较量子是 1024。实验矩阵的 100 万上限向下对齐，避免越过上限。

## 保存、恢复与推理

每个 checkpoint 包含 `model.zip`、`vecnormalize.pkl`、完整 `config.json` 和 `metadata.json`。后者记录算法、观测模式、aggregate 步数、worker 数、版本、配置哈希、保存时源码文件哈希、下一个 episode 的种子及是否完成训练更新。

SAC 的回放保存在同一 run 下的 `replay/`。两个原子替换的 bank 保留最近两个完成检查点的回放；另一个 partial bank 用于预算打断时的诊断。历史权重、归一化统计和元数据保留用于评估；更旧的回放 bank 会被新快照替换。恢复时严格校验回放归属，拒绝把当前回放配给旧策略。搬移训练结果时必须一起保留整个 run 目录，不能只复制 checkpoint 子目录。

`latest.json` 指向最近一次保存，`latest_complete.json` 指向最近完成更新的可恢复检查点。预算打断可能发生在 PPO rollout 中途，此时保存标记为 `complete_update=false`，不能用它代表一个已完成的公平比较步数，也不能用 `train_chunk` 接续为对齐实验。

```powershell
# 使用 latest_complete.json 内的 checkpoint 路径；--steps 是恢复后的总目标。
python chrono/scripts/train.py --resume "E:\Desktop\workspace\suspension_control_all\results\chrono\training\ppo_first\checkpoints\step_0000065536_REPLACE" --steps 131072 --name ppo_continued
```

恢复保留策略、优化器、观测归一化、SAC 回放与 worker 的下一回合种子；物理系统从新 episode 重建。没有保存原生求解器内部状态、正在运行的 episode、全部随机数发生器和 CUDA 状态，因此不宣称位级恢复。恢复配置、算法、观测模式、seed、worker 数必须与检查点匹配。

推理使用 `load_policy(checkpoint, env, device="cpu")`，返回接收原始观测的确定性策略函数。它加载同一套归一化统计并冻结更新，由调用者直接 `env.reset/step/close`，不会隐式 auto-reset。只加载自己信任的检查点，SB3/VecNormalize 使用 pickle 序列化。

## 12 组受限预算实验

```powershell
python chrono/scripts/run_experiments.py --describe
python chrono/scripts/run_experiments.py --name sedan_matrix --budget-hours 24
```

固定矩阵为 PPO/SAC × privileged/sensors × seed 42/43/44，共 12 组。默认先完成 1/2/4/8 worker 物理吞吐基准，再对全部 12 组各跑至少 2048 transition 的学习 pilot。pilot 计入环境启动、数据采集、梯度更新和 checkpoint I/O，不能用纯物理 FPS 冒充训练 FPS。

实验时钟从运行入口开始，安装、研发、手工先验检查不计入。总预算最多 24 小时，其中最多 80% 用于基准、pilot 与学习，预留 20% 供评估。根据所有组的 pilot 成本推导共同目标，并额外乘以 1.5 成本系数，缓解 SAC pilot 尚包含 replay 预热导致的乐观估计。目标上限为每组 100 万 aggregate transition。

完成 pilot 后按共同步数分块轮转，每轮改变首个组。只有所有 12 组均完成相同目标后才推进 `common_completed_steps`。某一组较早结束或耗尽预算时，正式比较仍加载最后一个全部组共有的完成检查点。结果同时保留计划目标、100 万请求上限未达清单，以及训练/评估未完成清单，不因先训练的组跑得更多而优先展示它。

CLI 的独立父进程 watchdog 在总预算用尽时终止实验进程树；普通 `train.py --budget-seconds` 是合作式截止，不能中断正在执行的原生求解或梯度计算。在 watchdog 强制终止时，仅原子发布的检查点和逐 case JSON 有效，`.saving-*` 等尚未发布内容不能当作完成产物。

若外部错误中断，可用同名和相同预算、步数上限 `--resume` 接续，并扣除已记录的计算时间；正常完成或时间耗尽的实验不在同一预算内自动重跑。若要改变配置或追加预算，应创建新的实验名。断电或操作系统直接杀死整个父进程时，最后一次状态写入后的少量运行时间可能无法计量。

## 测试集与结果解释

固定正式测试为 flat、bump、left_bump、sine、random 各配 1001–1004 四个种子，车速均为 15 m/s，共 20 case。验证集使用 2001–2004；训练 episode 使用独立的大整数种子流，并按 run seed、worker rank、episode 分开。测试集不能用于超参数选择。默认训练路面按配置的 `road` 生成，不会偷偷用测试 seed 调参。

平路、确定性减速带和正弦场景的多个种子可能只改变传感器噪声，不能把它们描述成多个独立道路样本。传感器模式含观测延迟和噪声，privileged 模式可以读取真实状态；LQR/skyhook 使用模型状态，属于有状态信息的参考基线，不能声称它们与 sensors 策略拥有同样的可用测量。

评估同时运行 passive、skyhook、LQR 及 12 个策略。输出 CSV/JSON 包含车身加速度、俯仰/侧倾 RMS、悬架行程峰值、轮胎最小载荷与卸载比例、执行器力和机械功。加速度/姿态/力 RMS、机械功、行程峰值、最小载荷和卸载比例使用 1 ms 物理子步汇总，避免丢掉控制周期内的峰值；额外的行程 RMS 与载荷标准差仍为控制周期采样指标。正/负机械功不等于实际电耗或电池回收能量。

汇总对完整且未提前终止的 case 计算均值，同时单列提前终止和未完成 case；不把提前失败的短记录混进均值来制造更低 RMS。`successful_cases` 仅表示完整运行，并不表示从未发生轮胎卸载或行程违规，须同时检查约束指标。TensorBoard 和物理日志只用于训练诊断，正式比较依赖固定测试集。

```powershell
tensorboard --logdir results/chrono
```

已有实验可随时生成中文 Markdown 与 CSV 汇总，无需启动 Chrono 或加载模型：

```powershell
python chrono/scripts/report.py --experiment sedan_matrix
```

也可将 `--experiment` 指向 `results/chrono/` 内的完整实验目录或 `experiment.json`。输出保存在该目录的 `report.md`、`report.csv`，包括 12 组策略和 3 个基线、共同训练步数、已结束/未完成 case、失败率与物理指标。报告优先读取对应共同步数的评估摘要；强制中断后可从已原子写入的逐 case JSON 恢复部分结果。缺失数据不填零，成功 case 的均值必须结合失败率阅读，不自动计算性能提升或统计显著性。

短时运行只能验证接口、有限值、检查点和并行链路。它不能证明策略收敛、比被动悬架更优，也不能替代实车、试验台或已标定高保真模型的验证。

接口依据：[SB3 向量环境与 VecNormalize](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)、[SB3 SAC 与回放保存接口](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)。实际依赖版本以运行检查点的 `metadata.json` 为准。
