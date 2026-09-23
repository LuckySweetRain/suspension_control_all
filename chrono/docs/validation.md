# 本机验收记录

环境：Windows，i5-14600KF，约32 GiB内存，独立Conda环境 `chrono-rl`。
本轮使用CPU版PyTorch；没有把GPU学习或GPU车辆物理标记为已验收。
所有命令从外层仓库根目录、激活该环境后执行。未运行pytest。
最终矩阵完成时间为北京时间2026年9月23日；目录中的20260922采用运行开始的UTC日期。

## Chrono连续地形：2026年9月23日

在七类道路扩展基础上接入 `RoadTerrain(ChTerrain)` 和连续原生三角网格，保持严格七自由度。
实现及限制见 [Chrono地形](terrain.md)。实际执行：

- `python chrono/scripts/check_terrain.py --with-vector`：5组通过。
  证据 `results/chrono/terrain/checks_20260923T015858109790Z.json`。
  覆盖七类及旧random的元数据重建、空间采样和解析导数、原生C++ `GetProperties`
  经SWIG director查询、表面点/单位法线、不同前后轮距以及真实双spawn进程。
  重建与四轮激励误差均为0；两种观测分别为 `(2,58)` 和 `(2,130)`，自动重置、有限值
  和子进程退出检查通过。静态及移动道路仍为30个速度坐标、23个约束、7个机械自由度。
- `python chrono/scripts/check_model.py`：原15项物理与Gym接口检查通过。
  证据 `results/chrono/validation/check_model_20260923T015154945826Z.json`。
- 原生窗口实际验收连续叠加路面、阶跃近景和缺少道路元数据的旧记录降级显示。
  `results/chrono/terrain/native_road_validation.json` 记录：所有6个视觉刚体固定且关闭碰撞，
  渲染前后模型状态差为0、时刻不变；抽查1443个实际网格顶点，高程差不超过1.74e-18m
  （标线扣除1mm偏置）。三角形内部仍是离散近似，不能据此宣称任意点完全相等。
  10秒道路含标线116574个三角形，本次窗口初始化约1.82秒；此值不是RL吞吐。
- `python chrono/scripts/view.py --controller skyhook --scenario composite --duration 2.1 --camera road --speed 4 --screenshot results/chrono/terrain/live_continuous_road.png --record results/chrono/terrain/live_continuous_road.jsonl`：
  完成2.1秒、211帧真实状态记录，保存连续道路截图。
  同时实际回放已有的 `results/chrono/visualization/composite_road_20260923.jsonl`，
  输出 `results/chrono/terrain/replay_continuous_road.png`。
- 回放入口预检5项通过：原记录通过；前后轴位置偏移（保持总轴距）、前/后半轮距变化、
  单帧高程篡改均在创建窗口前被拒绝。
  证据 `results/chrono/terrain/replay_preflight_checks.json`。
- 外层 `python scripts/check_project.py` 全部通过：
  `results/project_checks/20260923_095632_779851/report.json`。
  `python -m ruff check chrono`、`python -m ruff format --check chrono` 通过。

首轮地形检查错误地要求Irrlicht模块完全不被导入；失败报告
`results/chrono/terrain/checks_20260923T015619244930Z.json` 保留。
查明本机官方包装器自动导入Irrlicht后，检查改为拦截可视系统和网格构造器、验证
`viewer is None` 及不加载本项目渲染模块，修正后的完整检查已通过。
本轮没有重跑长期RL训练或吞吐对比；下方旧基准不包含新增地形查询的开销。

## 七类道路扩展：2026年9月23日

新增B/C目标谱、有限上升沿阶跃、叠加输入及七类预设，并接入单类/每回合随机选类训练、
28-case固定评估、原生记录和静态预览。定义与旧半车参数差异见 [道路说明](roads.md)。

实际执行与证据：

- `python chrono/scripts/check_roads.py --with-env`：最终12项全部通过。
  报告 `results/chrono/roads/checks_20260923T013918017742Z.json` 保存该次道路扩展提交的源文件哈希。
  覆盖解析导数、前后延迟、单侧轮迹、阶跃端点和积分、叠加与单次渐入、复现、
  固定评估预设隔离、PSD积分、2048种子集合统计、旧版兼容、非法参数、Gym和七类10秒仿真。
- 默认B级目标集合RMS为5.6285mm，2048种子实测左右误差均小于1%；目标相关系数
  0/0.6/1对应样本结果约0.0252/0.6216/1。同seed的C级幅值为B级2倍。
  这些是合成器统计检查，不是道路等级认证。
- `python chrono/scripts/check_model.py --output results/chrono/validation/road_extension_model_check_20260923.json`：
  原15项模型与接口检查全部通过。
- `python chrono/scripts/train.py --algorithm ppo --observation-mode sensors --workers 2 --steps 2048 --config chrono/configs/seven_roads_training.yaml --name seven_roads_smoke_20260923 --device cpu`：
  真实spawn并行采样、PPO更新及检查点保存完成；该短训不证明收敛。
- `python chrono/scripts/evaluate.py --baseline passive --road-suite seven --name seven_roads_passive_20260923`：
  28/28个测试case完整结束、0提前终止、0设计行程超限；12个case出现过轮胎卸载。
  所有已评估道路元数据与最终七类预设逐项核对一致。
- `python chrono/scripts/preview_roads.py --output-dir results/chrono/road_preview/seven_roads_final_20260923`：
  生成9张PNG、9份PDF、7份四轮高度/速度CSV和元数据；每类5001点，0–10秒，采样2ms。
  已检查总览、局部事件和理论PSD图；预览不包含车辆仿真。
- 原生Skyhook＋叠加路面实际运行2.04秒，截图及轨迹：
  `results/chrono/visualization/composite_road_20260923.png`、同名JSONL。
- 外层 `python scripts/check_project.py` 全部通过：
  `results/project_checks/20260923_093115_626911/report.json`。Ruff与格式检查通过，未运行pytest。

本轮没有重新运行完整12组RL矩阵；新七类均值不能直接与下方旧20-case实验均值比较。
固定道路套件默认仍为legacy，显式 `--road-suite seven` 才使用28case。

## 动力学和接口

`python chrono/scripts/check_model.py` 的15项检查全部通过，包括：

- 30个速度坐标、23个约束，严格7个机械自由度。
- 10秒静平衡、左右对称激励、单侧激励、主动作用力方向。
- 执行器时延、力/变化率限幅，瞬时功率与目标功率降额的区别。
- 单边轮胎接触、精确轴距时延、随机种子复现、重复清理。
- 基于Chrono实际端点和力回调的完整静态刚度/阻尼/输入雅可比核验。
- 1ms与0.5ms步长比较、JSON渲染帧、两种观测下Gymnasium/SB3检查。
- 低安装高度的参数校验及真实悬架端点穿越回归。

详细证据：外层 `results/chrono/validation/check_model_final.json`。
三秒被动凸包检查中，步长减半的车身加速度RMS相对差约0.4%；这不是所有
高幅值路面和饱和控制下的统一误差保证。Gymnasium关于无限观测边界的提示仍保留，
训练通过VecNormalize处理不同信号尺度。

## 原生窗口与基线

Irrlicht实际窗口、Tk原生控制面板、暂停/倍速/视角/外壳切换、201帧轨迹记录
及回放已在本机运行。独立渲染系统不会改变被控对象的状态。
截图及JSONL位于 `results/chrono/visualization/`。

原生窗口还实际加载了 `final_matrix_20260922` 中的 SAC 传感器策略
（seed 42，8192步），恢复相同观测归一化后完成0.64秒单侧凸包推理与渲染。
证据为 `results/chrono/visualization/rl_policy_final.png` 及同名JSONL，
其中保存65帧真实状态；这项检查证明策略与原生画面接通，不代表策略已收敛。

被动、Skyhook与LQR分别完成20个固定测试case，共60个完整回合，未触发提前终止。
这不表示没有瞬时轮胎卸载，也不表示主动控制已经调优。

| 控制器 | 20个case的加速度RMS平均值 m/s² | 轮胎卸载比例的case均值 |
|---|---:|---:|
| 被动 | 0.34225 | 0.1070% |
| Skyhook | 0.37308 | 0.1100% |
| LQR | 0.35122 | 0.1065% |

这里是各case指标的等权平均，包含平路；不是把全部信号拼接后计算的RMS。
原始case结果位于 `results/chrono/evaluation/acceptance_*_20260922/`。
基线系数未按测试集调参，不能从这些初始结果推断任何算法的最佳性能。

## 强化学习与依赖

PPO传感器模式使用2个spawn进程，SAC完整状态模式使用1个进程，均实际完成
2048→4096步的训练与恢复。检查覆盖SAC梯度更新与回放恢复、归一化统计、
优化器状态、40步策略推理以及有限值/动作边界。
证据：`results/chrono/checks/rl_runtime_smoke/smoke.json`。

`python chrono/scripts/benchmark.py --vector-steps 2000 --name final_workers_20260922`
实际测得：

| 独立环境进程 | 总控制步/秒 | 启动秒数 |
|---|---:|---:|
| 1 | 429.6 | 1.63 |
| 2 | 733.5 | 1.73 |
| 4 | 1458.5 | 1.80 |
| 8 | 1442.2 | 2.19 |

默认控制周期为10ms、每控制步包含10个物理步。上表为纯采样吞吐，不含网络学习。
按峰值95%以内优先较少进程的规则，推荐4个进程；不是宣称8进程总是更快。
结果保存在 `results/chrono/benchmarks/final_workers_20260922/benchmark.json`。

真实硬预算测试已确认实验父进程及其两个Chrono子进程均可被清理，CLI的1.8秒
截止测试保留了12个未完成训练组和15个未完成评估控制器清单。
证据：`results/chrono/checks/watchdog_runtime_smoke.json`。

`python -m pip check` 通过，开发安装后已在仓库外验证Python API导入。
Conda与pip精确包记录保存在 `results/chrono/validation/`。
外层 `python scripts/check_project.py` 全部通过，原半车项目未被替换。

## 最终短程实验矩阵

实际执行：

```powershell
python chrono/scripts/run_experiments.py --name final_matrix_20260922 --budget-hours 1 --max-steps 8192 --workers 4 --pilot-steps 2048
python chrono/scripts/report.py --experiment final_matrix_20260922
```

PPO/SAC × privileged/sensors × seed 42/43/44，共12组，均完成8192个累计环境步，
统一加载该步数的完整检查点。3个基线与12个策略各完成20个固定测试case，
共300个完整回合，0个提前终止、0个未完成。实验状态为 `finished`，计入预算约
1038.7秒（17.3分钟），没有超出1小时短训预算，也没有运行满24小时。
原始检查点、逐case记录及汇总位于
`results/chrono/experiments/final_matrix_20260922/`，报告为 `report.md`、`report.csv`。

本轮全部RL策略的case平均车身加速度RMS均高于被动基线：被动约0.342 m/s²，
RL约0.385～1.048 m/s²，因此没有证明舒适性提升或训练收敛。
6个SAC策略均至少在一个测试case超过90mm设计行程，最大约117.2mm；
所有控制器都出现过瞬时轮胎卸载。“完成”仅表示跑满全时程、未触发数值或物理包络
终止，不等于满足约束。设计行程进入奖励惩罚，行程的提前终止阈值为其2倍；
悬架安装端点穿越和其他异常另有终止条件。不得只看0个终止就声称安全达标。

较早的 `acceptance_matrix_20260922` 在几何适用域审查后被主动停止并标记为
`superseded`；该运行不用于最终结果。上面的完整矩阵已使用最终几何检查重新执行。

批量矩阵、预算和报告入口见 `training.md`。短训只证明流程可用，
不能证明策略收敛、实车安全或模型比CarSim更准确。参数标定边界见 `parameter_sources.md`。
