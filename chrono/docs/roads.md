# 七类道路与四轮激励

道路在空间中定义，通过 `RoadTerrain(pychrono.vehicle.ChTerrain)` 提供连续高程、表面点和法线，
再按给定车速采样四轮路径。轮胎力与原生连续道路网格使用同一个高程模型。
七自由度与垂向轮胎本构保持不变；地形坐标、横向插值及网格精度见 [Chrono地形](terrain.md)。

## 七类预设

配置在 `chrono/configs/roads/`，默认车速15m/s。尺寸均为米；这是独立的Chrono研究预设，
不是原半车道路的逐项复现。

| 名称 / kind | 默认形状与参数 | 配置文件 |
|---|---|---|
| B级 / `iso_b` | B级代表PSD，频带0.02–2 cycles/m | `iso_b.yaml` |
| C级 / `iso_c` | C级代表PSD，频带同上 | `iso_c.yaml` |
| 正弦 / `sine` | 幅值0.03，波长5 | `sine.yaml` |
| 阶跃 / `step` | 9m处开始上升0.03，过渡长0.3，此后保持 | `step.yaml` |
| 凹坑 / `pothole` | 9m处开始，深0.03、长0.8，负向升余弦 | `pothole.yaml` |
| 凸包 / `bump` | 9m处开始，高0.03、长0.8，正向升余弦 | `bump.yaml` |
| 叠加 / `composite` | B级背景＋30m处凸包＋75m处凹坑，局部尺寸均0.03×0.8 | `composite.yaml` |

局部事件可设置 `tracks: both/left/right`。叠加是同一位置处的高度与坡度分别求和，
各子项共用车速和一个渐入包络，不是分段拼接。可独立选子项轮迹和种子；未指定子项
种子时从父种子派生。原有 `flat`、`random`、`left_bump` 及默认legacy评估仍保留。

## B/C级谱定义

采用单边空间位移PSD：`Gd(n)=Gd(n0)*(n/n0)^(-2)`，`n0=0.1 cycles/m`，单位m³。
单边PSD在频带内的积分等于高程方差，见
[BS ISO 8608:2016官方预览§3.1–3.2](https://webstore.ansi.org/preview-pages/BSI/preview_30341600.pdf)。

| 等级 | 下界 | 默认几何均值 | 上界 |
|---|---:|---:|---:|
| B | 32 | 64 | 128 |
| C | 128 | 256 | 512 |

上表单位为10⁻⁶m³，均对应n₀=0.1 cycles/m；等级表见
[Liu等，2024，Table 1](https://www.mdpi.com/2076-3417/14/22/10310)。
可用 `gdn0_m3` 选择等级范围内的值；`roughness_m` 只控制旧 `random`。
空间频率cycles/m不能与角空间频率rad/m直接混用。

默认256个对数划分频带，用各频带几何中心频率和独立随机相位合成余弦。
幅值依据单边PSD方差定义逐频带精确积分：

`A_i = sqrt(2 * Gd(n0) * n0^2 * (1/n_lo - 1/n_hi))`。

每项余弦方差为Aᵢ²/2，不按单个种子的有限样本RMS再次缩放。
频带参数为 `n_min_per_m`、`n_max_per_m`，离散数量为 `spectral_modes`。
默认频带下理论集合RMS约B级5.63mm、C级11.26mm；短路段实际RMS会变化。
谱斜率2属于本合成模型假设。

左右轮迹采用公共分量与独立分量混合，保持两侧目标PSD。
`left_right_correlation` 是渐入前的目标集合相关系数ρ，默认0.6；理论幅值平方相干度
为ρ²，不能混称。有限路段相关性不保证等于ρ；常数相关性也不代表实测频率相关相干关系。

这是基于ISO 8608目标谱的有限带宽合成路面，不是道路等级认证。
[ISO标准正式范围](https://www.iso.org/standard/71202.html)是实测道路高程数据的统一报告。

## 几何延迟、渐入与阶跃

四轮顺序FL、FR、RL、RR。前轮取x=vt，后轮取x=vt−L；同侧后轮严格延迟L/v。
本车轴距2.776m、车速15m/s时延迟0.1851s。输出速度采用解析链式法则 `dz/dt=v*dz/dx`。

前3m默认采用三次平滑渐入，保证静平衡初始化时输入高程和速度连续。
`run_in_m: 0` 可关闭，此时随机或带相位正弦可能在起点非零。
渐入在组合后只应用一次，且按轮迹空间位置定义，保持后轮延迟。

阶跃采用 `H*(10u^3-15u^4+6u^5)`，u为过渡区归一化位置，两端一阶、二阶导数为零。
默认0.3m过渡在15m/s下耗时20ms。这是有限上升沿阶跃，**不是瞬时不连续阶跃**；
过渡长度必须大于零。理想高度跳变的速度含冲激，不能把它的速度直接写成0来代替。

随机路面最高时间频率为 `speed_mps*n_max_per_m`；采样必须满足Nyquist条件并检查
积分步长收敛。局部事件的持续时间也应覆盖足够物理步，低采样率预览可能遗漏尖峰。

## 使用

从仓库根目录运行，先 `conda activate chrono-rl`：

```powershell
python chrono/scripts/view.py --controller passive --scenario iso_c --hold
python chrono/scripts/view.py --controller skyhook --config chrono/configs/roads/composite.yaml --hold
python chrono/scripts/preview_roads.py --output-dir results/chrono/road_preview/my_seven_roads
python chrono/scripts/check_roads.py --with-env
python chrono/scripts/check_terrain.py --with-vector
```

预览导出总览、各类局部放大图、理论PSD图（PNG/PDF）、四轮高度/速度CSV和元数据，
不会覆盖已有输出目录。理论PSD图不是短轨迹的实测谱估计。

```powershell
# 每回合均匀选择七类之一，整个回合保持该场景
python chrono/scripts/train.py --algorithm ppo --observation-mode sensors --workers 4 --steps 65536 --config chrono/configs/seven_roads_training.yaml --name seven_roads_ppo
# 7类×4固定种子=28case，与旧20-case结果分开比较
python chrono/scripts/evaluate.py --baseline passive --road-suite seven --name seven_roads_passive
# 12组矩阵：七类混合训练和七类评估，总预算上限24小时
python chrono/scripts/run_experiments.py --config chrono/configs/seven_roads_training.yaml --road-suite seven --name seven_roads_matrix --budget-hours 24
```

`road.scenarios` 是可选训练场景列表，无显式场景的reset才会随机选择。
显式 `reset(options={'scenario': {...}})` 绕过该选择，供固定测试使用。
动作和两套观测维度保持不变，传感器策略不会自动得到路面类别或未来高度。

`reset` 返回的 `info['road_metadata']` 记录实际道路种子和谱参数；评估逐case JSON及
原生轨迹头也保存它。预览的seed直接用于发生器；Gym `reset(seed=...)` 先控制环境
随机流，再派生道路种子。要逐点匹配两种入口，应将元数据中的道路seed传给预览，或
显式配置 `road.seed`。训练时固定road.seed或子项seed会固定相应随机道路。

## 与外层半车定义的差异

旧 `configs/default.yaml` 七类配置仍保留。旧B/C滤波白噪声模型的G常数为128/512×10⁻⁶，
本实现默认64/256×10⁻⁶并明确采用单边空间PSD；合成方法、频带和启动处理也不同，不能
把本实现视作旧随机道路的等价复现。

旧凸包高0.10m、长0.25m、速度22.2222m/s，持续仅11.25ms；新预设较平缓。
旧正弦幅值0.025m、阶跃高0.05m、凹坑深0.06m长0.5m。旧尺寸可以手动配置，
但需重新检查步长和物理包络；时间起点转换为空间起点用 `start_m=speed_mps*start_time`。
旧阶跃速度恒0，未表示数学跳变处的冲激。

旧叠加为B级＋正弦＋凹坑，随机子项seed固定3021；本轮按用户确认采用随机背景＋
局部凹坑/凸包。旧 `road_scale=0.1` 仅是预瞄观测归一化尺度，不代表物理高程缩小十倍。
