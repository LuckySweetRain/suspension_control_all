# CarSim 快速预瞄 QP-MPC

该工程使用 CarSim 作为高保真整车被控对象，以线性半车模型作为 MPC 内部预测模型。控制器在线优化前、后轴两个作动力，并作为残差增量叠加在固定线性被动阻尼力上：

```text
IMP_FD_L1 = baseline_damper_L1 + scale × front_axle_force / 2
IMP_FD_R1 = baseline_damper_R1 + scale × front_axle_force / 2
IMP_FD_L2 = baseline_damper_L2 + scale × rear_axle_force / 2
IMP_FD_R2 = baseline_damper_R2 + scale × rear_axle_force / 2
```

当前默认 `scale=1.00`。控制器的输入矩阵已经显式包含部署比例和前、后悬架
运动比，因此 QP 预测的广义轴力与 CarSim 实际执行链一致。原来的 `scale=0.10`
会使实际等效轴力仅为预测指令的 6.51% 和 7.97%，现已不再作为默认配置。

这是凸 QP-MPC，不是增益型 MPC-lite。由于本机没有 Optimization Toolbox，QP 由项目内置的 OSQP 风格 ADMM 求解器完成，不依赖 `quadprog`。QP 包含：

- 20 步、0.20 s 道路预瞄；
- 车身垂向加速度和俯仰加速度代价；
- 前后悬架行程、轮胎挠度和控制力代价；
- 前后轴作动力硬约束；
- 作动力变化率硬约束；
- 基于道路感知误差上界的悬架行程约束收紧；
- 下发前再次执行作动力及变化率投影。

预测模型的被动阻尼由 CarSim 角点阻尼律 `3 N/(mm/s)` 换算得到：前、后轴
分别为 `2546.2 N*s/m` 和 `3811.3 N*s/m`。悬架行程现在作为在线 QP 硬约束
参与每一步上下界计算，不再只是预先生成但未使用的矩阵。前、后动态行程边界
均为 80 mm；原来的 45/55 mm 小于 CarSim C 级道路被动工况的实际行程，会使
QP 先验不可行。

## 距离域道路感知

道路传感器按固定空间栅格工作，默认分辨率为 `0.05 m`、有效感知距离为
`3.0 m`。噪声和偏置固定在道路坐标上，不会随控制器调用重新生成。MPC 仍在
时间域预测，实时车速只负责把第 `i` 步映射到空间查询点：

```text
preview_distance(i) = min(vehicle_speed * i * control_dt, sensor_range)
```

在 `N=20`、`control_dt=0.01 s` 时，35 km/h 需要约 1.94 m 的道路信息，3 m
传感器可以完整覆盖；80 km/h 时需要 4.44 m，当前传感器只能覆盖 67.5%。运行
`analyze_spatial_preview_speed()` 可生成车速—预瞄距离表和图。

当前鲁棒项使用道路高度和空间斜率的盒式误差上界收紧预测行程约束，最大收紧
比例限制为 25%。这是约束鲁棒化，不是完整的管束 MPC。

## CarSim 接口

工程复用：

```text
E:\Desktop\workspace\CarSim_database\carsim_active_suspension
```

当前 `Run_all.par` 同时包含 I/O 数据集和 Additional Data 命令，因此四个导入量重复为 8 路、39 个导出量重复为 77 路。适配器会向两组导入口发送相同命令，并读取后 39 路完整输出。若在 CarSim GUI 中消除重复配置，需要同步修改 `write_carsim_simfile.m`、S-Function 端口和运行脚本。

CarSim 的 `Pitch/AVy` 正方向与半车模型“车头上仰为正”的约定相反，状态重构时
必须乘以 `-1`。这是本轮联合仿真中影响最大的接口修正；力接口极性经正负两组
CarSim 实测确认应取 `+1`。

CarSim 内部参数文件的 `tstep` 与 simfile 中的 `EXT_MODEL_STEP` 必须保持一致。
运行脚本会生成本地参数副本并同步两处步长，避免 CarSim 物理距离与 Simulink
时间漂移。默认积分步长现为 `1 ms`：与 `0.5 ms` 参考结果的加速度 RMS 相差约
1%，积分次数减半；同步后的 `2 ms` 设置会使当前非线性车辆模型数值失稳。

## 运行

仓库已包含 `carsim_half_car_qp_mpc.slx`。运行前用 `require_carsim_model` 检查模型是否存在；
项目没有可用的自动模型构建器。原空构建文件已移除，缺少模型时需先从仓库恢复 `.slx`，
不会以空操作冒充构建成功。

```matlab
cd('E:\Desktop\workspace\suspension_control_all\carsim_mpc_quick');
addpath('src', 'tests');
validate_quick_mpc_offline();
tuning = tune_quick_mpc_offline(); % 权重筛选与 N=10/15/20/30 扫描
tuning_carsim = tune_quick_mpc_carsim(); % CarSim 在环权重与力极性筛选
require_carsim_model(quick_mpc_config()); % 检查仓库中已有的 .slx 模型
report = run_carsim_mpc_demo(); % 默认 deployment scale = 1.00
horizon = run_carsim_mpc_horizon_sweep(); % CarSim 中复核 N=10/15/20/30

% 统一运行坑包、CarSim B 级和 C 级道路的被动/MPC 对照实验
suite = run_carsim_mpc_road_suite();

% 50 ms 判定、100 ms 最小保持、30 ms 响应的一阶持续主动力
suite_hold = run_carsim_mpc_road_suite([], 'persistent_force');

% 五档 CDC 半主动部署；该模式目前只用于工程约束评估
suite_cdc = run_carsim_mpc_road_suite([], 'cdc');

% C 级道路的空间噪声、偏置、短感知距离和鲁棒收紧对照
robustness = run_carsim_mpc_robustness_suite();

% 不同车速下的所需预瞄距离和传感器覆盖率
speed_map = analyze_spatial_preview_speed();

% 35/60/80 km/h 下，坑包、B 级、C 级道路的被动/MPC 对照
speed_suite = run_carsim_mpc_speed_suite([35 60 80]);
```

单道路结果写入 `results/carsim_mpc_quick/`；多道路结果及图表写入
`results/carsim_mpc_quick/road_suite/`。坑包工况由 50 mm 凸包和 50 mm
凹坑组成，B/C 级路面直接读取当前 CarSim 数据库中的 `Class B #1` 与
`Class C #1`，三类工况均以 35 km/h 运行 12 s，并忽略前 0.5 s 初始化段。

当前正式结果相对线性被动悬架如下：

| 道路 | 车身加速度 RMS | 俯仰加速度 RMS | 前/后最大行程 | 轮胎动载 RMS |
|---|---:|---:|---:|---:|
| 坑包 | -15.29% | -13.43% | -2.90% / -4.83% | +6.49% |
| B 级 | -32.87% | -40.90% | -24.75% / -28.33% | +25.20% |
| C 级 | -25.13% | -31.12% | -16.59% / -25.06% | +1.70% |

负号表示相对被动基线降低。坑包工况的角点总力达到 `5000 N` 饱和边界；B 级
道路的轮胎动载代价也较明显，因此当前结果是偏舒适性的权衡，不是所有性能的
一致改善。

C 级道路加入 `2 mm` 空间噪声和 `1 mm` 偏置后，车身加速度 RMS 改善率为
24.82%；感知距离缩短到 `1.5 m`，并加入 `5 mm` 噪声与 `2 mm` 偏置后，该指标
仍为 23.47%。开启和关闭约束收紧在中等误差下得到相同车辆响应，说明 80 mm
行程约束当时未激活；不能把误差工况下的性能保持全部归因于鲁棒收紧。

持续主动力版本在 B/C 级道路仍有 8.91%/4.86% 的车身 RMS 改善。五档 CDC
映射则分别恶化 4.29%/3.26%，说明主动轴力 QP 不能通过事后投影直接变成有效
的 CDC 控制器。后续应以阻尼系数或离散档位为决策变量重新设计半主动 MPC。

多车速实验按前轮道路坐标 `10--110 m` 的统一空间区间统计，不使用相同仿真
时长，以免高速组驶出道路数据范围后混入平路尾段。CarSim 实际平均车速为
34.94、59.93 和 79.93 km/h。主要结果如下，正值表示相对线性被动悬架降低：

| 道路 | 车速 (km/h) | 车身加速度 RMS | 俯仰加速度 RMS | 最小预瞄覆盖率 |
|---|---:|---:|---:|---:|
| 坑包 | 35 / 60 / 80 | 15.29% / 16.43% / 15.35% | 13.43% / 6.59% / 13.63% | 100.0% / 90.0% / 67.5% |
| B 级 | 35 / 60 / 80 | 33.24% / 38.14% / 36.14% | 41.22% / 26.30% / 24.02% | 100.0% / 89.9% / 67.5% |
| C 级 | 35 / 60 / 80 | 25.21% / 30.39% / 24.91% | 31.45% / 18.57% / 13.16% | 100.0% / 89.8% / 67.4% |

三种道路、三种车速下，MPC 均降低了车身和俯仰加速度 RMS；但固定 3 m
传感器在 60/80 km/h 时无法覆盖完整的 0.20 s 预测域。B/C 级道路的俯仰改善
随车速升高而下降，坑包工况在 60/80 km/h 时前后最大行程反而增加，说明现有
权重和预瞄距离不能按低速标定直接外推。汇总文件位于
`results/carsim_mpc_quick/speed_suite/`。

## 方法边界

- MPC 内部是半车模型，只控制升沉和俯仰，不主动控制侧倾。
- 左右轮获得相同的轴向增量，不适合左右路面高度明显不同的工况。
- 当前结果覆盖 35/60/80 km/h，但每个组合仍是单次确定性运行；感知误差试验也只有固定随机种子，不能代替重复试验与统计检验。
- 舒适性和悬架行程改善的同时，轮胎动载有所增加；后续需要调整轮胎挠度权重或加入软约束。
- CDC 版本目前是主动 MPC 命令的工程映射，不是经过独立优化的半主动 MPC。
- 论文正式实验前应辨识 CarSim 降阶模型、增加 Kalman 状态估计，并进行更密集的车速扫描、多参数和更强基线验证。
