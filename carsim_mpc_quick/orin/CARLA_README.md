# CARLA–CarSim–Orin 基础三端闭环

2026-10-27 已在本机实测 CARLA 0.9.11 场景、原生 CarSim 插件和 Python 主控三端闭环。
本入口不需要 MATLAB。CarSim 2020 Browser/许可管理器需要保持运行。

## 已验证的架构

```text
Python 主时钟（本机）
  ├─ CarSim DLL：每 1 ms 推进车辆动力学
  ├─ 每 10 ms：CarSim 77 路状态 → 直连 TCP → Orin MPC → 8 路悬架力
  └─ 每 50 ms：CarSim 姿态 → CARLA 中关闭物理的车辆 → 场景/相机
```

该路线由 Python 独占推进 CarSim DLL，三端模式不调用 `vehicle.enable_carsim`。
避免原生插件和 Python 各自维护一份车辆动力学。原生插件路线单独测试和使用。

**当前边界：CARLA 提供场景显示和相机采样，轮胎接触/路面激励仍来自 CarSim 数据集。**
相机道路感知尚未连接 Orin，Orin MPC 预瞄仍读取已部署的 CarSim 路面图。
CARLA 障碍物、碰撞和地形尚不反馈给这一条 Python 主控动力学路线；不能把当前演示当作
“CARLA 地形驱动轮胎接触”的完整闭环，也不能据此验证视觉预瞄控制性能。
车身姿态采用显式坐标反射和初始对齐，尚需用转弯/坡道/侧倾专用工况标定。

## 环境与启动

- CARLA：`E:/Desktop/workspace/CARLA_0.9.11/WindowsNoEditor`，服务端口 `127.0.0.1:2000`。
- 客户端与服务器均实测为 `0.9.11`；随包 egg 是 `cp37-win-amd64`。
- 已创建 Conda 环境 `carla0911`（Python 3.7.12）；仅用于这套旧 CARLA 前端。
  本项目主环境与 Orin 环境不降级。新机器可使用同目录 `environment_carla0911.yml` 创建环境。
- 原生 VehicleSim 插件描述版本 `2020.0.3`，目标 Unreal Engine 4.24。
- Orin：`192.168.50.2:55001`，沿用已部署的 `suspension-mpc.service`。

首次启动 CARLA，可在 PowerShell 执行：

```powershell
$carlaRoot = 'E:/Desktop/workspace/CARLA_0.9.11/WindowsNoEditor'
Start-Process -FilePath "$carlaRoot/CarlaUE4.exe" -WorkingDirectory $carlaRoot -WindowStyle Hidden -ArgumentList @('-carla-rpc-port=2000','-quality-level=Low','-ResX=800','-ResY=600','-windowed','-nosound','-RenderOffScreen')
```

这是后台渲染模式，相机图像保存为工件。若服务器已经监听 2000 端口，不要重复启动。
当前进程保持运行，测试脚本结束时会销毁自己创建的车辆、相机，并恢复原世界同步设置。
运行脚本期间不要启动其他 CARLA 主时钟；脚本检测到已有车辆会主动退出。

在仓库根目录执行：

```powershell
# 仅测试 CARLA 场景和 RGB 相机（默认 CARLA 物理）
conda run -n carla0911 python carsim_mpc_quick/orin/carla_stack.py --mode scene --seconds 2

# 单独测试原生 CARLA–CarSim 插件（油门控制，使用 CARLA 地形）
conda run -n carla0911 python carsim_mpc_quick/orin/carla_stack.py --mode plugin --seconds 2

# Python 统一推进的 CARLA + CarSim + Orin 悬架控制闭环
conda run -n carla0911 python carsim_mpc_quick/orin/carla_stack.py --mode triple --seconds 12
```

必须使用 `conda run` 或正确激活环境，以加载 Conda 的 DLL 搜索路径。
测试脚本在启动时校验客户端/服务器版本、CarSim 的 8 进口/81 出口/1 ms 步长、
Orin MAT 数据哈希和每步响应序号；控制输出必须有限且不超过角点 5000 N 边界。
追加的四个 CarSim 出口是 Xcg_TM、Ycg_TM、Yaw、Roll；前 77 路保持原控制器协议。

## 实测结果

| 测试 | 仿真时长 | 相机帧 | Orin 控制次数 | 结果 |
|---|---:|---:|---:|---|
| CARLA 场景 | 2 s | 40 | 0 | 通过 |
| 原生 CARLA–CarSim 插件 | 2 s | 40 | 0 | 通过，车辆位移约 5.80 m |
| Python 主控三端闭环 | 2 s | 40 | 200 | 通过，无回退 |
| Python 主控三端闭环 | 12 s | 240 | 1200 | 通过，无回退 |

12 s 工况车辆位移约 116.34 m，最大归一化 QP 约束误差 `0.000859449`。
最大 Orin 请求往返耗时约 69.12 ms，因此仍是仿真时间同步，尚未满足逐步 10 ms 墙钟实时。
该次 1200 步请求中位数约 7.27 ms，95 分位约 16.07 ms；每步都有非零主动轴力输出。
使用的仍是之前在 CarSim–Orin 上验证过的 3200 次 ADMM 部署预算和已知路面工况。
原生插件的 CarSim 日志确认推进到 2.0005 s，并生成独立结果文件；日志包含轮胎表外插提示，
所以插件测试结论限于接口和推进功能正常，不代表该历史车辆/地形组合已完成物理标定。

证据均在 `results/carla_stack/`：

- `scene_20261027_163619/report.json`
- `plugin_20261027_163644/report.json`
- `triple_20261027_163755/report.json`
- `triple_20261027_163854/report.json`、`trace.json`、`first.png`、`last.png`

每次运行使用独立目录，包含生成的 CarSim 参数、simfile、求解器日志和图像。
历史原生插件 simfile 只作为输入模板，源数据库和原文件不改写。
首次相机检查因传感器采样周期边界漏帧失败；改为每个世界 tick 采样后，40/240 帧严格配对通过。
首次三端检查因新 EXPORT 写在 PAR 的 END 之后未生效而主动退出；修正后校验到 81 路并通过。

## 下一阶段

若目标是视觉道路预瞄实验，下一步应先统一 CARLA 可见路面和 CarSim 轮胎激励的几何来源，
校准坐标、道路高度、车速和传感器时间戳，再把相机/LiDAR 估计出的路面预瞄接入 Orin。
若坚持原生插件负责地形接触，则需补充插件/服务端的悬架 IMPORT/EXPORT 批量读写与同步接口；
当前打包环境未找到对应 C++ 源码，不能仅修改 `manual_control_carsim.py` 实现这些能力。

官方原生集成说明：<https://carla.readthedocs.io/en/0.9.11/carsim_integration/>。
