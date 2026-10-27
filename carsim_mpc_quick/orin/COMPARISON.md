# 左右同步对照：Orin MPC 与被动悬架

从项目根目录运行：

```powershell
# 默认组合路面，左侧 MPC，右侧被动，10 s。
./carsim_mpc_quick/orin/start_environment_loop.ps1 -Compare

# 只比较一种起伏
./carsim_mpc_quick/orin/start_environment_loop.ps1 -Compare -RoadProfile wave
./carsim_mpc_quick/orin/start_environment_loop.ps1 -Compare -RoadProfile bump
./carsim_mpc_quick/orin/start_environment_loop.ps1 -Compare -RoadProfile pothole

# 无窗口复核
./carsim_mpc_quick/orin/start_environment_loop.ps1 -Compare -NoVisualize
```

目前对照入口支持生成的试验路面，默认将 RoadProfile 的默认值 town 选择为 combined。
起伏参数与 [ROUGH_ROADS.md](ROUGH_ROADS.md) 相同。运行会替换当前世界，存在其他车辆时拒绝
替换。完成后保留对照窗口；关闭窗口后可再次启动。当前范围为 0.05～13 s，默认 10 s 能经过
全部三段起伏。

## 比较方式

- 两个独立 Windows 子进程各自加载、初始化和积分一份 CarSim DLL。没有在同一个 DLL 全局
  状态里交替冒充两辆车，也没有让两侧共用同一条车辆轨迹。
- 道路为同一高程定义的两个平行副本，中心线 y=2 m 和 y=10 m；各自用场景射线查询轮胎地形。
  两条道路各用 600 个射线点独立核验。左右车辆初始状态和完整运行参数文件必须相同。
- 左侧状态和真实深度图送到 Orin 55002 上的 MPC。右侧主动轴力为零，仍保留原弹簧以及
  相同的被动阻尼。右侧每个角点阻尼命令为 `3 × CmpD` N，其中 CmpD 单位为 mm/s，
  角点限幅 5000 N；具体系数从 Orin 所加载 MAT 的配置返回，而非凭显示标签猜测。
- 两个进程收到同一步指令后分别推进 10 个 1 ms 积分步，主进程等两侧完成后才进入下一步。
  只有主进程每 5 个控制步 tick 一次 CARLA；两侧 RGB 和左侧深度/语义相机均取同一帧号。
- RGB 是两个实际相机视图。两侧都采用固定世界高度 3 m 跟拍，水平视场 65°、俯角 20°；
  相机只随各自车辆水平移动，不跟随车身升沉、俯仰或侧倾，也没有放大车身运动。
- 左侧前 0.5 s 的深度预瞄覆盖建立期间使用被动监督模式。本次此阶段左右状态逐样本完全
  一致；随后仅左侧启用 MPC。右侧的零主动力不代表总悬架力或阻尼力为零。

窗口包含左右画面、主动力、垂向加速度、轮跳压缩量、俯仰和累计 RMS。两侧加速度与俯仰
曲线使用各自共同的纵轴尺度；第三张图为左侧前后主动轴力。
画面驱动的是 CarSim 车身位姿，尚未单独驱动 CARLA 车轮骨骼的悬架压缩动画；轮跳以
CarSim 数值为准，不能由可视化网格的车轮位置推断轮胎接触状态。

## 已验证结果（2026-10-27）

2 s 冒烟检查：`results/carla_comparison/20261027_201847`。
10 s 无窗口：`results/carla_comparison/20261027_202117`。
10 s 可视化：`results/carla_comparison/20261027_203229`（最终跟拍视角）。

本次组合路面 10 s，1000 控制步、200 次双视图更新：

| 指标 | MPC 主动悬架 | 被动悬架 |
|---|---:|---:|
| 车身垂向加速度 RMS | 0.317925 m/s² | 1.609573 m/s² |
| 车身垂向加速度绝对峰值 | 1.038644 m/s² | 4.951069 m/s² |

本次 RMS 相对变化为 −80.25%。这里的 RMS 是全部 1000 个 10 ms 样本的未加权统计，包含
预热区间，不是 ISO 频率加权舒适性指标，也不能推广为真实车辆、其他车速或路面的效果保证。
当前执行器是原 ideal_force 配置，仍是离线同步比较，不保证 10 ms 墙钟实时。

两侧目标车速均为原 35 km/h；两条轨迹按各自动力学自然推进，不强行覆盖位置或速度。
本次实际车速最大差约 0.0603 km/h。参数文件 SHA256 相同，初始状态最大差为零，前 50 个
被动预热步的状态差为零。右侧全部 1000 步的阻尼力已与现有 MATLAB 对照控制器的被动分支
独立核对，误差为零；两个求解器时钟、命令与保存状态逐样本对齐检查通过。

工件：`report.json`、双侧主记录 `trace.json`、`active/trace.json`、`passive/trace.json`、
两侧 `initial.json` 与 CarSim 输出文件、`first_pair.png` 和 `last_pair.png`。
`fairness_check.json` 保存独立核验结论。可重复核验：

```powershell
conda run -n suspension-control python carsim_mpc_quick/orin/verify_comparison.py results/carla_comparison/20261027_203229
```

本轮主项目 `scripts/check_project.py` 七组通过，报告
`results/project_checks/20261027_202149_619989/report.json`。另执行主项目 Ruff、修改子项目
的独立 Python 3.7 Ruff、格式检查及启动器的 PowerShell 语法检查，没有运行 pytest。
