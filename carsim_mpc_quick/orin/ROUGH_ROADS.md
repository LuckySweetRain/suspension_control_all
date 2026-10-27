# 波浪路、减速带与凹陷测试路

本入口使用 CARLA 0.9.11 的 OpenDRIVE standalone 功能，在当前 Windows 运行包中生成真实的
道路网格，无需先搭建 UE 编辑器工程。它支持本次纵向高程试验，不等于能够动态生成任意三维
越野网格或松散砂石。官方机制见：
[OpenDRIVE standalone mode 0.9.11](https://carla.readthedocs.io/en/0.9.11/adv_opendrive/)。

## 直接运行

从项目根目录运行，保持 Orin 直连、55002 控制服务及 CarSim 可用：

```powershell
# 三种起伏连续排列；默认 10 s，真实相机、主动力、加速度及前后轴预瞄路高曲线。
./carsim_mpc_quick/orin/start_environment_loop.ps1 -RoadProfile combined

# 各自独立的路面
./carsim_mpc_quick/orin/start_environment_loop.ps1 -RoadProfile wave
./carsim_mpc_quick/orin/start_environment_loop.ps1 -RoadProfile bump
./carsim_mpc_quick/orin/start_environment_loop.ps1 -RoadProfile pothole

# 无窗口测试
./carsim_mpc_quick/orin/start_environment_loop.ps1 -RoadProfile combined -NoVisualize

# 返回原 Town03 验证路线
./carsim_mpc_quick/orin/start_environment_loop.ps1 -RoadProfile town
```

生成路面会替换当前 CARLA 世界；有其他车辆时入口拒绝替换。完成后保留结果窗口，关闭窗口
即可退出启动命令。生成路保持在 OpenDriveMap，`town` 模式会将其切回 Town03。
本模式忽略城市地图的 SpawnIndex，固定从道路 s=10 m 起步。道路长 160 m、宽 4 m，采用原来
约 35 km/h 的车辆配置，运行时长限制为 13 s，默认 10 s 足以使前后轴经过全部起伏。

## 第一版参数

位置采用道路纵向坐标，车辆起点为 s=10 m。

| 路面 | 位置 | 高度与长度 |
|---|---|---|
| 连续波浪路 | s=25～50 m | 振幅 ±15 mm，波长 5 m，两端平滑收敛 |
| 宽缓减速带 | s=60～64 m | 高 30 mm、长 4 m，升余弦轮廓 |
| 宽缓凹陷 | s=85～89 m | 深 25 mm、长 4 m，负升余弦轮廓 |

三者都横向贯穿车道，左右轮同相，适配当前半车模型。`pothole` 是宽缓凹陷，不是局部圆坑，
尚未覆盖单轮落坑、左右轮迹差异或整车侧倾控制。图中纵向剖面会夸张垂直比例；道路起伏为厘米
级，从追尾相机上可能不明显，可结合窗口中的毫米级路高曲线观察。

## 闭环中的真实数据来源

`rough_road.py` 将高程写入分段三次多项式 OpenDRIVE，采样结点间隔 0.1 m，CARLA 网格
纵向顶点间距 0.05 m。设计高程只用于创建网格和独立验收：

1. CarSim 轮胎地形高度、坡度来自对 CARLA 网格的射线查询。
2. Orin 接收该网格渲染的深度图，自行反投影、保留空间历史、生成前后轴预瞄。
3. 客户端不向 Orin 发送未来设计高程；只传相机标定、已知道路 XY 边界和初始状态参考标定。
4. MPC 仍采用原 ±2000 N 轴力限幅及权重，不因新路面放宽约束。

0.9.11 的运行时道路网格射线标签为 NONE，语义相机没有标准 Road 标签。因此仅在本入口
创建的专用道路世界里，使用已知 XY 车道范围筛选未分类网格的有效深度点；这不是训练好的
路面识别网络。地形回调也只在该道路边界内接受 NONE。城市地图继续使用原语义真值掩膜，
仍拒绝未分类接触。专用模式不允许再添加 `--obstacle-at`，也不声明提供底盘碰撞求解。

## 验证结果（2026-10-27）

四组测试全部完成，每组 10 s、1000 控制步、200 次图像更新；两轴均经过各自全部特征。
总报告：`results/carla_rough_suite/20261027_195501/report.json`。

| 工况 | 工件目录（results/carla_environment 下） | 深度预瞄与设计高程最大差 |
|---|---|---:|
| wave | 20261027_195501 | 0.460 mm |
| bump | 20261027_195529 | 0.501 mm |
| pothole | 20261027_195557 | 0.408 mm |
| combined | 20261027_195622 | 0.573 mm |

每组核对 38000 个前后轴预测高度样本，并按实际输入时刻对齐状态；设计函数仅在离线校验器
使用。每组另有 600 个独立网格射线点，组合网格最大误差约 0.00035 mm。特征附近保存
`approach_wave`、`approach_bump`、`approach_pothole` 相机包和截图，并用独立场景射线核对
20 个重建点，最大误差小于 0.07 mm。上述精度限于这些理想仿真样本，不是实物相机保证。

`road_mesh_check.json` 保存网格校验，`road_preview_check.json` 保存预瞄校验，`road.xodr`
和 `road_specification.json` 保存几何输入，完整状态与动作保留在 `trace.json`。
剖面图为 `results/carla_rough_suite/20261027_195501/road_profiles.png`。

可重复运行全部验收：

```powershell
conda run --no-capture-output -n carla0911 python carsim_mpc_quick/orin/validate_rough_road_suite.py
```

Town03 回归 1 s 已通过（`20261027_195850`，预热后实际启用 MPC）。主项目
`scripts/check_project.py` 七组通过，报告为
`results/project_checks/20261027_195514_304742/report.json`。代码检查包括主项目 Ruff 和独立
Python 3.7 目标的子项目 Ruff 检查，未使用 pytest。

## 当前边界

这仍是离线同步试验；图像更新峰值不满足 10 ms 墙钟实时。波浪激励下前轴存在短时正负限幅，
保留在报告中，不能以接口通过代替控制性能验收。尚未完成被动悬架对照、不同速度和更强路面
的性能评估，也不包含轮胎离地与碰撞的全面验证。

曾遇到一次 Town03 切回 OpenDriveMap 时 RPC 超时，已重启本次 CARLA 服务并恢复运行。
完成窗口现在先释放 CARLA 客户端再保留图像，避免已完成窗口继续持有旧世界引用。若服务器
端口存在但 RPC 不响应，应先检查并重启该 CARLA 实例；不要把连接超时当作试验通过。
