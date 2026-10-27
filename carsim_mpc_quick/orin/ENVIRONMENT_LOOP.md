# CARLA 环境反馈闭环：当前实现与未完成项

本入口取代“只把 CarSim 姿态显示到 CARLA”的演示路线，用真实 CARLA 场景地形和相机数据驱动
CarSim–Orin。**完整的原生多方向碰撞冲量耦合尚未完成**，不能把本原型等同于通用碰撞仿真。
旧 `carla_stack.py --mode triple` 保留用于对照，它仍使用原 CarSim 路面。

## 数据链路

1. 本机 Python 独占 CarSim DLL，以 1 ms 积分。
2. 官方 `vs_road_install_terrain_function` 回调将轮胎位置变换到 CARLA，使用真实场景射线交点
   返回地面高度、坡度。新运行参数中移除原 C 级路面定义，不混合两套路面。
3. CARLA RGB、深度、语义相机以 20 Hz 同帧采集。深度/语义原始图像压缩后通过直连发送到 Orin。
4. Orin 重建道路点云和空间预瞄，再以 100 Hz 仿真采样运行 MPC；不调用原 MAT 中的静态路面查询。
5. 正向接触原型查询 CARLA 场景几何，将柔顺接触反力通过 `IMP_FX_AERO Add` 加到 CarSim 的
   气动力参考点。接触后 Orin 监督器保持被动悬架。它不是原生 PhysX 冲量，也没有完整的侧向、
   车车碰撞、接触力矩、变形、恢复系数与材料辨识。

地形回调遵循本机官方 `E:/Carsim2020.0_Prog/Help/Memos/VS_Terrain.pdf` 的签名与单位。
外力变量和单位通过 `vs_write_doc_files` 从当前车辆模型生成并核对。
摩擦系数目前固定为 0.85、滚动阻力为 0.01，未接入 CARLA 材料/摩擦触发器映射。
场景高度使用 5 cm 缓存栅格，尚未验证动态地形、摩擦变化和复杂路沿接触。

## 图像预瞄的性质

预瞄真正来自深度图像反投影，但路面掩码使用 CARLA 语义相机的仿真真值，**不是训练好的 RGB
路面识别网络**。RGB 相机供显示/记录；没有把 RGB 文件存在磁盘就宣称其参与控制。
感知没有使用 CARLA 射线高度：射线只用于动力学接触和独立几何校验。

采用内参、相机外参与里程坐标，保存有限点云历史覆盖车底遮挡区域。
起始近场道路拟合切平面；状态和预瞄相对同一个切平面转换，避免将整段坡度当作路面突变。
没有观测的近场为边界外推，并增加不确定度；观测缺失或图像年龄超过 0.15 s 时拒绝该步，
不回退到原静态路面继续冒充视觉闭环。观测区域高度误差参数为 15 mm，遮挡区为 30 mm；
这些是当前工程设置，不是经过统计认证的误差上界。

初始化前用独立的原模型配置取得 CG/道路路径原点偏移，并立即终止该标定实例，不推进车辆。
随后才安装 CARLA 地形并初始化正式求解器。若在初始化后才改坐标原点，会使坡面瞬间平移到
已落地轮胎下方，已实测可造成非物理冲击，故正式入口使用先标定、后初始化的顺序。

## 运行

保持 CARLA 0.9.11 的 2000 端口、CarSim Browser 许可、本机–Orin 直连正常。
Orin 新服务为 `suspension-environment.service`，监听 55002；旧 55001 服务保留。
`deploy_orin.py` 已包含新服务和代码的可重复部署，不在配置中保存密码。

```powershell
# 无障碍物的短坡道验证；显示真实画面、控制力和加速度
conda run --no-capture-output -n carla0911 python carsim_mpc_quick/orin/run_environment_loop.py --seconds 0.5 --spawn 53 --visualize

# 正向静态障碍物接触反馈原型：前方 6 m 放置 streetbarrier
conda run --no-capture-output -n carla0911 python carsim_mpc_quick/orin/run_environment_loop.py --seconds 1.2 --obstacle-at 6
```

当前拒绝已有其他车辆的世界，防止更改别人的仿真，同时限制了多车辆验证。
碰撞射线排除自车车辆标签；未分类的 `NONE` 接触会主动中止，不冒充已知接触面施加强制动力。
接触力模型使用 1e6 N/m 弹簧、8e4 Ns/m 阻尼，最大反力 100 kN，且按关闭速度与 1680.2 kg
等效车辆质量限制单步冲量。相关参数尚未进行碰撞实车/高保真模型标定。

所有工件进入 `results/carla_environment/<时间>/`，包含逐步状态、指令、相机帧号、MPC 预瞄、
接触反力、首帧原始深度/语义图及校准包，失败记录也保留。当前是离线同步联合仿真，不保证
10 ms 墙钟实时。`passed` 仅表示该次脚本的数值与接口检查通过，不表示全面物理验收。

## 已执行的验证（2026-10-27）

| 验证 | 结果与证据目录 |
|---|---|
| 平路地形与图像闭环，0.2 s | 20 步、3430 次地形回调、4 帧更新，无接触；`20261027_172403` |
| 坡道名义工况，0.5 s | 50 步、8530 次回调、10 帧图像更新，无接触；`20261027_173659`、可视化 `20261027_173933` |
| 图像几何独立校验 | 20 个道路点，最大高度误差 9.67 mm，平均 5.70 mm；`20261027_173659/camera_geometry_check.json` |
| 控制输入依赖 | 同一状态下，真实图像预瞄与零预瞄的最大角点力差约 0.214 N；禁止静态查询的断言通过；`20261027_173933/input_dependency_check.json` |
| 过期图像 | 0.2 s 旧图像被拒绝，不使用静态路面兜底 |
| 障碍物，1.2 s | 0.34 s 发现接触；34.68 km/h 降到约 0.194 km/h，反力确实进入 CarSim；最终回归 `20261027_175823` |
| 较长坡道，2 s | 脚本推进完成但遇到 `NONE` 接触，不能计为名义坡道/通用碰撞验收；`20261027_175120`。现代码改为遇到该类接触中止 |
| 原控制器回归 | 本机与 Orin 均保留 80 步 MATLAB 对照，力误差约 1e-12 N，复位/非有限输入/不可行回退通过 |
| 主项目回归 | 七组检查通过；最终报告 `results/project_checks/20261027_175832_512653/report.json` |

几何验证和输入依赖可重复执行：

```powershell
conda run -n carla0911 python carsim_mpc_quick/orin/verify_camera_geometry.py results/carla_environment/20261027_173659
python carsim_mpc_quick/orin/verify_environment_inputs.py results/carla_environment/20261027_173933
```

## 完整碰撞仍缺的构建条件

用户已确认没有本机源码工程。目前本机只有运行包，未找到 UE4.24 编译环境及 VehicleSim 插件源码。
公共 CARLA 0.9.11 源码已定位；该版本的碰撞处理在
[CarlaWheeledVehicle.cpp](https://github.com/carla-simulator/carla/blob/0.9.11/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Vehicle/CarlaWheeledVehicle.cpp)
中，`OnCarSimHit` 和 `OnCarSimOverlap` 是待实现函数。较新版本的自动切回物理行为不能直接套用。
因此前面关于当前版本会自动切回默认物理的表述按此版本核对结果更正。

完整原生接入需要：取得匹配版本源码与 UE4.24/VehicleSim 插件构建条件，在物理子步提取接触点、
法向/切向冲量、力矩，变换到 CarSim 坐标并进行守恒的双向交换，扩展 RPC 的 IMPORT/EXPORT
与状态同步，重新编译服务器和匹配 Python 客户端，再做坡道、路沿、侧碰、车车碰撞与能量审计。
当前不能通过改 Python 标志位替代这一步，也不应取消可行性和有限值检查来让失败场景继续运行。
