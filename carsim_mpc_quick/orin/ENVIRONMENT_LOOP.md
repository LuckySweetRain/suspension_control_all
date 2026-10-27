# CARLA 环境反馈闭环：当前实现与未完成项

本入口取代“只把 CarSim 姿态显示到 CARLA”的演示路线，用真实 CARLA 场景地形和相机数据驱动
CarSim–Orin。**完整的原生多方向碰撞冲量耦合尚未完成**，不能把本原型等同于通用碰撞仿真。
旧 `carla_stack.py --mode triple` 保留用于对照，它仍使用原 CarSim 路面。

**2026-10-27 晚已修复主动轴力长期负饱和。** 原先的“通过”主要是接口、有限值和约束检查，
没有证明控制行为合理。下文早期测试保留为历史记录；当前物理坐标与控制输出以“负饱和修复”
一节的验证为准，不能把早期厘米级相机误差视为可以接受的已完成标定。

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
未来路面预瞄没有使用 CARLA 射线高度；深度反投影使用像素中心 `(u+0.5,v+0.5)`。
另有一次性的动力学初始化接触高度标定，用来对齐 CarSim 相对初始平衡状态和相机绝对路面。
因此这是带初始动力学标定的深度预瞄，不是完全不依赖外部标定的纯视觉状态估计。

采用内参、相机外参与里程坐标，以 5 cm × 10 cm 地面单元保存空间历史，覆盖车底遮挡区域；
不再简单截取最近 8 万个像素，避免重复近场像素挤掉后轮尚未到达的地面。
起始近场道路拟合切平面；前后轴高度、车身升沉和俯仰及其速度一起转换到该平面坐标。
前后轴分别检查预瞄覆盖率；覆盖未建立时主动轴力为零、保留被动阻尼，记录 `preview_warmup`，
不把外推路面作为已观测预瞄启用 MPC。图像缺失或年龄超过 0.15 s 时拒绝该步，
不回退到原静态路面继续冒充视觉闭环。观测区域高度误差参数为 15 mm，遮挡区为 30 mm；
这些是当前工程设置，不是经过统计认证的误差上界。

协议 v2 的第一帧图像现在必须带 `initial_road_heights_m=[前轴初始高度,后轴初始高度]`，
由同一 CarSim 动力学地形在初始化轮胎位置取得，后续预瞄只来自相机。缺少标定会拒绝连接，
旧 55001/v1 MATLAB 对照链路不受影响。新客户端与 Orin `environment_server.py`、
`depth_preview.py` 必须一起更新，`deploy_orin.py` 已包含这些文件。

## 主动力长期负饱和修复与验证

定位到三个共同影响控制的缺陷：

1. 深度反投影漏掉半像素中心偏移，路面高度系统性偏高。独立 40 点射线对比中，旧方法
   平均偏高约 9.63 mm、最大 19.00 mm；修复后最大约 0.063 mm。
2. 初始切平面从未来道路拟合，和前后轴各自的初始接触高度不同。原先只减去行驶距离产生的
   升沉，漏掉常量和俯仰参考项，形成虚假轮胎变形。现在按初始接触高度构造完整的刚性参考
   变换，并验证悬架行程、轮胎变形和模型输出在变换前后不变。
3. 点云按最近像素截断导致后轴预瞄丢失，且后轴错误复用了前轴覆盖标志。改为保留空间单元，
   独立记录两轴覆盖并在建立覆盖前保持被动模式。本次默认路线预热为前 0.50 s。

力限幅 ±2000 N、MPC 权重和 MAT 参数均未改变。相同 Town03/159 路段、10 s 对照：

| 主动轴力接近负限幅（≤−1999 N）的步数比例 | 修复前 | 修复后 |
|---|---:|---:|
| 前轴 | 69.5% | 0% |
| 后轴 | 57.7% | 0.1% |

修复前为 `20261027_190919`，修复后为 `20261027_192825`，可视化复测
`20261027_193016` 重现相同结果，均保留全部轨迹。对比图和 MAT 哈希一致性记录在
`results/force_saturation_diagnosis/active_force_comparison.png`、`comparison.json`。
这证明本工况的持续负饱和消除，不代表所有场景都不会触及执行器限幅，也不是控制性能全面验收。

补充验证：

- 修复后 20 个相机点与独立射线最大高度差约 0.056 mm，几何检查门槛由 30 mm 收紧到 2 mm。
- `verify_reference_frame.py` 的模型输出变换误差小于 1e-14；旧变换在该样本出现约 44 mm
  等效轮胎变形误差，新增空间历史和缺少初始化标定的拒绝检查通过。
- 1.2 s 障碍物 `20261027_193233`、0.8 s 坡道 `20261027_193240` 均通过；前者接触时
  使用被动监督模式，不能据此验证主动控制碰撞性能。
- 主项目 `scripts/check_project.py` 七组通过，报告
  `results/project_checks/20261027_192623_730609/report.json`；修改的 Python 文件单独运行
  Ruff 检查和格式检查，使用 Python 3.7 目标以兼容 CARLA ABI。

空间历史去重增加图像更新开销；修复后 Orin 计算 p95 仍约 65 ms，不满足 10 ms 硬实时。
当前几何抽查精度也不等于真实相机、动态遮挡和所有路面的误差保证。

```powershell
conda run -n carla0911 python carsim_mpc_quick/orin/verify_camera_geometry.py results/carla_environment/20261027_192825
conda run -n suspension-control python carsim_mpc_quick/orin/verify_reference_frame.py results/carla_environment/20261027_192825
```

初始化前用独立的原模型配置取得 CG/道路路径原点偏移，并立即终止该标定实例，不推进车辆。
随后才安装 CARLA 地形并初始化正式求解器。若在初始化后才改坐标原点，会使坡面瞬间平移到
已落地轮胎下方，已实测可造成非物理冲击，故正式入口使用先标定、后初始化的顺序。

## 运行

保持 CARLA 0.9.11 的 2000 端口、CarSim Browser 许可、本机–Orin 直连正常。
Orin 新服务为 `suspension-environment.service`，监听 55002；旧 55001 服务保留。
`deploy_orin.py` 已包含新服务和代码的可重复部署，不在配置中保存密码。

```powershell
# 一键检查直连与服务，必要时启动 CARLA，显示相机与控制曲线。
# 默认 Town03 的 159 号起点、10 s 仿真时间；完成后保留窗口，关闭窗口即可退出。
./carsim_mpc_quick/orin/start_environment_loop.ps1

# 不显示窗口，用于批量复核；仍然使用 CARLA 真实相机图像。
./carsim_mpc_quick/orin/start_environment_loop.ps1 -NoVisualize

# 只验证 Python、Pillow DLL 和 Tk 渲染，不连接 Orin、不改变 CARLA 世界。
./carsim_mpc_quick/orin/start_environment_loop.ps1 -CheckOnly

# 无障碍物的短坡道验证；显示真实画面、控制力和加速度
conda run --no-capture-output -n carla0911 python carsim_mpc_quick/orin/run_environment_loop.py --seconds 0.5 --spawn 53 --visualize

# 正向静态障碍物接触反馈原型：前方 6 m 放置 streetbarrier
conda run --no-capture-output -n carla0911 python carsim_mpc_quick/orin/run_environment_loop.py --seconds 1.2 --spawn 0 --obstacle-at 6
```

当前拒绝已有其他车辆的世界，防止更改别人的仿真，同时限制了多车辆验证。
碰撞射线排除自车车辆标签；未分类的 `NONE` 接触会主动中止，不冒充已知接触面施加强制动力。
接触力模型使用 1e6 N/m 弹簧、8e4 Ns/m 阻尼，最大反力 100 kN，且按关闭速度与 1680.2 kg
等效车辆质量限制单步冲量。相关参数尚未进行碰撞实车/高保真模型标定。

所有工件进入 `results/carla_environment/<时间>/`，包含逐步状态、指令、相机帧号、MPC 预瞄、
接触反力、首帧原始深度/语义图及校准包，失败记录也保留。当前是离线同步联合仿真，不保证
10 ms 墙钟实时。`passed` 仅表示该次脚本的数值与接口检查通过，不表示全面物理验收。

默认路段经过 10 s 验证，不能把默认时长任意延长理解为已验收长路线。159 号起点在 10.22 s
附近会遇到未分类的静态几何，程序保留拒绝记录并中止。已在无车辆情况下复查射线命中，确认
该命中仍存在于场景中；没有把它直接当作自车误检而过滤掉。其他路段也可能遇到这类几何。

`report.json` 现在记录各阶段耗时的 p50/p95/p99/max 和超过 10 ms 的比例：
`rpc_ms` 包含 Orin 的感知、控制、序列化与传输，不能称为纯网络 RTT；`solver_ms` 包含
CarSim 积分及地形查询；`cycle_ms` 还包含接触查询和对应步的相机采集、显示处理。
失败也记录已完成步数、时间和拒绝接触的射线坐标，并保存最后一张 `failure.png`。

## 本轮持续运行与接口验证（2026-10-27）

| 验证 | 结果 |
|---|---|
| 原生插件与 VS Connect 探测 | `results/carla_native_probe/20261027_184850/report.json`：插件车辆可移动，启用前后进程均无 UDP 监听；没有验证 VS Connect 握手或悬架变量。继续采用 VS API 主控链路 |
| 10 s 完整地形/图像/Orin 闭环 | `20261027_185112`、优化后 `20261027_185722`、可视化 `20261027_185843` 均通过；每次 1000 步、200 次图像更新、170030 次地形回调 |
| 优化后的正面障碍物 | `20261027_185828`：120 步，0.34 s 接触，最终约 0.194 km/h |
| 预瞄计算优化 | 排序分组中位数、相同点集复用；新图像和横向选点变化均使缓存失效。Orin 上 8 组对照与旧实现误差不超过 1e-12；8 万点重复预瞄中位耗时约 6.22→1.12 ms，新图像后的首次聚合约 5.24 ms。数据见 `results/carla_carsim_research_20261027/depth_aggregation_benchmark.json` |
| 数值兼容性 | 深度通道乘法前转换浮点，修复 NumPy 2 的 uint8 溢出异常；Orin NumPy 1.26 下与旧解码和预瞄对照通过 |
| 图像输入依赖与过期拒绝 | `20261027_185722/input_dependency_check.json`：同状态下真实图像与零预瞄角点力差约 157.55 N；拒绝过期图像，不调用静态路面 |
| 新路段图像几何抽查 | `20261027_185722/camera_geometry_check.json`：20 点，最大约 17.32 mm、平均 8.19 mm。最大误差超过当前 15 mm 观测区工程设定，后者不能视为已验证的鲁棒误差上界 |
| 控制器与主项目回归 | Orin 80 步 MATLAB 对照最大力误差约 7.96e-13 N；`scripts/check_project.py` 七组通过，报告 `results/project_checks/20261027_185323_087121/report.json` |

可视化 10 s 工况的 Orin 计算 p50 约 7.95 ms、p95 约 44.75 ms；完整控制周期 p50 约
21.32 ms、p95 约 55.85 ms。优化改善了重复预瞄与中位耗时，仍未解决图像更新峰值，也未达到
100 Hz 墙钟实时。图像掩码仍为仿真真值，完整碰撞仍按下节列为未完成项。

### 可视化启动时 Pillow DLL 报错

若 `PIL._imaging` 报“DLL load failed：找不到指定的模块”，已确认的一种原因是
`libtiff 4.4.0` 依赖 `libdeflate.dll`，但环境安装的 `libdeflate 1.25` 只提供
`deflate.dll`。Codex 附带的额外 DLL 路径可能掩盖此问题，必须在普通 PowerShell 环境复查。
2026-10-27 已将本机 carla0911 的 libdeflate 固定为 1.14，并验证无 Codex 路径时 Pillow、
NumPy 和 Tk 真实图像渲染通过。没有通过复制其他应用的 DLL 或全局修改 PATH 解决。
本机 `conda-meta/pinned` 同时固定 `libdeflate ==1.14`，避免后续常规更新重新引入不匹配版本。

新建独立环境的版本清单为 `carla0911.environment.yml`：

```powershell
conda env create -f carsim_mpc_quick/orin/carla0911.environment.yml
# 已有环境若再次遇到上述同一依赖问题：
conda install -n carla0911 -c conda-forge libdeflate=1.14
./carsim_mpc_quick/orin/start_environment_loop.ps1 -CheckOnly
```

本机本次使用已缓存的包离线修复，避免网络源连接失败影响修复：
`conda install -n carla0911 --offline --yes E:/anaconda3/pkgs/libdeflate-1.14-hcfcfb64_0.tar.bz2`。
启动器现在先检查导入和 Tk 渲染，失败时尚未创建 CARLA 车辆或连接控制流。
修复后在移除 Codex/libheif/poppler/jxrlib 路径的进程中，通过同一启动器完成 10 s 可视化闭环：
`results/carla_environment/20261027_190919/report.json`，1000 控制步、200 次图像更新，
`passed=true`。依赖预检的 `-CheckOnly` 和 PowerShell 语法解析也通过。

原生接口探测可重复运行（需当前世界没有其他车辆）：

```powershell
conda run --no-capture-output -n carla0911 python carsim_mpc_quick/orin/probe_native_interface.py
```

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
