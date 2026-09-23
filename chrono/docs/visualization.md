# 原生三维可视化与轨迹回放

从仓库根目录、启用 `chrono-rl` 环境后运行。使用 Project Chrono 的 Irrlicht 原生窗口，无浏览器和 Web 服务。仅打开可视化时初始化渲染器；并行训练 worker 不创建窗口。

```powershell
python chrono/scripts/view.py --controller skyhook --scenario bump --hold
python chrono/scripts/view.py --controller passive --scenario sine --speed 0.25 --camera side
python chrono/scripts/view.py --controller skyhook --scenario composite --camera road --hold
python chrono/scripts/view.py --checkpoint results/chrono/你的运行目录/checkpoints/final --hold
```

`--checkpoint` 指向同时包含 `model.zip`、`vecnormalize.pkl`、`metadata.json`、`config.json` 的目录。加载时恢复观测模式及归一化统计，推理使用确定性动作，不再训练归一化器。路径必须替换为实际存在的 checkpoint 目录。

界面提供暂停、0.05～4 倍速、车身外壳显示/隐藏，以及等轴、侧面、正面、俯视、前方道路五种视角。鼠标拖动旋转/平移，滚轮缩放。仪表显示真实仿真时间、道路前进距离、车身垂向加速度、侧倾/俯仰、四角悬架行程和实际执行器力。界面采用英文以避免原生默认字体缺少中文字形。

部分官方 PyChrono 10 构建没有包装 Irrlicht 的具体 GUI 控件类型。这种情况下会打开一个 **Tk 原生控制面板**，3D 画面仍在 Irrlicht 原生窗口中；两者都不是浏览器。若当前构建提供完整控件绑定，控制面板直接绘制在 Irrlicht 内。关闭任一窗口均结束可视化。截图仅保存 3D 窗口；四角力标注仍在截图内，行程及完整数值仪表在原生控制面板中。

## 视觉与物理的对应关系

- 坐标为 x 向前、y 向左、z 向上；顺序为 FL、FR、RL、RR。正侧倾为右手 x 旋转，即左侧上抬；正俯仰为右手 y 旋转，即车头下俯。
- 可视化使用独立 Chrono 系统中的固定、无碰撞刚体，逐帧读取实际车身姿态和轮心位置。它不增加物理自由度，不改变质量、惯量、接触或悬架力。
- 车身采用简化外观，默认隐藏外壳以便看清悬架。车轮是未形变圆柱；没有虚构车轮旋转、转向或连杆机构。
- 黄色弹簧的端点来自仿真导出的实际悬架位置。连续道路由 `ChTriangleMeshConnected` 和 `ChVisualShapeTriangleMesh` 建立，顶点调用与轮胎相同的 `RoadTerrain.GetHeight`；网格为地形的离散近似，默认纵向间距不超过2.5cm。
- 采用随车坐标显示：车身的水平位置保持不变，道路整体按 `-v*t` 平移；同侧前后轮经过同一个路面事件，间隔为轴距/车速。白色标线与道路同步移动，没有新增纵向车辆动力学。横向插值假设见 [Chrono地形](terrain.md)。
- 所有物体坐标保持真实米制尺度，没有放大悬架运动。红/蓝箭头表示正/负执行器力在车身上的方向，箭头长度按 **0.15 m/kN** 绘制，是力的图示而非位移。
- 画面中轮胎可能轻微穿入路面：物理模型以轮胎径向压缩计算法向力，圆柱外观本身没有随载荷形变。离地判据以记录的物理量为准。

完整 Chrono 安装包含官方 `vehicle/sedan/sedan_chassis_vis.obj` 外壳，但本项目暂不接入。Chrono 10 的 [C++ Sedan 车身](https://github.com/projectchrono/chrono/blob/10.0.0/src/chrono_models/vehicle/sedan/Sedan_Chassis.cpp) 使用质心相对参考原点的 z 偏置 0.2 m，而同名网格的 [JSON 车身定义](https://github.com/projectchrono/chrono/blob/10.0.0/data/vehicle/sedan/chassis/Sedan_Chassis.json) 使用 0.7 m；其质量参数也不同。进一步核对安装包的悬架参考坐标后，两种原点转换均不能直接对应本项目静态平衡的轮心高度。为避免外壳与轮胎错位，目前保留经过实际渲染验证的简化几何车壳，不缩放或任意平移官方网格掩盖差异，也不复制第三方资产到仓库。采用部分 Sedan 质量、惯量及尺寸参考不等于完整复现该车型。

## 记录、回放、截图

```powershell
# 无图形地记录每个控制步；只在父进程导出，适合先批量计算再看结果
python chrono/scripts/view.py --controller lqr --scenario bump --no-window --record results/chrono/visualization/lqr_bump.jsonl

# 回放不调用 DoStepDynamics，也不需要重跑控制器
python chrono/scripts/view.py --replay results/chrono/visualization/lqr_bump.jsonl --speed 0.25 --hold

# 原生渲染冒烟与截图（会打开窗口，完成后自动关闭）
python chrono/scripts/view.py --controller skyhook --scenario bump --duration 2 --screenshot results/chrono/visualization/skyhook.png

# 显示几何车壳；默认剖视显示悬架
python chrono/scripts/view.py --controller skyhook --scenario left_bump --body-shell --hold
```

所有新工件写入仓库 `results/`。已存在的记录和截图不被覆盖。JSONL 第一行为版本 `chrono-suspension-frame-v1` 的 metadata，包含完整配置、种子、场景、控制器和 `resolved_road`；后续每行对应一个控制时刻。回放从解析后的道路元数据重建同一次随机实现，并逐帧核对四轮高程，超出1e-8m绝对容差（另加1e-8相对容差）就报错。回放直接使用记录的姿态，没有重新积分或插值，因此渲染帧率高于控制频率时可能重复显示状态。

完整道路回放还会逐帧核对车轮平面坐标，防止覆盖配置后改变前后轴位置或半轮距，导致画面道路与实际轮迹错位。旧记录缺少 `resolved_road` 时，终端和窗口明确提示仅显示记录中的四轮高程平台，不猜测完整随机道路。也支持没有 metadata 的 `export_frame()` JSONL，此时应通过 `--config` 传入生成它时的配置，尤其保证车轮半径一致。时间必须单调不减；位置、路面输入必须有限，四元数必须为 wxyz 顺序的单位四元数。缺少悬架端点时按当前七自由度模型的车身安装位置重建视觉端点。

API：实时显示用 `NativeViewer(model, config)`；回放用 `NativeViewer(None, config, road=road, time_range_s=(start, end))`。另有 `render(frame) -> bool`、`screenshot(path)`、`close()`。`render()` 返回 `False` 表示关闭窗口；`paused`、`speed` 为原生控件更新的回放状态。渲染模块不自行推进环境，由调用方负责控制周期和速率。

## 依赖与验证边界

无图形训练使用 `pychrono` 核心与 `pychrono.vehicle` 地形接口，不生成道路网格。本机官方PyChrono 10的Vehicle包装器也会导入Irrlicht与FSI模块，需要随包提供这些依赖；它不会自动创建窗口。打开原生画面时才初始化Irrlicht渲染器并需要桌面图形会话。控件绑定不完整时使用 Conda 的 `tk` 包（Python 标准库接口 `tkinter`）。帧率、暂停、倍速只影响显示和步进调度，不改变动力学时间步长。

原生接口参考：[Chrono 10 Irrlicht 头文件](https://github.com/projectchrono/chrono/blob/10.0.0/src/chrono_irrlicht/ChVisualSystemIrrlicht.h)、[绘图辅助接口](https://github.com/projectchrono/chrono/blob/10.0.0/src/chrono_irrlicht/ChIrrTools.h)、[官方可视化示例](https://github.com/projectchrono/chrono/blob/10.0.0/src/demos/python/irrlicht/demo_IRR_assets.py)。
