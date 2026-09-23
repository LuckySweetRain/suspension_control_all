# Chrono连续道路地形

七类道路均已接入原生 `pychrono.vehicle.ChTerrain`：`RoadTerrain` 提供任意平面坐标的
`GetHeight`、`GetPoint`、`GetNormal` 和名义摩擦系数查询。每个无图形环境拥有独立的
地形对象；四个轮胎力通过 `model.road_sample()` 查询这个对象。

原生窗口从同一个高程模型建立 `ChTriangleMeshConnected`，用
`ChVisualShapeTriangleMesh` 渲染连续道路，包含随路面起伏的边线和虚线。
因此凹坑、凸包、阶跃和随机背景都拥有实际的三维路面几何，且与动力学使用同一次随机实现。

## 坐标与形状

道路纵向站位记为 `s`，车身质心到前轴距离为 `a`、到后轴距离为 `b`。
地形的固定世界坐标为 `x=a+s`，z向上、y向左。前轮站位为 `v*t`，后轮为
`v*t-(a+b)`；地形查询不会再额外施加一次轴距延迟。

随车视角中网格坐标为 `x=a+s-v*t`，只平移一次。车辆水平位置仍受七自由度约束，
前进距离由给定车速确定；这与新增纵向加速、转向和车轮转动动力学不同。

左右轮迹来自已有道路发生器。设前后半轮距中较小者为 `h`：

- `y>=h` 使用左轮迹；`y<=-h` 使用右轮迹。
- 两者之间用 `u=(y+h)/(2*h)`、`w=3*u²-2*u³`，计算 `z=(1-w)*z_right+w*z_left`。
- `GetNormal` 使用同一表达式的解析纵、横坡度，返回单位法线。

外侧保持相应轮迹高程，保证前后轮距不同也不会改变原有四轮输入。
这是从双轮迹构建完整宽度地形的明确插值假设；尚未导入实测二维路面或横坡数据。

## 网格与动力学

默认路宽3.6m、纵向节点间距不超过0.025m、横向7个节点。网格覆盖回放或当前回合长度，
另留前后可见距离，一次创建并分块绘制；每帧只移动固定、无碰撞的可视刚体。
网格顶点由解析地形直接采样，三角形内部为线性近似。可在配置中调整
`visualization.road_mesh_dx_m` 和 `visualization.road_width_m`。缩短极窄局部事件时应同时
加密网格和检查物理积分步长；改变网格密度不会改变轮胎使用的解析高程。
标线高出表面1mm用于避免闪烁，道路高程、车辆运动均保持真实米制尺度。

本实现保留严格7个机械自由度和现有单边垂向轮胎弹簧/阻尼本构。没有叠加
`RigidTerrain` 刚体碰撞力，也没有把地形斜法线转换成额外的水平轮胎力。
`GetCoefficientFriction` 的0.8是地形接口的名义值，目前不参与垂向轮胎计算。
采用完整轮胎接触、纵横向行驶或转向需另建车辆模型，不能把本项目称为完整整车接触仿真。

## 使用与并行训练

从仓库根目录运行：

```powershell
conda activate chrono-rl
python chrono/scripts/view.py --controller skyhook --scenario composite --camera road --hold
python chrono/scripts/view.py --controller passive --scenario step --camera side --speed 0.25 --hold
python chrono/scripts/check_terrain.py --with-vector
```

Python接口：

```python
import pychrono as ch

from chrono_suspension.env import ChronoSuspensionEnv

env = ChronoSuspensionEnv()
env.reset(seed=42, options={"scenario": {"kind": "composite"}})
terrain = env.model.terrain
location = ch.ChVector3d(10, 0.8, 0)
height_m = terrain.GetHeight(location)
normal = terrain.GetNormal(location)
env.close()
```

训练worker使用 Chrono 核心和Vehicle地形接口，不建网格、不创建窗口。
本机官方PyChrono 10的 `vehicle.py` 会连带导入Irrlicht和FSI包装器，因此运行时也需要这些
随包提供的模块；导入模块不等于初始化渲染器。训练不要求创建桌面图形上下文。
随机种子仍由各环境派生，原有PPO/SAC多进程入口及动作、观测维度保持不变。
新轨迹记录完整 `resolved_road`，回放核对每帧高程和车轮平面坐标后才建立道路。
回放配置的前后轴距、半轮距必须与录制几何一致，不能仅保持轴距总和相同。
缺少此信息的旧轨迹
只显示记录平台并明确提示，不能从回合种子猜测道路种子。

接口依据：[Chrono 10 ChTerrain定义](https://github.com/projectchrono/chrono/blob/10.0.0/src/chrono_vehicle/ChTerrain.h)、
[PyChrono ChTerrain director绑定](https://github.com/projectchrono/chrono/blob/10.0.0/src/chrono_swig/interface/vehicle/ChTerrain.i)。
