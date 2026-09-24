# 七自由度通用轿车：参数来源与适用范围

核查日期：2026-09-22。对应配置：`configs/sedan_reference.yaml`。

用户CarSim数据库的D-Class Sedan另有独立配置 `configs/carsim_dclass.yaml`；其来源、运动比
换算、非线性表与限制见 [CarSim参数迁移](carsim_parameters.md)。本页继续描述默认Sedan基准。

本模型采用 **Project Chrono 10.0.0 Sedan 的部分公开参数，加上明确声明的轮端等效工程假设**。它不是原 Sedan 多体模型的完整几何约化，也没有经过实车数据标定。`sedan_reference` 表示可追溯的研究参考车，不表示对具体量产车型的复刻。

## 1. 坐标、顺序与七自由度的定义

- 世界坐标为 x 向前、y 向左、z 向上；四角数组固定为 FL、FR、RL、RR。
- 七个机械自由度是车身升沉、侧倾、俯仰和四个簧下质量的垂向位移；机械状态包含七个位移及七个速度。
- 车身主体和四个簧下质量共五个动态刚体。其30个速度坐标受23个双边约束限制；轮胎弹性力、悬架力和主动执行器不增加机械自由度。
- 四轮不包含轮转、转向、侧向运动；车辆前进通过空间路面采样和轴距时延表示。完整悬架连杆运动学、轮胎侧偏/纵滑及车身弹性不属于本七自由度模型。
- 俯仰采用绕 y 轴的右手正方向，因此正俯仰使车头降低。不同文献中的“车头抬起为正”不能直接代入。

## 2. 直接引用的官方参数

所有链接固定到上游 `10.0.0` tag，避免跟随 `main` 变化。

| 本模型字段 | 数值 | 上游含义及引用方式 |
|---|---:|---|
| `sprung_mass_kg` | 1515 kg | Sedan chassis body mass；本模型将该主体质量用作簧载质量，未声称包含约化后所有悬架连杆的簧载贡献 |
| `inertia_kgm2` | [222.8, 944.1, 1053.5] kg·m² | 车身主体质心处的 Ixx、Iyy、Izz；上游惯性积为0 |
| `front_distance_m` / `rear_distance_m` | 1.388 / 1.388 m | 前后悬架参考点 x=±1.388，轮心局部 x=0；上游车身CG局部 x=0 |
| `front_half_track_m` / `rear_half_track_m` | 0.7979 / 0.8 m | 左侧轮心在前、后悬架坐标系内的 y 值 |

来源：[车身主体](https://github.com/projectchrono/chrono/blob/10.0.0/src/chrono_models/vehicle/sedan/Sedan_Chassis.cpp)、[整车悬架安装位置](https://github.com/projectchrono/chrono/blob/10.0.0/src/chrono_models/vehicle/sedan/Sedan_Vehicle.cpp)、[前双横臂硬点](https://github.com/projectchrono/chrono/blob/10.0.0/src/chrono_models/vehicle/sedan/Sedan_DoubleWishbone.cpp)、[后多连杆硬点](https://github.com/projectchrono/chrono/blob/10.0.0/src/chrono_models/vehicle/sedan/Sedan_MultiLink.cpp)。

上游车身CG相对 chassis reference 的位置为 `(0, 0, 0.2) m`，这是**局部参考坐标偏置**，不是车身CG离地高度。因此本项目的0.72 m车身高度不能以该偏置作为依据。

## 3. 按官方公式派生的轮胎参数

Sedan 采用245/40R18轮胎；轮胎质量11.5 kg，载荷指数LI=97，载荷指数参考胎压350 kPa，使用胎压220 kPa。轮辋质量13.2 kg。[轮胎输入参数](https://github.com/projectchrono/chrono/blob/10.0.0/src/chrono_models/vehicle/sedan/Sedan_TMeasyTire.cpp)、[轮辋参数](https://github.com/projectchrono/chrono/blob/10.0.0/src/chrono_models/vehicle/sedan/Sedan_Wheel.cpp)。

`GuessPassCar70Par` 的垂向参数估算使用：

```text
section_height = 0.245 * 0.40 = 0.098 m
rim_diameter = 18 * 0.0254 = 0.4572 m
unloaded_radius = section_height + rim_diameter / 2 = 0.3266 m
load_force = 730 * 9.81 = 7161.3 N              # LI 97
maximum_reference_deflection = 0.16 * section_height = 0.01568 m
k_t = load_force / maximum_reference_deflection = 456715.5612244898 N/m
c_t = 2 * 0.5 * sqrt(k_t * 11.5) = 2291.77419351943 N·s/m
```

0.5是官方头文件给出的默认垂向阻尼比。此估算路径中，胎压改变额定载荷参数，但上述 `k_t`、`c_t` 公式本身没有再乘胎压比例。配置保留计算精度是为了复现公式，不代表测量精度。[公式与载荷指数表](https://github.com/projectchrono/chrono/blob/10.0.0/src/chrono_vehicle/wheeled_vehicle/tire/ChTMeasyTire.cpp)、[默认参数声明](https://github.com/projectchrono/chrono/blob/10.0.0/src/chrono_vehicle/wheeled_vehicle/tire/ChTMeasyTire.h)。

当前七自由度模型只采用该估算结果形成垂向弹簧阻尼力，并实施接触分离/法向力非负条件；它没有实例化完整TMeasy轮胎，也未复制其轮胎压到底附加刚度、摩擦和滑移处理。尖锐路面或轮胎严重压缩工况不能按完整TMeasy的验证范围解释。

## 4. 明确的工程假设与未完成的约化

| 项目 | 默认值 | 性质与限制 |
|---|---|---|
| 四角等效簧下质量 | 每角45 kg | 工程假设；未进行悬架连杆动能等效，整车参考总质量为1515+4×45=1695 kg |
| 轮端悬架刚度 | 前30,000、后32,000 N/m | 工程假设，不是官方弹簧元件刚度；适用于当前垂向等效力元件 |
| 轮端阻尼 | 前2200、后2400 N·s/m | 工程假设，压缩/回弹倍率默认均为1 |
| 车身CG初始离地高度 | 0.72 m | 工程假设；悬架车身端安装点在CG下方0.1 m，由当前模型代码定义 |
| 渐进止挡起点 / 行程界限 | ±0.065 / ±0.09 m | 相对静平衡处悬架力元件长度的变化；不是官方硬点行程 |
| 止挡二次系数 | 2,000,000 N/m² | 工程假设，两侧止挡对称；超过0.065 m的部分按平方增加反力 |
| 三次弹簧项 | 0 N/m³ | 默认关闭，避免把无数据的非线性描述为已标定行为 |
| 主动力限 / 力变化率 | 每角3500 N / 50,000 N/s | 工程假设 |
| 执行器时间常数 / 时延 | 0.02 / 0.01 s | 一阶响应与纯时延假设；这些内部状态不增加机械自由度 |
| 执行器功率参数 | 每角5000 W | 当前实现按相对速度限制目标力；受力变化率和一阶动态影响，不等同于瞬时实际功率的硬钳位 |
| 物理步长 / 控制周期 | 0.001 / 0.01 s | 数值设置；应通过步长减半比较确认关注工况的时间离散误差 |

轮心高度不作为独立调节参数，而由静态载荷和轮胎压缩计算：

```text
front_sprung_corner_load = m_s * g * b / (2 * (a + b))
rear_sprung_corner_load = m_s * g * a / (2 * (a + b))
corner_tire_load = sprung_corner_load + m_u * g
wheel_z0 = tire_radius - corner_tire_load / k_t
```

当前模型通过静载预载初始化悬架，其参考长度为静平衡处的安装长度；这不等于官方弹簧自由长。

**没有直接抄入轮端配置的官方元件参数：**

| 官方悬架元件 | 前双横臂 | 后多连杆 |
|---|---:|---:|
| 弹簧元件刚度 N/m | 73574.10163 | 167062 |
| 减振器元件阻尼 N·s/m | 15054.53731 | 15000 |
| 弹簧自由长 m | 0.511468474 | 0.339 |
| 显式弹簧预载 N | 7492.646764 | 构造时未提供额外预载 |

前悬架代码还在自由长±0.04 m处设置止挡；这一位置不是七自由度模型相对静平衡的轮端行程。前源码虽然声明了非线性减振器类，实际构造器接入的仍是线性阻尼。后源码没有对应的同一套止挡设置。来源为前、后悬架链接。

质量约化同样没有完成：仅将轮辋、轮胎、spindle、upright相加，前角为27.2 kg，后角为29.004 kg；前上/下控制臂另有2.643 kg，后三类连杆另有21.858 kg。连杆既平移又转动，不能把其全部质量直接堆到轮端；遗漏的簧载贡献也不能凭空宣称已经计入1515 kg。

若以后升级为真正的几何约化基线，应在相同静平衡工作点从硬点运动学计算弹簧和减振器各自的运动比，通过虚功和动能映射轮端力、阻尼与质量。常见近似为 `k_w ≈ k_s * r_s²`、`c_w ≈ c_d * r_d²`；预载存在时还需检查几何刚度项，不能仅平方缩放。完成前应持续将本表中的等效参数标为 `assumed`。

## 5. 官方 Windows 原生可视化包核查

已核对官方 Anaconda 包：`projectchrono/label/release::pychrono=10.0.0=py312h418371c_0`，平台 `win-64`，Python范围 `>=3.12,<3.13`。[包元数据 API](https://api.anaconda.org/package/projectchrono/pychrono)。

该包SHA256为 `8429923678c714c8b49f0a04c92990b1002457396c0765cfc250ac46fa34d92e`。构建配方明确启用 `IRRLICHT`、`VEHICLE`、`PYTHON`，关闭 `VSG`；同时还打包了本项目不需要的Sensor/FSI相关依赖。[10.0.0 Windows构建配方](https://github.com/projectchrono/chrono/blob/10.0.0/contrib/packaging-python/conda/bld.bat)。

原生包依赖包括Irrlicht1.8.5、MKL2025.2.0、CUDA12.8相关运行库。2026-09-22已通过官方release与conda-forge两个频道实际安装到独立的 `chrono-rl` 环境。`pychrono`、`pychrono.irrlicht` 导入、NumPy矩阵运算、Irrlicht实时渲染、轨迹录制及原生回放均已在本机运行；截图位于外层 `results/chrono/visualization/`。Conda精确包清单位于 `results/chrono/validation/conda_explicit_win64.txt`。

本机构建没有完整暴露Irrlicht GUI控件类型，因此三维窗口配合Tk原生控制面板使用，仍然无需浏览器。默认学习器采用PyTorch CPU构建，GPU构建作为独立可选安装，不能把已安装的Chrono CUDA依赖当作已通过GPU学习器验收。

这里的CUDA包依赖不表示七自由度车辆物理在GPU批量运行，也不代表已确认本机NVIDIA驱动可运行Sensor/CUDA12.8功能。训练物理由CPU多进程运行；学习器的PyTorch构建及CUDA版本须按本机驱动另行选择。项目不应为了使用Irrlicht或CPU车辆动力学自动升级显卡驱动。

对比核查的旧官方 `9.0.0=py310_4852` 在同样两个频道下因缺少它锁定的 `mkl=2020.0.0` 而解算失败，不能描述为已经可用的备用方案。已有core-only安装也不能靠额外安装一个同名Python渲染包获得编译时未包含的Chrono可视化模块。

## 6. 研究报告中的表述

建议采用“基于Chrono刚体与约束求解器的七自由度主动悬架模型；车身/几何采用官方Sedan公开参数，轮胎垂向系数复现官方TMeasy估算公式，轮端悬架和执行器采用公开列出的工程假设”。不要使用“实车标定模型”“完全复现Sedan”或“已实现GPU并行车辆仿真”。数值检查、短时训练和策略性能比较也不能替代实车标定。
