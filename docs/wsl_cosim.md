# WSL 联合仿真工作流

## 当前可运行的边界

2026-10-29：WSL 使用 Ubuntu-22.04，独立 Conda 环境为 `suspension-cosim`。
当前默认模式的仿真主循环、CARLA Python 客户端、地形查询、传感器处理、可视化和录像
都在 WSL 本地执行。Windows 只保留 CarSim DLL 求解桥接，以及现有 CARLA 渲染服务；
模型推理与悬架接口仍在 Orin。Windows DLL 不会加载进 Linux Python。

```text
WSL Ubuntu
  scripts/run_cosim.py                 管理入口
    suspension-carla0911 / Python 3.7  本地主循环、相机、路面、显示、录像
      ├─ CARLA RPC 2000 → Windows CARLA 渲染服务
      ├─ solver RPC 56001 → Windows carsim_bridge.py → CarSim DLL
      └─ TCP 55002/55003 → Orin

WSL suspension-cosim / Python 3.10     数值验证、分析和绘图
```

CarSim 确实提供 Linux 求解器：

- [CARLA 0.9.11 官方 CarSim 集成说明](https://carla.readthedocs.io/en/0.9.11/carsim_integration/)
  在 Ubuntu 一节明确使用 `SOFILE /opt/carsim_2020.0/lib64/libcarsim.so.2020.0`。
- [CarSim 厂商介绍](https://www.carsim.com/downloads/pdf/CarSim_Introduction.pdf)
  列出了 Linux 64 位库 `libcarsim.so.2020.1`，并说明非实时 Linux 版没有 VS Browser。
- [2026.1 官方系统要求](https://aws.carsim.com/users/pdf/release_notes/all/system_requirements.pdf)
  列出 Ubuntu 22.04/24.04；该说明不能倒推 CarSim 2020 或 WSL 已受支持。

本机尚未取得并验证 Linux 求解器和对应许可。官网文档中的存在性不代表匿名公开下载、
Windows 许可自动通用或 WSL 许可已经可用。未来全 Linux 迁移还需要匹配的 Linux 库、
许可、数据库路径和 CARLA Linux 客户端 ABI 的真实验证。当前运行器不自动下载商业软件。
本次点击官方 CarSim 下载和 2020.0 插件下载入口均跳转到厂商账户登录页，
没有取得匿名可下载的完整 Linux 安装包。

## 激活独立环境

在 WSL 终端执行：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate suspension-cosim
cd /mnt/e/Desktop/workspace/suspension_control_all
```

环境位置为 `~/anaconda3/envs/suspension-cosim`，不使用项目中的受保护 Conda 目录。
已另外建立最小客户端环境 `~/anaconda3/envs/suspension-carla0911`。这是 CARLA 0.9.11
Linux API 的 Python 3.7 ABI 要求；管理入口会自动调用它，日常不需要手动切环境。
保留 3.10 的管理/分析环境，避免为旧仿真 API 降级整套研究依赖。
新设备可从仓库根目录重建：

```bash
conda env create -f environments/cosim-wsl.yml
conda env create -f environments/carla0911-wsl.yml
python scripts/setup_carla_linux_client.py
```

客户端初始化脚本从 CARLA 0.9.11 Linux 发行包提取匹配的 egg，只保留约 24 MB
客户端，不复制另一套地图和服务端；已有发行包可用 `--archive` 指定。首次在线提取需要
流式读取较大的官方发行包，可能耗时较长。包源、大小和 SHA256 会保存到同目录 JSON。
两个 WSL 环境均不包含 Torch、Gymnasium、PyChrono、MATLAB 或 Linux CarSim 商业库。
它不替代根目录 `environment.yml` 定义的 RL 研究环境；训练研究仍使用自己的环境。

## 一份资源配置

首次复制 `configs/cosim.example.json` 为 `configs/cosim.local.json`，然后填写本机路径。
本机的 local 配置已经生成，不提交 Git，也不包含密码。路径字段可用 Windows 绝对路径，
或相对于仓库根目录的相对路径；不随终端当前目录改变。

| 字段 | 用途 |
| --- | --- |
| `carla_root`、`carla_egg` | Windows CARLA 服务目录与兼容旧入口的 Windows egg |
| `linux_frontend_python`、`linux_carla_egg` | WSL 客户端解释器及 Linux 0.9.11 egg |
| `carla_host`、`carsim_bridge_host`、`carsim_bridge_port` | CARLA 与本机求解器 RPC 端点 |
| `carsim_program`、`carsim_library` | CarSim 安装目录与 Windows DLL |
| `carsim_database` | 当前车辆数据库 |
| `carsim_template` | 已生成的基础 8/77 参数模板 |
| `windows_conda`、`windows_environment` | Windows Conda 可执行文件与旧 CARLA 前端环境 |
| `orin_host` | Orin 直连地址 |

传入其他配置可使用 `--config /path/to/config.json`。Windows 内部每步仍按 1 ms 积分，
WSL 每 10 ms 仿真时间交换一次控制与状态。Windows 地形回调缺少地面采样时，
在同一有界 RPC 通道批量请求 WSL 查询 CARLA；WSL 等待积分结果时处理这些回调，
避免双方同步等待死锁。5 cm 网格缓存减少重复跨系统请求；相机和模型数据不经过 DLL 桥。
桥接只监听本机 127.0.0.1；当前 WSL 镜像网络已验证可用，未修改系统网络模式。
每轮有地形回调的求解器结束后，桥接进程退出，避免复用已失效的 DLL 全局回调。

## 日常使用

先保持 CarSim Browser 和有效求解器许可，确认 Orin 服务已启动。

```bash
# 核对 WSL 本地 Python、CARLA API、Pillow 与 WSLg；不验证商业许可
python scripts/run_cosim.py --check-only

# 只打印启动命令，不启动服务或车辆
python scripts/run_cosim.py --planner transfuser --town Town03 --dry-run

# 可视化并录制，运行结束自动关闭本轮窗口
python scripts/run_cosim.py --planner transfuser --town Town03 --spawn 159 \
  --unknown-contact record --seconds 30 --record --close-on-complete
```

换 Town 时重新选择有效起点。`record` 是未分类接触只记录的演示策略，
不是完整碰撞验证。主动悬架仍由 Orin 的持久服务开关保持关闭：主动轴力为零，
被动阻尼继续工作；这个启动器不会修改该开关。

原 Windows 命令仍可用：

```powershell
.\carsim_mpc_quick\orin\start_environment_loop.ps1 -Planner transfuser -Town Town03 -Seconds 30 -Record
```

若要显式退回“WSL 只启动 Windows 主循环”的旧模式，可添加
`--backend windows-interop`。当前拆分模式先支持单车闭环；双车对照仍使用该兼容模式，
防止在同一 Windows DLL 进程中并发运行两份车辆模型。

结果仍在仓库 `results/carla_environment/<时间戳>/`，没有复制另一套结果目录。
WSL 可直接读取 CSV/JSON，也可运行 `plot_carsim_states.py` 导出图：

```bash
MPLBACKEND=Agg python carsim_mpc_quick/orin/plot_carsim_states.py --help
```

## 精简范围

- 从 PowerShell 删除重复的依赖检查、网络检查、CARLA 启动和命令拼装实现，统一到
  `scripts/run_cosim.py`。原脚本只保留参数兼容及 Windows Conda 引导。
- 从 DLL 包装器及启动器移除硬编码资源路径，共用 `runtime/paths.py` 和 JSON 配置。
- 不复制原来的 600 多行仿真主循环到 WSL，不另维护一套传感器、地形或控制协议。
- 新增的 `carsim_bridge.py` 仅管理 DLL 与回调；`runtime/solver_rpc.py` 保持原
  `initial/step/t/close` 接口。原主循环直接在 Linux 执行，不通过 Windows 执行它。
- 保留主研究工程、Chrono、MATLAB 源码和验证入口。这些有不同用途，不作为重复代码删除；
  数据集、权重、论文源稿、旧实验结果及受保护目录不清理。

WSL 检查不需要 CARLA 的 Windows ABI：

```bash
python carsim_mpc_quick/orin/verify_frontend_runtime.py
python carsim_mpc_quick/orin/verify_controller.py \
  --data results/orin/quick_mpc_data.mat --fixture results/orin/parity_fixture.mat
python carsim_mpc_quick/orin/verify_stop_start.py
ruff check .
```

完整主研究工程的 `scripts/check_project.py` 仍在安装了 Torch/Gymnasium 的研究环境中执行。

## 本次验证记录

本次真正拆分后的记录：

- `results/carla_environment/20261029_173144/`：Town03，100 步、20 帧录像、WSLg 显示通过。
  报告 `coordinator_platform=linux`、`solver_backend=windows-rpc`，
  Windows DLL 进程 `solver_process_has_carla_client=false`；两端 CARLA 均为 0.9.11。
- `verify_solver_rpc.py` 对照真实 Windows 本地积分与 WSL RPC：各 100 步、17,030 次
  非平地面回调，全部 82 维状态最大误差约 `4.46e-11`。这是接口数值对照，不是控制性能评估。
- 原始主动悬架关闭状态继续保留，运行入口没有修改 Orin 服务开关。

下列为上阶段仅通过 Windows interop 的记录，不能作为 WSL 本地主循环的证据：

- 从 WSL 发起 Town03 的 1 s 仿真，真实 CARLA + CarSim + Orin 完成 100 步，
  原始结果：`results/carla_environment/20261029_165225/`。
- 从 WSL 发起 0.5 s 录制，完成 50 步并输出 10 帧视频，结果：
  `results/carla_environment/20261029_165510/`。
- WSL 中协议分帧/资源检查、零速 DLL 平台拒绝、80 步 MATLAB 数值对照、
  停车起步检查和中文状态图导出通过；数值对照角点力最大误差约 `1.25e-12 N`。
- Windows 和 WSL 的 Ruff 检查通过。主研究工程七项检查通过，报告：
  `results/project_checks/20261029_165354_467629/report.json`。

这些是拆分工作流的短时集成检查，未验证 Linux CarSim 求解器、WSL 商业许可、
长距离自动驾驶或墙钟实时性。没有据此删除历史研究代码或重新开启主动悬架。
