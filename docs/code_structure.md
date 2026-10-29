# 代码结构与入口导航

2026-10-29：联合仿真日常启动逻辑集中到 `scripts/run_cosim.py`，原
`start_environment_loop.ps1` 只保留兼容包装。资源路径统一读取
`configs/cosim.local.json`；WSL 独立轻量环境见 [WSL 工作流](wsl_cosim.md)。
这不改变下文的研究工程和部署边界。

2026-10-27 整理。主 Python 研究工程、Chrono 工程和 MATLAB/CarSim 工程有不同的模型、配置和
运行环境，分别保留，不把相似名称的实现合并成同一个动力学模型。

## 三个代码边界

| 范围 | 入口与职责 | 环境 |
|---|---|---|
| 主研究工程 | `scripts/` 调用 `config.py`、`models/`、`roads/`、`envs/`、`controllers/`、`rl/`、`experiments/`、`visualization/` | suspension-control；源码兼容 Python 3.10 |
| Chrono | `chrono/scripts/` 调用 `chrono/chrono_suspension/`，独立维护整车模型、配置和检查 | 见 chrono/README.md |
| MATLAB/CarSim | `carsim_mpc_quick/run_*.m`、`src/` 与仓库已有 `.slx` 模型 | MATLAB、外部 CarSim 数据库 |
| CARLA/Orin 前端 | 下表中的命令入口和 Windows `runtime/` 公共基础设施 | carla0911，Python 3.7 |
| Orin 控制端 | `controller.py`、`depth_preview.py`、`server.py`、`environment_server.py`、`native_solver.py`、`qp_native.cpp` | Orin 独立目录，由 deploy_orin.py 上传 |

## CARLA/Orin 目录

```text
carsim_mpc_quick/orin/
  start_environment_loop.ps1     统一日常入口；支持 -Compare 和 -RoadProfile
  run_environment_loop.py       单车地形、深度预瞄、接触反馈编排
  run_comparison.py             主动/被动双求解器与同一 CARLA 时钟编排
  comparison_worker.py          子进程独占 CarSim DLL
  live_view.py                  单车界面
  comparison_view.py            左右同步界面
  rough_road.py                 OpenDRIVE 几何生成与独立网格校验
  carla_terrain.py               地形坐标转换与轮胎接触回调
  carla_contact.py               有限的正向柔顺接触原型
  runtime/
    paths.py                    仓库、CARLA、数据库路径；延迟加载 CARLA ABI
    carsim.py                   CarSim DLL 生命周期、I/O 和积分
    sensors.py                  相机创建、同帧队列、深度数据包
    protocol.py                 有界 JSON 行通信及连接资源释放
  controller.py                 与 MATLAB 对照的 MPC 控制律
  depth_preview.py              图像点云、空间历史、预瞄与状态参考系
  server.py                     Orin v1 服务，55001
  environment_server.py         Orin v2 服务，55002
  deploy_orin.py                 Orin 白名单部署
  verify_*.py                   数值/几何/协议验证
  validate_rough_road_suite.py   四种路面端到端验收
  carla_stack.py                保留的场景、原生插件与旧三端诊断入口
  probe_native_interface.py     原生插件/UDP 接口探测
  carla0911.environment.yml     唯一的 CARLA 前端 Conda 定义
```

入口调用公共模块，不再为共享功能导入另一个运行入口。公共模块导入时不连接网络、不加载
CARLA 扩展、不创建求解器、不推进仿真。只有应用主循环 tick CARLA，spawn 子进程各自实例化
`runtime.carsim.CarSim`。Windows `runtime/` 不属于 Orin 上传白名单，控制端不依赖它。

## 直接运行

```powershell
# CARLA 地形与 Orin 深度预瞄闭环
./carsim_mpc_quick/orin/start_environment_loop.ps1 -RoadProfile combined
# 左 MPC / 右被动，独立动力学，同一时钟
./carsim_mpc_quick/orin/start_environment_loop.ps1 -Compare
# 原生插件/旧路线诊断仍使用原入口
conda run -n carla0911 python carsim_mpc_quick/orin/carla_stack.py --help
```

详细运行边界见 `carsim_mpc_quick/orin/ENVIRONMENT_LOOP.md`、`ROUGH_ROADS.md` 和
`COMPARISON.md`。`carla_stack.py --mode triple` 保留原 CarSim 数据集路面，不能当作视觉预瞄闭环。

## 清理与兼容性

- 删除旧 `environment_carla0911.yml`，文档统一引用固定兼容 DLL 版本的
  `carla0911.environment.yml`，环境名仍为 carla0911。
- 删除仅有换行的 `build_carsim_mpc_model.m`。它没有构建实现，不能重建模型；六个调用点改为
  `src/require_carsim_model.m` 明确检查已有 `.slx`。模型本身保留，缺少时明确报错。
- 保留 `__init__.py` 等包标记、仍被文档引用的诊断入口、历史论文脚本与独立算法实现。
  不以“当前没运行”或名称相似作为删除依据。
- 根目录 `python -m ruff check .` 现在覆盖 Orin 前端；其子目录配置使用 Python 3.7，主工程
  仍为 Python 3.10。以后不必为 Orin 反复手写 `--isolated` 检查列表。
- 原 `docs/file_inventory.md` 的已有未提交修改保持不变；实验数据、权重、模型、研究材料、
  结果和受保护的临时/Conda 目录未清理。

验证工件见 `results/code_cleanup_20261027/`，包括初始审计、修改前源码备份、提取定义的
AST 对照和重构前后 1000 步轨迹比较。该结果说明本次重构保持行为，不构成新的性能提升结论。
