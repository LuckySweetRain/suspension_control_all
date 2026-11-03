# Python / CarSim 联合仿真

当前保留 Python 控制器、CARLA/Orin 接口和 CarSim 求解器桥接。MATLAB 源码、Simulink 模型及其运行入口已于 2026-11-03 移除。

## 当前运行链路

Linux/WSL 主循环 → RPC → Windows `orin/carsim_bridge.py` → CarSim DLL。
Python 使用 ctypes 直接调用求解器，不需要 MATLAB 或 Simulink。
CarSim 安装、有效许可和外部车辆数据库仍然必需；现有 Windows DLL 不能在 Linux 容器中直接加载。

- [WSL 工作流与资源配置](../docs/wsl_cosim.md)
- [Orin 控制器与部署](orin/README.md)
- [CARLA 接入](orin/CARLA_README.md)
- [地形与深度预瞄闭环](orin/ENVIRONMENT_LOOP.md)
- [主动/被动对照](orin/COMPARISON.md)

从仓库根目录运行 `python scripts/run_cosim.py --help` 查看入口。
资源配置使用 `configs/cosim.local.json`；数据库位于仓库之外。

## 数值资产与历史结果

Orin 控制器仍需已有的 `results/orin/quick_mpc_data.mat` 和 `parity_fixture.mat`。
这些是 SciPy 读取的数值数据，不是 MATLAB 运行时。它们保留在本地，不随 Git 上传，迁机需要单独备份。
MATLAB 参数生成器已移除，目前没有等价的 Python 矩阵生成器；修改车辆参数、预测域或权重后不能直接沿用旧矩阵。

论文、历史实验结果、模型权重和外部数据库未删除。
旧方法说明和实验记录见 [历史记录](../docs/history/matlab_carsim.md)，旧代码可从 Git 提交 `f915fe7` 追溯。
