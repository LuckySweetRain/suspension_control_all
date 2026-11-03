# Python Orin 控制器与 CarSim 联合仿真

当前主入口为 `scripts/run_cosim.py`，详见 [WSL 工作流](../../docs/wsl_cosim.md)。
Windows Python 直接加载 CarSim DLL，Orin 运行 NumPy/SciPy 控制器及可选 C++ 求解器；不依赖 MATLAB/Simulink。

## 运行与部署

- [CARLA 接入](CARLA_README.md)、[环境闭环](ENVIRONMENT_LOOP.md)、[起伏路](ROUGH_ROADS.md)、[双车对照](COMPARISON.md)。
- [自动驾驶](AUTODRIVING.md)、[风扇配置](FAN_CONTROL.md)。
- 保持 CarSim 求解器许可有效，并配置外部数据库、CARLA 与 Orin 地址。
- 从仓库根目录运行 `python scripts/run_cosim.py --help` 查看运行参数。

部署前，将已有且匹配的 `quick_mpc_data.mat` 和 `parity_fixture.mat` 放到 `results/orin/`。
SciPy 直接读取 MAT 数据，无需安装 MATLAB；数据不在 Git 仓库内，必须单独备份。
当前没有 Python 参数矩阵生成器，不能在改变车辆参数或 QP 设置后继续使用旧矩阵。

先在本机验证：

```bash
python carsim_mpc_quick/orin/verify_controller.py --data results/orin/quick_mpc_data.mat --fixture results/orin/parity_fixture.mat
```

需要更新板端时再执行（会上传文件、验证并重启服务）：

```bash
python carsim_mpc_quick/orin/deploy_orin.py --host 192.168.50.2
```

部署工具使用 Paramiko，交互读取密码并校验文件哈希。新设备依赖见 `requirements.txt`。
控制器保留复位、非有限输入拒绝和不可行状态回退检查；回归样本通过不代表完整联合仿真或实时截止保证。

## 历史证据

MATLAB 生成的数值对照样本和实验结果作为研究证据保留。
旧 TCP/Simulink 运行入口及 `.m` 文件已删除；旧命令仅见 [历史记录](../../docs/history/matlab_orin.md)。
旧源码可从 Git 提交 `f915fe7` 追溯。
