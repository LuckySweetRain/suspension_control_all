# Orin 控制器与 CarSim 联合仿真

CARLA 场景、相机和三端基础闭环的启动与测试见 [CARLA_README.md](CARLA_README.md)。
当前地形闭环、起伏路面和左右对照分别见 [ENVIRONMENT_LOOP.md](ENVIRONMENT_LOOP.md)、
[ROUGH_ROADS.md](ROUGH_ROADS.md)、[COMPARISON.md](COMPARISON.md)。
命令入口、Windows 公共 `runtime/` 和 Orin 部署边界见 [代码结构](../../docs/code_structure.md)。

本配置让 Windows 上的 CarSim 作为被控对象，Jetson Orin 执行现有预瞄 QP-MPC。
控制律、矩阵、预瞄与单位取自 `quick_mpc_config.m` 和 `build_quick_mpc_data.m`。
当前支持 `ideal_force` 主动轴力模式；不支持直接切换 CDC、持续作动力或 RL 权重。

## 接线与固定地址

| 用途 | Orin 网口 | Orin 地址 | 另一端 |
|---|---|---|---|
| 上网、原 SSH 管理通道 | eth0，MAC 尾号 `05:fd` | 原 `192.168.1.41` 配置保留 | 路由器 |
| 控制数据直连 | eth1，MAC 尾号 `05:fe` | `192.168.50.2/24` | 本机 Realtek 以太网，`192.168.50.1/24` |

机壳标号与 Linux 名称应按实际插线核对，已验证 eth1 直连的协商速率为 1 Gbps。
两条连接同时使用需要两根网线。目前只有一根，可先保留 eth1 ↔ 本机用于仿真；
补线后把 eth0 接回路由器。Windows Wi-Fi 可继续上网。

Orin 的 NetworkManager 配置名为 `suspension-direct`，固定绑定 eth1 与对应 MAC，
自动连接，IPv4 不设置默认路由，IPv6 关闭。本机直连口不设置网关和 DNS。
不要对 Wi-Fi 或 eth0 执行直连网卡的重配置。

新电脑需要配置时，可用管理员 PowerShell 执行 `setup_direct_ethernet.ps1`。
脚本会检查网卡描述和现有地址；本机专用网卡接口索引为 22，换电脑须重新核对。

## 日常运行

1. 保持 eth1 与本机网线相连，并确认 `ping 192.168.50.2` 可达。
2. 打开 CarSim 2020.0 Browser，并打开当前 `carsim_active_suspension` 数据库，确保求解器许可可用。
3. 在 MATLAB 执行：

```matlab
addpath('E:/Desktop/workspace/suspension_control_all/carsim_mpc_quick/orin');
result = run_orin_carsim(12); % 12 秒仿真；默认 Orin 为 192.168.50.2:55001
```

该入口在 `results/orin/` 生成独立模型 `carsim_orin_tcp.slx`、独立 CarSim 参数副本与结果，
不替换原 `carsim_half_car_qp_mpc.slx`。先关闭已打开的原模型，避免 Save As 时覆盖编辑状态。
CarSim 积分步长保持 1 ms，控制采样时间保持 10 ms。控制端每个仿真步同步等待 Orin 返回。

Orin 程序位于 `/home/nvidia/suspension_control/orin/`。用户级服务已启用，且 `Linger=yes`：

```bash
ssh nvidia@192.168.50.2
systemctl --user status suspension-mpc.service
journalctl --user -u suspension-mpc.service -n 30 --no-pager
# 需要手动停止或重启时：
systemctl --user stop suspension-mpc.service
systemctl --user restart suspension-mpc.service
```

服务启动后读取 MAT 文件和原生求解器，不使用 GPU。只允许一个仿真客户端；
不要同时运行链路检查与 CarSim 仿真。

## 更新控制参数或代码

先在 MATLAB 执行：

```matlab
prepare_orin_data;
```

再在仓库根目录的本机 PowerShell 执行：

```powershell
python carsim_mpc_quick/orin/deploy_orin.py --host 192.168.50.2
```

部署脚本使用本机 Paramiko，交互输入 SSH 密码，不保存密码。
它上传代码和专用 MAT/对照样本，在 Orin 用现有 g++ 编译 C++ ADMM，运行数值对照后重启服务。
NumPy/SciPy 已存在于 Orin；新设备依赖见 `requirements.txt`。初次连接保存 SSH 主机公钥，
后续验证公钥；SHA-256 确保 MATLAB 与 Orin 使用完全相同的矩阵和路面数据。
不要只更新本机 MAT 而忘记重新部署。

## 验证与故障行为

`prepare_orin_data.m` 生成 80 步 MATLAB 参考序列，覆盖状态重构、道路预瞄、
热启动、非零主动控制力。`verify_controller.py` 对照并检查复位、非有限输入与不可行状态回退。
2026-10-27 Orin 原生后端与参考的最大角点力差为约 `8e-13 N`；单步计算中位数约 3.1 ms，
95 分位约 4.9 ms，80 步样本最大约 7.4 ms。这些是短时实测，不是最坏时间保证。

MATLAB 中可执行 `report=check_orin_link;` 验证真正的双端 TCP、重复 Outputs 缓存和
数据哈希错误回退。80 步实测中，首次调用包含连接和初始化开销，约 20.7 s；
其余调用中位数约 7.4 ms，95 分位约 12.8 ms，最大约 18.8 ms。
当前是同步处理器在环仿真，尚未保证 100 Hz 墙钟实时；不能直接据此连接真实作动器。

每帧是带版本号、连续序号和仿真时间的有界 JSON 行。断线、读超时、序号/时间/哈希不匹配、
非有限响应或实质性 QP 不可行时，本次仿真锁定被动阻尼；重新启动仿真才复位。
TCP 读超时默认 100 ms，不代表 10 ms 实时截止。角点力维持 ±5000 N 饱和，
基线阻尼保持 `3 N/(mm/s)`，轴力等分左右，重复输出 8 路。

原算法采用 800 次迭代上限。12 s 工况复核发现 t=9.14 s 的问题需要 1103 次迭代才收敛，
因此 `prepare_orin_data.m` 为 Orin 部署单独覆盖为 3200 次，原 `quick_mpc_config.m` 基线不变。
提高的是数值求解预算，不修改车辆参数、代价权重、约束或回退阈值。该预算可能增加单步耗时。
达到迭代上限时可能仍给出满足约束容差的近似解。
移植版保留该行为，继续报告 `converged=false`，不把“未满足停止条件”直接等同于不可行。
有限且归一化约束违背不超过 0.005 的候选解经过轴力及变化率投影后可下发；
大于该阈值则锁定被动阻尼。这是数值容差判定，不是实际悬架安全认证。

仿真结果保存 `fault`、原始 `exports` 和 `times`。若进入回退，入口先保存证据再报错，
不会把被动回退段静默标为完整的主动控制结果。10 路诊断沿用原模型定义，
第 5 路表示 QP 收敛，不表示网络连接或主动控制持续率。

## 本次验收记录（2026-10-27）

- 真实 CarSim Bounce Sine Sweep 工况与 Orin 直连运行 2 s，无锁定回退。
  QP 收敛比例 `0.9950`，最大归一化约束误差 `0.00059991`。
  CarSim 报告计算时间/仿真时间比为 `1.302`，进一步说明这次不满足墙钟实时。
- 证据：`results/orin_carsim_verified.log`、`results/orin/carsim_orin_run_2s.mat`。
  `results/orin_matlab_link.log` 与 `results/orin/matlab_link_check.mat` 保存通信检查。
- 最初的 12 s 扩展复核在 MATLAB 启动阶段遇到“无法联系登录服务”，没有开始仿真。
  用户完成登录后，800 次迭代版本在 9.14 s 触发求解回退，证据保留在
  `results/orin/carsim_orin_run_12s_fallback.mat` 与 `results/orin_carsim_12s.log`。
  单独复核同一个 QP：800 次时约束违背为 0.006432；1103 次时收敛，违背降为 0.000590。
- 3200 次部署版本完整运行 12 s，`fault.failed=false`，无通信或求解锁定回退。
  QP 收敛比例 `0.9916736`，最大归一化约束误差 `0.000859449`；
  少数步骤达到 3200 次上限但候选解仍满足可接受的约束容差。
  CarSim 总体计算时间/仿真时间比为 `0.891667`；该总体比值不保证逐步满足 10 ms 截止。
  证据：`results/orin_carsim_12s_3200.log`、`results/orin/verified_12s_summary.json`、
  `results/orin/carsim_orin_run.mat`。原生后端 80 步参考力误差仍小于 `8e-13 N`。
- `python -m ruff check .` 与
  `python -m ruff check --no-force-exclude carsim_mpc_quick/orin` 通过。
- `E:/anaconda3/envs/suspension-control/python.exe scripts/check_project.py` 全部七组检查通过。
  报告在 `results/project_checks/20261027_154552_908279/report.json`。
  最初用 base Python 尝试时缺少 gymnasium，随后已切换正确的 Conda 环境完成检查。
- Orin 编译命令使用 `g++ -std=c++17 -O3 -Wall -Wextra -Werror -fPIC -shared`，通过编译；
  原生后端已通过 MATLAB 数值对照与故障输入检查。没有运行 pytest。

这些检查证明当前接口和短时控制流程可以运行，不证明长时稳定性或控制性能改善。

本项目 Python 通用回归使用 Conda 的 `suspension-control` 环境；不要运行 pytest。
Ruff 检查 Orin 子目录时须加 `--no-force-exclude`，因为原子项目默认在排除列表内。
