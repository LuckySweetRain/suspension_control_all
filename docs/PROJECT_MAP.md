# 项目地图

- config.py、configs/：YAML 继承和算法覆盖。
- models/、roads/、envs/：车辆动力学、路面和 Gymnasium 接口。
- controllers/：PID、SPDF、MPC、策略适配及动作安全处理。
- rl/：四种强化学习算法、网络与经验回放。
- scripts/train_rl.py：训练；scripts/evaluate_all.py：权重加载与公共评估。
- experiments/、visualization/：轨迹、指标和绘图。
- scripts/run_cosim.py：联合仿真主入口，PowerShell 脚本为兼容包装。
- carsim_mpc_quick/：独立 MATLAB/CarSim、CARLA/Orin 工程。
- docs/projects/：由 projects 整体迁入的答辩源稿、素材与导出成果。
- results/、datasets/：实验与数据，见 EXPERIMENT_INDEX.md。

数据流：配置和路面 → 车辆环境 → 观测 → 控制器或策略 → 动作约束 → 动力学更新 → 奖励、指标与轨迹。

历史实验名不代表当前功能；历史观测维度须与对应代码、配置核对。Chrono 已移除；半车算法主线与 CarSim/CARLA 独立验证平台保留。
