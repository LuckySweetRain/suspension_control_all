# CarSim 固定运行输入

`runtime_run.par` 是 Python 联合仿真使用的本地基础参数模板，属于运行输入，
不是训练结果。由对应 CarSim 数据库导出/准备后放在本目录。
它与外部车辆数据库和求解器配套，不随 Git 分发；新机器需单独准备。

默认路径及 `configs/cosim.example.json` 均指向这里，机器本地设置为
`configs/cosim.local.json` 的 `carsim_template` 字段。

每次运行派生的 `runtime.par`、`runtime.sim`、求解器输出及审计日志仍放在
`results/carla_environment/<时间戳>/`，因为它们是本次运行的产物。
模型训练结果仍位于各自的 `results/<实验名>/`。

2026-11-05 从 `results/orin/runtime_run.par` 等字节迁移至本目录。
迁移时仍在运行的旧进程可能持有旧绝对路径，因此旧副本暂时保留以兼容；
当前默认配置和本地配置不再使用旧位置。后续实验不要引用旧副本。
