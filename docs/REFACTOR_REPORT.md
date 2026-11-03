# 项目整理报告

## 范围和初始状态

初始 HEAD 为 fc56565。已有 docs/file_inventory.md 修改及未跟踪 command.md，未重置或覆盖这些工作；文件导航仅增量更新材料路径。

## 已完成

- projects 整体迁入 docs/projects，内部目录保持不变。README、文件导航、Git 忽略和 Ruff 排除同步更新。回滚方法：目标不存在时，将 docs/projects 原样移回 projects，再恢复对应导航路径。
- 新增 PROJECT_MAP.md 和 EXPERIMENT_INDEX.md，以及可重新生成索引的 tools/update_maintenance_index.py。
- config.py、rl/networks.py、scripts/evaluate_all.py 增加中文模块说明、维度说明和兼容约束，不改变计算与随机行为。
- 明确联合仿真主入口与 PowerShell 兼容包装关系。
- 保留 results 原位置及权重修改时间，避免改变 latest 候选。复现要求显式指定权重。

## 七项问题进度

| 问题 | 处理与剩余工作 |
| --- | --- |
| 实验导航落后 | 建立全体顶层目录索引；逐实验参数、Git 和有效性核验尚未完成 |
| 结果用途混杂 | 建立证据导航；物理清理尚未执行 |
| 历史能力混淆 | 项目地图明确历史名称与当前能力边界，记录 v8 原报告的负面证据 |
| 中文说明不足 | 配置、网络、权重入口已补充；其他核心模块仍待逐项完善 |
| 启动入口不一致 | 修正文档中的兼容包装器表述 |
| latest 依赖路径 | 保留历史查找行为和资产路径，补充注释及显式权重建议 |
| 活动任务不明确 | 检出 CARLA、CarSim、Python、WSL 进程；目录占用尚不能排除，因此未删除结果 |

## 待执行事项

用户已要求删除 Chrono。包含递归删除的命令被执行策略拒绝，返回 blocked by policy，尚未删除 chrono、envs/chrono_quarter_car_env.py、configs/chrono_quarter_car.yaml、scripts/run_chrono_active_suspension.py。因此对应功能导航和校验特例暂保留，避免文档虚报已完成。

用户指定保留权重和论文原始结果，仅清理旧检查、调试和可再生成产物。当前尚未完成逐文件引用、占用和可再生成依据核验，没有删除结果文件，也没有以 smoke/probe 名称直接认定可删。

历史维护记录不改写为本次成果。训练清单缺失的种子、Git、目的和结论标为未知，不能把文件存在等同于结果有效。

## 验证

- conda run -n suspension-control python -m ruff check config.py rl/networks.py scripts/evaluate_all.py tools/update_maintenance_index.py：通过。
- conda run -n suspension-control python scripts/validate_training_config.py --config configs/default.yaml --out results/maintenance_20261029/config_validation.json：通过，七种场景有效。
- conda run -n suspension-control python scripts/check_gym_registration.py --config configs/default.yaml --smoke：通过，观测 34 维、动作 2 维、数值有限。

未运行完整训练、pytest、CARLA/CarSim/Orin 联合仿真；未改变控制器或训练计算。短检查不能证明训练收敛或性能提升。

## 半车范围收敛

用户已手动删除 Chrono，本轮确认目录不存在。删除默认配置无人读取的 reduced_full_car_mpc、半车评估中的四角与侧倾兼容分支、先验门控中半车恒为零的侧倾项，以及校验脚本中的 Chrono 特例。保留旧控制器名称的显式拒绝检查，避免错误配置静默运行。

按用户确认保留 CarSim/CARLA/Orin：其内部预测控制器是半车模型，整车仅用于高保真验证。历史全车结果和论文数据未修改；当前半车指标函数不再生成全车专属指标。历史回算应使用原代码版本。

验证补充：Ruff（controllers/prior.py、experiments/evaluation.py、scripts/check_project.py）通过；conda run -n suspension-control python scripts/check_project.py 七项检查全部 PASS，报告 results/project_checks/20261029_182245_499782/report.json。Gymnasium 提示动作空间未归一化和观测边界无限，未报失败。未验证完整训练收敛或联合仿真。

## GitHub 同步核验（2026-11-03）

- 原 projects 下 147 个受版本控制文件迁入 docs/projects，Git 内容哈希全部一致。
- 在 suspension-control Conda 环境运行以下命令，全部通过：
  - python -m ruff check .
  - python scripts/check_project.py
  - python scripts/validate_training_config.py --config configs/default.yaml --out results/github_upload_20261103/config_validation.json
  - python scripts/check_gym_registration.py --config configs/default.yaml --smoke
- 项目检查报告：results/project_checks/20261103_154752_518914/report.json。Gymnasium 对动作归一化和无限观测边界给出提示。
- 短时检查不能证明训练收敛或控制性能提升；未运行完整训练和 CarSim/CARLA/Orin 联合仿真。结果、数据集和权重保持本地忽略规则。

## 移除 MATLAB/Simulink（2026-11-03）

按用户要求删除 29 个 MATLAB 脚本和 1 个 Simulink 模型，覆盖旧 QP-MPC 配置/矩阵生成、MATLAB 联合仿真、Orin S-Function/TCP 入口及 MATLAB 论文绘图脚本。更新根目录及子项目指南、文件导航和部署缺失数据提示。原两份子项目 README 作为历史实验说明归档在 docs/history/；旧代码可从 Git f915fe7 追溯。

保留 Python→CarSim DLL/RPC、CARLA/Orin 控制器、论文材料和全部已有实验资产。results/orin/quick_mpc_data.mat 与 parity_fixture.mat 由 SciPy 读取，不需要 MATLAB；已确认本地存在。当前没有 Python 矩阵生成器，新设备需单独恢复这两份匹配资产，调整 QP 配置需要另行实现并验证生成流程。

在 suspension-control Conda 环境执行：

- python -m ruff check .：通过。
- python carsim_mpc_quick/orin/verify_controller.py --data results/orin/quick_mpc_data.mat --fixture results/orin/parity_fixture.mat：NumPy 后端 80 步通过，最大力误差 1.2505552149377763e-12 N，复位/非有限输入/锁定回退通过。
- python scripts/check_project.py：七项通过，报告 results/project_checks/20261103_165613_617325/report.json；Gymnasium 保留动作归一化和无限观测边界提示。
- python scripts/run_cosim.py --help：入口可用。
- git diff --check：通过。工作源码范围已无 .m/.slx/.mdl/.mlx 文件；未扫描受保护目录，未删除历史结果。

未启动真实 CarSim/CARLA/Orin 联合仿真、远端部署或完整训练，短时回归不代表控制性能或实时性保证。原有 .qoder 两项删除保持未提交，不包含在本次改动中。
