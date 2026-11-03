"""生成只读资产导航；不加载权重、不移动或删除实验文件。"""

from pathlib import Path


def main() -> None:
    """按结果目录生成索引，未知状态不依据目录名称推断。"""
    root = Path(__file__).resolve().parents[1]
    rows = []
    for path in sorted((root / "results").iterdir()):
        if not path.is_dir() or path.is_symlink():
            continue
        evidence = [
            p.name for p in path.iterdir()
            if p.is_file() and p.suffix in {".json", ".yaml", ".md", ".csv"}
        ][:12]
        rows.append(
            f"| [{path.name}](../results/{path.name}/) | "
            + ("、".join(evidence) or "需检查子目录") + " | 未知 |"
        )
    (root / "docs/EXPERIMENT_INDEX.md").write_text(
        "# 实验资产索引\n\n"
        "保留权重、论文原始结果、数据集及训练日志。目录名称不作为成功、失败或重复的证据。\n\n"
        "## 已核对证据\n\n"
        "- 20260624_214005_ppo：manifest 记载 PPO、1 episode、CPU、ppo.yaml，历史观测 914 维、动作 2 维；不能据此认定收敛。\n"
        "- si_rppo_halfcar_v8：原 claim 报告的核心 PPO 结论为 weak_or_contradicted；ready 不是性能优势结论，未重新验证。\n"
        "- report_final_current_20260624：多算法、多路面轨迹，公平比较条件尚未核实。\n"
        "- archive：沿用 results_cleanup_20260922 的原路径映射。\n\n"
        "实验目的、基线差异、参数、种子、Git、环境、完成状态未核实的均为未知。\n\n"
        "| 目录 | 顶层证据文件，最多十二项 | 有效性 |\n| --- | --- | --- |\n"
        + "\n".join(rows)
        + "\n\n## 新实验规则\n\n"
        "训练沿用 results/<时间戳>_<算法>/，评估使用独立目录和显式权重路径。"
        "保存命令、完整配置、种子、Git 与工作区差异、环境、数据来源、时间、退出码和指标单位。"
        "失败与中断保留证据；latest 只用于开发，不代表最佳模型。\n",
        encoding="utf-8",
    )
    (root / "docs/PROJECT_MAP.md").write_text(
        "# 项目地图\n\n"
        "- config.py、configs/：YAML 继承和算法覆盖。\n"
        "- models/、roads/、envs/：车辆动力学、路面和 Gymnasium 接口。\n"
        "- controllers/：PID、SPDF、MPC、策略适配及动作安全处理。\n"
        "- rl/：四种强化学习算法、网络与经验回放。\n"
        "- scripts/train_rl.py：训练；scripts/evaluate_all.py：权重加载与公共评估。\n"
        "- experiments/、visualization/：轨迹、指标和绘图。\n"
        "- scripts/run_cosim.py：联合仿真主入口，PowerShell 脚本为兼容包装。\n"
        "- carsim_mpc_quick/：独立 MATLAB/CarSim、CARLA/Orin 工程。\n"
        "- docs/projects/：由 projects 整体迁入的答辩源稿、素材与导出成果。\n"
        "- results/、datasets/：实验与数据，见 EXPERIMENT_INDEX.md。\n\n"
        "数据流：配置和路面 → 车辆环境 → 观测 → 控制器或策略 → 动作约束 → 动力学更新 → 奖励、指标与轨迹。\n\n"
        "历史实验名不代表当前功能；历史观测维度须与对应代码、配置核对。"
        "Chrono 已移除；半车算法主线与 CarSim/CARLA 独立验证平台保留。\n",
        encoding="utf-8",
    )
    p = root / "README.md"
    text = p.read_text(encoding="utf-8")
    text = text.split("\n## ?????")[0].rstrip()
    # 上一轮终端编码导致新增段落出现问号，只替换该新增段落。
    lines = text.splitlines()
    while lines and ("????" in lines[-1] or not lines[-1].strip()):
        lines.pop()
    p.write_text("\n".join(lines) + "\n\n## 维护与追溯\n\n"
                 "参见 [项目地图](docs/PROJECT_MAP.md)、[实验索引](docs/EXPERIMENT_INDEX.md) "
                 "和 [整理报告](docs/REFACTOR_REPORT.md)。复现请显式指定权重，latest 不代表最佳模型。\n",
                 encoding="utf-8")


if __name__ == "__main__":
    main()
