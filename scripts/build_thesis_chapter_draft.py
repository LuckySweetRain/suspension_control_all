"""Build a Chinese thesis chapter draft from current evidence artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _pairwise_row(path: Path, comparison: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    rows = df[df["Comparison"] == comparison] if "Comparison" in df else pd.DataFrame()
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def _read_strategy_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def build_thesis_chapter_draft(
    out_path: Path = Path("docs/thesis_chapter_draft_cn.md"),
    bp_summary_path: Path = Path("results/mpc_parameter_seed_sweep_formal_gate/mpc_parameter_seed_sweep_summary.json"),
    si_strategy_csv_path: Path = Path("results/si_rppo_strategy_evidence/si_rppo_strategy_evidence_table.csv"),
    si_preview_pairwise_path: Path = Path("results/si_rppo_preview_robustness/si_rppo_preview_robustness_pairwise.csv"),
) -> dict[str, str]:
    bp = _load_json(bp_summary_path)
    strategy = _read_strategy_table(si_strategy_csv_path)
    si_vs_ppo = _pairwise_row(si_preview_pairwise_path, "SI-RPPO vs PPO scratch")
    si_vs_sac = _pairwise_row(si_preview_pairwise_path, "SI-RPPO vs SAC")
    si_vs_td3 = _pairwise_row(si_preview_pairwise_path, "SI-RPPO vs TD3")

    strategy_rows = []
    if not strategy.empty:
        for _, row in strategy.iterrows():
            strategy_rows.append(
                f"- {row['Component']}：{row['Evidence']}，主要结果为 {row['MainResult']}，状态为 `{row['Status']}`。"
            )

    lines = [
        "# 毕业论文方法与实验章节草稿",
        "",
        "本文档由当前实验 artifact 自动生成，用于把工程证据迁移到毕业论文正文。所有结论均应保持与证据包、claim matrix 和 completion audit 一致。",
        "",
        "## 1. 研究路线定位",
        "",
        "本文不将道路视觉重建或 PreScan 路面轮廓提取作为核心研究内容，而是假设上游感知模块已经输出沿车辆轮迹的路面不平度预瞄序列。研究重点是：在已知路面预瞄、执行器约束和安全约束下，如何设计可解释、可部署的主动悬架控制策略。",
        "",
        "因此，本文控制路线组织为：SPDF 传统专家基线、fixed MPC-lite 结构化预瞄控制基线、BP-NMPS 神经 MPC 参数调度器，以及 SI-RPPO 安全教师门控强化学习策略网络。",
        "",
        "## 2. 控制问题与路面预瞄表示",
        "",
        "四轮主动悬架控制器的输入包括车辆状态、上一时刻执行器动作以及固定时域的路面预瞄序列。对于四角全车模型，预瞄 token 可表示为",
        "",
        "```text",
        "P_t[k] = [z_fl, z_fr, z_rl, z_rr]_{t + k dt}",
        "```",
        "",
        "其中 `z_fl,z_fr,z_rl,z_rr` 分别表示前左、前右、后左和后右车轮对应的路面高度。本文的重点不是从图像中恢复该序列，而是研究该序列进入控制器后的使用方式。",
        "",
        "## 3. SPDF 基线",
        "",
        "SPDF 作为已完成的传统反馈控制基线，用于给出不依赖路面预瞄的专家控制参考。论文中应将 SPDF 定位为 classical baseline，而不是本文主要学习型贡献。该基线的作用是帮助比较 fixed MPC-lite、BP-NMPS 和 SI-RPPO 在路面预瞄信息可用时的控制收益与部署代价。",
        "",
        "## 4. BP-NMPS：强化学习辅助的 MPC 参数调度",
        "",
        "BP-NMPS 的核心思想是让神经策略学习 MPC-lite 的参数缩放，而不是直接输出悬架执行器力。策略网络根据路况摘要、基准 MPC-lite 参数和车辆配置输出归一化参数动作：",
        "",
        "```text",
        "a_theta = [a_kz, a_kdz, a_kpitch, a_kdpitch, a_kroll, a_kdroll, a_kpreview]",
        "K_deploy = K_base * scale(a_theta)",
        "u_t = MPC_lite(x_t, P_t; K_deploy)",
        "```",
        "",
        "这种设计保留了 MPC-lite 的结构化和可解释性，同时将强化学习搜索空间从高频力指令空间转移到低维控制参数空间。部署时使用 calibration gate：如果确定性神经参数调度器在校准场景中没有优于 fixed MPC-lite，则部署回退到 fixed MPC-lite。",
        "",
        "当前 formal 5-seed held-out seed sweep 的结果为：",
        "",
        f"- seed count：`{bp.get('seed_count', 'missing')}`；",
        f"- support rate：`{_fmt(bp.get('support_rate'), 3)}`；",
        f"- mean deployed score delta：`{_fmt(bp.get('mean_policy_delta_vs_fixed'), 6)}`；",
        f"- mean unsafe delta：`{_fmt(bp.get('mean_unsafe_delta_vs_fixed'), 3)}`；",
        f"- non-inferiority status：`{bp.get('noninferiority_status', 'missing')}`；",
        f"- improvement status：`{bp.get('improvement_status', 'missing')}`。",
        "",
        "因此，论文中可将 BP-NMPS 表述为保守的自适应 MPC 参数调度器：当校准证据支持时进行小幅 gain 调整，当证据不足时回退 fixed MPC-lite。该结论不应被扩大为“神经策略全面替代 MPC”。",
        "",
        "建议正文引用表：`tab:bp_nmps_seed_sweep`、`tab:bp_nmps_evidence`、`tab:bp_nmps_preview_robustness`。",
        "",
        "## 5. SI-RPPO：安全教师门控的策略网络设计",
        "",
        "SI-RPPO 用于直接力输出强化学习控制。其创新点不是简单使用 PPO，而是改变策略的可部署形式：通过离线模仿初始化、增量动作参数化、投影感知约束和安全教师门控，使 PPO 的主动修正必须在状态相关门控下偏离安全教师。",
        "",
        "部署动作可写为：",
        "",
        "```text",
        "u_t = Projection_U(g_t * DeltaPPO(o_t, u_{t-1}) + (1 - g_t) * u_passive)",
        "```",
        "",
        "其中 `g_t` 为 safe-teacher improvement gate。低激励或低置信场景下，`g_t` 接近 0，策略保持接近安全教师；高激励场景下，`g_t` 打开，允许 PPO 输出主动悬架修正力。",
        "",
        "当前 SI-RPPO 策略网络证据可概括为：",
        "",
        *strategy_rows,
        "",
        "在 preview corruption robustness 矩阵中，SI-RPPO 与主要学习基线的平均差异为：",
        "",
        f"- 相比 PPO scratch：return delta `{_fmt(si_vs_ppo.get('EpisodeReturnDelta'), 1)}`，unsafe delta `{_fmt(si_vs_ppo.get('UnsafeStepsDelta'), 1)}`，action-delta RMS delta `{_fmt(si_vs_ppo.get('ActionDeltaRMS_NDelta'), 1)}`，projection-error delta `{_fmt(si_vs_ppo.get('PolicyProjectionErrorDelta'), 3)}`；",
        f"- 相比 SAC：return delta `{_fmt(si_vs_sac.get('EpisodeReturnDelta'), 1)}`，action-delta RMS delta `{_fmt(si_vs_sac.get('ActionDeltaRMS_NDelta'), 1)}`，projection-error delta `{_fmt(si_vs_sac.get('PolicyProjectionErrorDelta'), 3)}`；",
        f"- 相比 TD3：return delta `{_fmt(si_vs_td3.get('EpisodeReturnDelta'), 1)}`，unsafe delta `{_fmt(si_vs_td3.get('UnsafeStepsDelta'), 1)}`，action-delta RMS delta `{_fmt(si_vs_td3.get('ActionDeltaRMS_NDelta'), 1)}`。",
        "",
        "这些结果支持的表述是：SI-RPPO 显著改善 PPO scratch 的安全性、平滑性和投影可行性；相对 TD3 当前也表现出更好的 return/safety/smoothness；相对 SAC 则是 tradeoff，return 不占优，但动作更平滑且投影误差显著降低。因此，论文中不应声称 SI-RPPO 全面优于 SAC，而应强调其对在线 PPO 可部署性的改造。",
        "",
        "建议正文引用表：`tab:si_rppo_gate_evidence`、`tab:si_rppo_preview_robustness`、`tab:si_rppo_strategy_evidence`。",
        "",
        "## 6. 实验章节组织建议",
        "",
        "实验章节可按如下顺序组织：",
        "",
        "1. 仿真环境与指标：四角全车动力学、路面预瞄、执行器滞后/速率限制、安全约束和 reward/score 组成。",
        "2. SPDF、PASSIVE、fixed MPC-lite 基线：说明传统专家控制与结构化预瞄控制的参考地位。",
        "3. BP-NMPS 参数学习：报告 formal seed sweep、evidence table、gain schedule 和 preview robustness。",
        "4. SI-RPPO 策略网络消融：报告 PPO scratch、BC-PPO、gate evidence、preview robustness 以及 SAC/TD3 对照。",
        "5. 讨论与局限性：强调当前证据支持可部署性和保守改进，不支持所有道路/所有感知误差/所有 RL baseline 下的普遍最优性。",
        "",
        "## 7. 可直接使用的论文结论边界",
        "",
        "本文可 defensible 地声称：",
        "",
        "- 在已感知路面不平度预瞄可用的条件下，BP-NMPS 能以保守方式学习 MPC-lite 参数调度，并通过部署门控避免劣化 fixed MPC-lite。",
        "- SI-RPPO 通过安全教师门控和投影感知策略结构，使在线 PPO 在主动悬架执行器约束下更平滑、更安全、更可部署。",
        "- 当前结果支持直接路面预瞄控制路线，而不是完整视觉重建或 VLA 平台路线。",
        "",
        "本文不应过度声称：",
        "",
        "- BP-NMPS 已经全面替代 MPC。",
        "- SI-RPPO 在所有道路、所有预瞄误差和所有 off-policy baseline 上全面最优。",
        "- 本文完成了端到端视觉-语言-动作主动悬架平台。",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return {"out": str(out_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/thesis_chapter_draft_cn.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_thesis_chapter_draft(out_path=args.out)
    print(f"Saved thesis chapter draft to {result['out']}")


if __name__ == "__main__":
    main()
