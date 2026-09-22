"""Summarize saved experiments without loading Chrono or running policies."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results" / "chrono"
METRICS = (
    "body_acc_rms_mps2",
    "roll_rms_rad",
    "pitch_rms_rad",
    "suspension_travel_peak_m",
    "wheel_unloaded_fraction",
    "actuator_force_rms_n",
    "positive_mechanical_work_j",
    "negative_mechanical_work_magnitude_j",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _mean(values: list[Any]) -> float | None:
    numbers = [_number(value) for value in values]
    if not numbers or any(value is None for value in numbers):
        return None
    return sum(numbers) / len(numbers)


def _default_groups() -> list[dict[str, Any]]:
    return [
        {
            "id": f"{algorithm}_{mode}_seed{seed}",
            "algorithm": algorithm,
            "observation_mode": mode,
            "seed": seed,
        }
        for seed in (42, 43, 44)
        for mode in ("privileged", "sensors")
        for algorithm in ("ppo", "sac")
    ]


def collect_rows(directory: Path, state: dict[str, Any]) -> list[dict[str, Any]]:
    common = int(state.get("common_completed_steps", 0))
    planned = state.get("planned_timesteps")
    cases = state.get("test_cases") or [
        {"id": f"{kind}_{seed}"}
        for kind in ("flat", "bump", "left_bump", "sine", "random")
        for seed in (1001, 1002, 1003, 1004)
    ]
    scheduled = len(cases)
    groups = state.get("groups") or _default_groups()
    controllers = [{"id": name, "algorithm": "baseline"} for name in ("passive", "skyhook", "lqr")]
    controllers.extend(groups)
    rows = []
    for group in controllers:
        label = group["id"]
        case_dir = directory / "evaluation" / f"common_{common}" / label
        summary_path = case_dir / "summary.json"
        if summary_path.is_file():
            summary = _read_json(summary_path)
            source = "summary.json"
        else:
            summary = state.get("evaluation", {}).get(label, {})
            source = "experiment.json" if summary else "not_evaluated"
        records = summary.get("cases", [])
        if not records:
            # A hard stop may leave complete atomic case files without having
            # reached summary publication. Only read the predefined case IDs.
            recovered = [
                _read_json(case_dir / f"{case['id']}.json")
                for case in cases
                if (case_dir / f"{case['id']}.json").is_file()
            ]
            if recovered:
                records = recovered
                source = "case_json_files"
        successful = [
            row
            for row in records
            if row.get("status") == "completed" and not row.get("terminated", False)
        ]
        terminated = [
            row
            for row in records
            if row.get("status") == "terminated" or row.get("terminated", False)
        ]
        if records:
            success_count = len(successful)
            failed_count = len(terminated)
            finished = success_count + failed_count
            metrics = {
                key: _mean([row.get("metrics", {}).get(key) for row in successful])
                for key in METRICS
            }
        else:
            finished = int(summary.get("finished_cases", 0))
            success_count = int(summary.get("successful_cases", 0))
            failed_count = int(summary.get("terminated_cases", max(0, finished - success_count)))
            metrics = {
                key: _number(summary.get("successful_case_mean", {}).get(key)) for key in METRICS
            }
        observed_travel = [
            value
            for row in records
            if (value := _number(row.get("metrics", {}).get("suspension_travel_peak_m")))
            is not None
        ]
        observed_load = [
            value
            for row in records
            if (value := _number(row.get("metrics", {}).get("tire_load_min_n"))) is not None
        ]
        is_baseline = group.get("algorithm") == "baseline"
        latest = None if is_baseline else int(group.get("completed_steps", 0))
        row = {
            "controller": label,
            "algorithm": group.get("algorithm"),
            "observation_mode": "true_state_reference"
            if is_baseline
            else group.get("observation_mode"),
            "training_seed": group.get("seed"),
            "latest_complete_training_steps": latest,
            "evaluated_common_training_steps": None
            if is_baseline or source == "not_evaluated"
            else common,
            "planned_training_steps": None if is_baseline else planned,
            "planned_training_complete": None
            if is_baseline or planned is None
            else latest >= planned,
            "scheduled_cases": scheduled,
            "finished_cases": finished,
            "successful_cases": success_count,
            "failed_cases": failed_count,
            "incomplete_cases": max(0, scheduled - finished),
            "failure_rate_among_finished": failed_count / finished if finished else None,
            "evaluation_complete": finished == scheduled,
            "observed_travel_peak_m_including_failed_cases": max(observed_travel)
            if observed_travel
            else None,
            "observed_tire_load_min_n_including_failed_cases": min(observed_load)
            if observed_load
            else None,
            "summary_source": source,
            "error": summary.get("error", ""),
            **{f"successful_case_mean_{key}": value for key, value in metrics.items()},
        }
        rows.append(row)
    return rows


def _display(value: Any, scale: float = 1.0, suffix: str = "") -> str:
    number = _number(value)
    return "—" if number is None else f"{number * scale:.4g}{suffix}"


def _steps(value: Any) -> str:
    number = _number(value)
    return "—" if number is None else f"{int(number):,}"


def write_report(directory: Path) -> tuple[Path, Path]:
    state = _read_json(directory / "experiment.json")
    rows = collect_rows(directory, state)
    csv_path = directory / "report.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Chrono 主动悬架实验汇总",
        "",
        f"生成时间：{generated}。此报告仅读取现有结果，不启动训练或重新评估。",
        "",
        f"运行状态：`{state.get('status', 'unknown')}`；共同完成训练步数：**{state.get('common_completed_steps', 0)}**；"
        f"每组计划目标：{state.get('planned_timesteps', '尚未完成 pilot')}；请求上限：{state.get('max_timesteps', '未知')}。",
        f"累计计入预算：{_display(state.get('consumed_s'), 1 / 3600)} 小时；总预算：{_display(state.get('budget_hours'))} 小时。",
        "",
        "## 完整性与公平比较",
        "",
        "所有策略的正式评估使用同一个共同完成训练步数。较新的单组 checkpoint 不会替代该比较点。失败率的分母是已结束 case；未完成 case 另列，不能把未运行视为成功。这里的成功仅指完成全时程且未触发终止，不表示满足设计行程限制或从未发生轮胎卸载，须结合物理指标判断。",
        "",
        "| 控制器 | 最新完成步数 | 评估步数 | 已结束/计划 case | 成功 | 失败 | 失败率 | 未完成 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        name = str(row["controller"]).replace("|", "\\|")
        lines.append(
            f"| {name} | {_steps(row['latest_complete_training_steps'])} | {_steps(row['evaluated_common_training_steps'])} | "
            f"{row['finished_cases']}/{row['scheduled_cases']} | {row['successful_cases']} | {row['failed_cases']} | "
            f"{_display(row['failure_rate_among_finished'], 100, '%')} | {row['incomplete_cases']} |"
        )
    lines.extend(
        [
            "",
            "## 物理指标",
            "",
            "下表为成功完成 case 的指标均值，必须与上表的失败率一起阅读，不能只比较该表来排名。行程极值和轮胎最小载荷另列在 CSV，包含可读到的失败 case。缺测值显示为 —，不填零。",
            "",
            "| 控制器 | 车身加速度 RMS (m/s²) | 侧倾 RMS (deg) | 俯仰 RMS (deg) | 行程峰值均值 (mm) | 轮胎卸载比例 | 正机械功 (J) | 负机械功幅值 (J) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        name = str(row["controller"]).replace("|", "\\|")
        cells = [
            _display(row[f"successful_case_mean_{key}"], scale, suffix)
            for key, scale, suffix in (
                ("body_acc_rms_mps2", 1.0, ""),
                ("roll_rms_rad", 180 / math.pi, ""),
                ("pitch_rms_rad", 180 / math.pi, ""),
                ("suspension_travel_peak_m", 1000.0, ""),
                ("wheel_unloaded_fraction", 100.0, "%"),
                ("positive_mechanical_work_j", 1.0, ""),
                ("negative_mechanical_work_magnitude_j", 1.0, ""),
            )
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## 限制与缺失数据",
            "",
            "- 机械功不等于电耗或电池回收能量。",
            "- LQR/skyhook 使用真实模型状态；传感器策略使用带延迟和噪声的测量，信息条件不同。",
            "- 确定性道路上的不同 seed 可能只改变观测噪声，不代表独立道路样本。",
            "- 本报告不自动作收敛、显著性或性能提升声明；成功样本均值可能受失败筛选影响。",
        ]
    )
    incomplete = [row["controller"] for row in rows if not row["evaluation_complete"]]
    incomplete_training = [
        row["controller"] for row in rows if row["planned_training_complete"] is False
    ]
    lines.append(f"- 评估未完成：{', '.join(incomplete) if incomplete else '无'}。")
    lines.append(
        f"- 训练未达到计划目标：{', '.join(incomplete_training) if incomplete_training else '无或尚未确定计划目标'}。"
    )
    for row in rows:
        if row["error"]:
            error = str(row["error"]).replace("\n", " ")
            lines.append(f"- `{row['controller']}` 评估错误：{error}")
    if state.get("error"):
        lines.append(f"- 实验错误：{str(state['error']).replace(chr(10), ' ')}")
    report_path = directory / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", required=True, help="Experiment name, directory, or experiment.json path."
    )
    args = parser.parse_args()
    candidate = Path(args.experiment)
    if not candidate.exists():
        candidate = RESULTS_ROOT / "experiments" / args.experiment
    if candidate.is_file():
        candidate = candidate.parent
    candidate = candidate.resolve()
    if not candidate.is_relative_to(RESULTS_ROOT.resolve()):
        parser.error("Experiment and generated reports must be below results/chrono.")
    if not (candidate / "experiment.json").is_file():
        parser.error("No experiment.json exists at the requested experiment.")
    for path in write_report(candidate):
        print(path)


if __name__ == "__main__":
    main()
