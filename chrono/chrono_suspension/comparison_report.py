"""Auditable seven-road baseline reports without importing the Chrono runtime."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROAD_NAMES = {
    "iso_b": "B 级随机路面",
    "iso_c": "C 级随机路面",
    "sine": "正弦",
    "step": "阶跃",
    "pothole": "凹坑",
    "bump": "凸包",
    "composite": "随机 + 凹凸叠加",
}
ROAD_LABELS = ("ISO B", "ISO C", "Sine", "Step", "Pothole", "Bump", "Composite")
CONTROLLERS = ("passive", "skyhook", "lqr")
CONTROLLER_NAMES = {"passive": "被动", "skyhook": "Skyhook", "lqr": "LQR"}
WHEELS = ("FL", "FR", "RL", "RR")
SCALAR_METRICS = (
    "body_acc_rms_mps2",
    "body_acc_peak_mps2",
    "roll_rms_rad",
    "roll_peak_rad",
    "pitch_rms_rad",
    "pitch_peak_rad",
    "suspension_travel_rms_m",
    "suspension_travel_peak_m",
    "tire_load_min_n",
    "tire_load_min_ratio",
    "tire_dynamic_load_rms_ratio",
    "wheel_unloaded_fraction",
    "actuator_force_rms_n",
    "positive_mechanical_work_j",
    "negative_mechanical_work_magnitude_j",
    "absolute_mechanical_work_j",
    "net_mechanical_work_j",
)
ARRAY_METRICS = (
    "suspension_travel_rms_per_wheel_m",
    "suspension_travel_peak_per_wheel_m",
    "suspension_travel_min_per_wheel_m",
    "suspension_travel_max_per_wheel_m",
    "tire_load_min_per_wheel_n",
    "tire_load_std_n_per_wheel",
    "tire_load_min_ratio_per_wheel",
    "tire_dynamic_load_rms_ratio_per_wheel",
    "wheel_unloaded_fraction_per_wheel",
    "positive_mechanical_work_per_wheel_j",
    "negative_mechanical_work_magnitude_per_wheel_j",
)


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float, np.number)) and math.isfinite(float(value)):
        return float(value)
    return None


def _array(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 4:
        return None
    numbers = [_number(item) for item in value]
    return None if any(item is None for item in numbers) else numbers


def _metric(row: dict, key: str) -> float | None:
    metrics = row.get("metrics", {})
    direct = _number(metrics.get(key))
    if direct is not None:
        return direct
    if key in ("absolute_mechanical_work_j", "net_mechanical_work_j"):
        positive = _number(metrics.get("positive_mechanical_work_j"))
        negative = _number(metrics.get("negative_mechanical_work_magnitude_j"))
        if positive is not None and negative is not None:
            return positive + negative if key.startswith("absolute") else positive - negative
    if key in ("tire_load_min_ratio", "tire_dynamic_load_rms_ratio"):
        values = _array(metrics.get(key + "_per_wheel"))
        if values is not None:
            return min(values) if key == "tire_load_min_ratio" else max(values)
    return None


def _fmt(value: Any, scale: float = 1.0, digits: int = 3) -> str:
    number = _number(value)
    return "缺失" if number is None else f"{number * scale:.{digits}f}"


def _mean(values: list[float | None]) -> float | None:
    # Do not silently average an incomplete set of roads, cases or wheels.
    if not values or any(value is None for value in values):
        return None
    return float(np.mean(values))


def _summary(rows: list[dict], controller: str, road: str, threshold: float) -> dict:
    successful = [row for row in rows if row.get("status") == "completed"]
    travel_values = [_metric(row, "suspension_travel_peak_m") for row in rows]
    unload_values = [_metric(row, "wheel_unloaded_fraction") for row in rows]
    result = {
        "controller": controller,
        "road": road,
        "cases": len(rows),
        "completed": len(successful),
        "failed_or_incomplete": len(rows) - len(successful),
        "travel_threshold_exceeded_cases": (
            sum(value > threshold for value in travel_values)
            if rows and all(value is not None for value in travel_values)
            else None
        ),
        "wheel_unloaded_cases": (
            sum(value > 0.0 for value in unload_values)
            if rows and all(value is not None for value in unload_values)
            else None
        ),
    }
    for key in SCALAR_METRICS:
        if road == "all":
            values = [
                _mean([_metric(row, key) for row in successful if row["case"]["kind"] == kind])
                for kind in ROAD_NAMES
            ]
        else:
            values = [_metric(row, key) for row in successful]
        result["mean_" + key] = _mean(values)
        observed = [_metric(row, key) for row in rows]
        finite = [value for value in observed if value is not None]
        result["valid_" + key] = len(finite)
        result["worst_" + key] = (
            (min(finite) if key in ("tire_load_min_n", "tire_load_min_ratio") else max(finite))
            if finite
            else None
        )
    return result


def _paired_improvement(rows: list[dict], baseline: list[dict], key: str) -> tuple[Any, int]:
    passive = {row["case"]["id"]: row for row in baseline if row.get("status") == "completed"}
    road_ratios = []
    paired_count = 0
    for kind in ROAD_NAMES:
        pairs = [
            (row, passive[row["case"]["id"]])
            for row in rows
            if row.get("status") == "completed"
            and row["case"]["kind"] == kind
            and row["case"]["id"] in passive
        ]
        selected_mean = _mean([_metric(row, key) for row, _ in pairs])
        passive_mean = _mean([_metric(base, key) for _, base in pairs])
        if selected_mean is None or passive_mean is None or passive_mean <= 1e-12:
            return None, sum(
                row.get("status") == "completed" and row["case"]["id"] in passive for row in rows
            )
        road_ratios.append(100.0 * (1.0 - selected_mean / passive_mean))
        paired_count += len(pairs)
    return _mean(road_ratios), paired_count


def _write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _case_csv(test_results: dict, path: Path) -> None:
    flat = []
    for controller in CONTROLLERS:
        for row in test_results.get(controller, []):
            item = {
                "controller": controller,
                "case_id": row["case"]["id"],
                "road": row["case"]["kind"],
                "status": row.get("status"),
                "termination_reason": row.get("termination_reason"),
                "duration_s": row.get("duration_s"),
                "speed_mps": row["case"].get("speed_mps"),
                "case_json": json.dumps(row["case"], ensure_ascii=False, sort_keys=True),
            }
            item.update({key: _metric(row, key) for key in SCALAR_METRICS})
            for key in ARRAY_METRICS:
                values = _array(row.get("metrics", {}).get(key))
                for index, wheel in enumerate(WHEELS):
                    item[f"{key}_{wheel}"] = None if values is None else values[index]
            flat.append(item)
    _write_csv(path, flat)


def _make_plot(summaries: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {(row["controller"], row["road"]): row for row in summaries}
    panels = (
        ("body_acc_rms_mps2", 1.0, "Body vertical acceleration RMS", r"m/s$^2$"),
        ("pitch_rms_rad", 180.0 / math.pi, "Pitch RMS", "deg"),
        ("roll_rms_rad", 180.0 / math.pi, "Roll RMS", "deg"),
        ("suspension_travel_peak_m", 1000.0, "Suspension peak travel (mean of cases)", "mm"),
        ("wheel_unloaded_fraction", 100.0, "Wheel-time fraction unloaded", "%"),
        ("absolute_mechanical_work_j", 1.0, "Actuator absolute mechanical work", "J / case"),
    )
    colors = ("#687686", "#0072B2", "#D55E00")
    hatches = ("", "//", "..")
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(3, 2, figsize=(15, 11), layout="constrained")
        x = np.arange(len(ROAD_NAMES))
        for ax, (key, scale, title, unit) in zip(axes.flat, panels, strict=True):
            for offset, controller in enumerate(CONTROLLERS):
                heights = [lookup[(controller, road)].get("mean_" + key) for road in ROAD_NAMES]
                values = [np.nan if value is None else value * scale for value in heights]
                bars = ax.bar(
                    x + (offset - 1) * 0.25,
                    values,
                    width=0.24,
                    label=("Passive", "Skyhook", "LQR")[offset],
                    color=colors[offset],
                    hatch=hatches[offset],
                    edgecolor="white",
                    linewidth=0.5,
                )
                for index, bar in enumerate(bars):
                    failed = lookup[(controller, tuple(ROAD_NAMES)[index])]["failed_or_incomplete"]
                    missing = not np.isfinite(bar.get_height())
                    if failed or missing:
                        label = f"!{failed}" if failed else ""
                        if missing:
                            label = "N/A " + label
                        ax.annotate(
                            label,
                            (bar.get_x() + bar.get_width() / 2, 0 if missing else bar.get_height()),
                            xytext=(0, 2),
                            textcoords="offset points",
                            ha="center",
                            fontsize=7,
                            rotation=90 if missing else 0,
                            color="#A32638",
                        )
            ax.set_title(title, loc="left", fontweight="bold")
            ax.set_ylabel(unit)
            ax.set_xticks(x, ROAD_LABELS, rotation=18)
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=0.22)
            ax.set_axisbelow(True)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper right", ncols=3, frameon=False)
        fig.suptitle(
            "CarSim D-Class parameters / Chrono seven-DOF baseline comparison",
            x=0.01,
            ha="left",
            fontsize=15,
            fontweight="bold",
        )
        fig.supxlabel(
            "Test cases only; each bar averages completed cases of one road. "
            "!n = failed/incomplete cases; consult report before comparing.",
            fontsize=10,
        )
        fig.savefig(path, dpi=200, facecolor="white")
        fig.savefig(path.with_suffix(".svg"), facecolor="white")
        plt.close(fig)


def _representative_cases(data: dict) -> dict[str, dict]:
    """Choose from predeclared inputs only; never inspect response magnitudes."""
    chosen = {}
    length_keys = data.get("design", {}).get("length_parameter_by_road", {})
    for kind in ROAD_NAMES:
        cases = [case for case in data.get("test_cases", []) if case.get("kind") == kind]
        if not cases:
            continue
        if kind in ("iso_b", "iso_c"):
            chosen[kind] = min(
                cases, key=lambda case: (case["seed"], -case["speed_mps"], case["id"])
            )
        else:
            length_key = length_keys.get(
                kind,
                {"sine": "wavelength_m", "step": "transition_length_m"}.get(kind, "length_m"),
            )
            chosen[kind] = min(
                cases,
                key=lambda case: (
                    -case["height_m"],
                    case[length_key],
                    -case["speed_mps"],
                    case["id"],
                ),
            )
    return chosen


def _timeseries_subtitle(case: dict) -> str:
    details = [f"speed={case['speed_mps']:g} m/s", f"seed={case['seed']}"]
    if case["kind"] not in ("iso_b", "iso_c"):
        details.insert(0, f"height={case['height_m'] * 1000:g} mm")
        length_key = {"sine": "wavelength_m", "step": "transition_length_m"}.get(
            case["kind"], "length_m"
        )
        details.insert(1, f"{length_key.removesuffix('_m')}={case[length_key]:g} m")
    if case["kind"] == "composite":
        sides = ", ".join(
            f"{component['kind']}:{component.get('tracks', 'both')}"
            for component in case.get("components", [])
            if component["kind"] in ("bump", "pothole")
        )
        details.append("local sides=" + sides)
    else:
        details.append(f"tracks={case.get('tracks', 'both')}")
    return "; ".join(details)


def _load_timeseries(data: dict, output_dir: Path, case: dict) -> tuple[dict, list[str]]:
    signals, errors = {}, []
    for controller in CONTROLLERS:
        selected = data.get("selected_controllers", {}).get(controller, {})
        candidate_id = selected.get("id")
        row = next(
            (
                r
                for r in data.get("test_results", {}).get(controller, [])
                if r["case"]["id"] == case["id"]
            ),
            None,
        )
        if candidate_id is None or row is None:
            errors.append(f"{controller}: 缺少冻结控制器或工况记录")
            continue
        path = output_dir / "test" / candidate_id / f"{case['id']}.npz"
        if not path.is_file():
            errors.append(f"{controller}: 缺少 NPZ 时序")
            continue
        try:
            with np.load(path, allow_pickle=False) as archive:
                values = {
                    key: np.asarray(archive[key], dtype=float)
                    for key in (
                        "time_s",
                        "body_acc_mps2",
                        "pitch_rad",
                        "roll_rad",
                        "suspension_travel_m",
                        "tire_load_n",
                        "mechanical_power_w",
                    )
                }
            time_values = values["time_s"]
            if time_values.ndim != 1 or time_values.size == 0:
                raise ValueError("时间轴为空或维度错误")
            if np.any(np.diff(time_values) <= 0):
                raise ValueError("时间轴没有严格递增")
            for key, value in values.items():
                expected = (
                    (time_values.size, 4)
                    if key in ("suspension_travel_m", "tire_load_n", "mechanical_power_w")
                    else time_values.shape
                )
                if value.shape != expected or not np.isfinite(value).all():
                    raise ValueError(f"{key} 维度错误或包含非有限值")
            loads = _array(row.get("static_tire_load_n"))
            if loads is None or min(loads) <= 0:
                raise ValueError("缺少有效四轮静载")
            signals[controller] = {
                "time": time_values,
                "status": row.get("status"),
                "panels": (
                    values["body_acc_mps2"],
                    np.rad2deg(values["pitch_rad"]),
                    np.rad2deg(values["roll_rad"]),
                    np.max(np.abs(values["suspension_travel_m"]), axis=1) * 1000,
                    np.min(values["tire_load_n"] / np.asarray(loads), axis=1),
                    np.sum(values["mechanical_power_w"], axis=1),
                ),
            }
        except (OSError, KeyError, ValueError) as exc:
            errors.append(f"{controller}: 时序不可用（{exc}）")
    return signals, errors


def _make_timeseries(data: dict, output_dir: Path) -> tuple[dict[str, Path], list[str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths, notes = {}, []
    selected_cases = _representative_cases(data)
    panels = (
        ("Body vertical acceleration", r"m/s$^2$"),
        ("Pitch", "deg"),
        ("Roll", "deg"),
        ("Max |suspension travel| across wheels", "mm"),
        ("Min tire load / static load across wheels", "ratio"),
        ("Sum of actuator mechanical power", "W"),
    )
    colors, styles = ("#687686", "#0072B2", "#D55E00"), ("-", "--", "-.")
    for kind, name in ROAD_NAMES.items():
        case = selected_cases.get(kind)
        if case is None:
            notes.append(f"- {name}：图不可用，缺少预先指定的测试工况。")
            continue
        signals, errors = _load_timeseries(data, output_dir, case)
        if errors:
            notes.append(f"- {name}（`{case['id']}`）：图不可用；" + "；".join(errors) + "。")
            continue
        path = output_dir / "timeseries" / f"{kind}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        with plt.rc_context(
            {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
        ):
            fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, layout="constrained")
            for panel_index, (ax, (title, unit)) in enumerate(zip(axes.flat, panels, strict=True)):
                for index, controller in enumerate(CONTROLLERS):
                    record = signals[controller]
                    label = ("Passive", "Skyhook", "LQR")[index]
                    if record["status"] != "completed":
                        label += " (" + str(record["status"]) + ")"
                    ax.plot(
                        record["time"],
                        record["panels"][panel_index],
                        color=colors[index],
                        linestyle=styles[index],
                        linewidth=1.1,
                        label=label,
                    )
                ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
                ax.set_ylabel(unit)
                ax.grid(alpha=0.22)
                ax.set_axisbelow(True)
                ax.set_xlim(0, data.get("config", {}).get("physics", {}).get("episode_seconds", 10))
                if (
                    panel_index in (1, 2)
                    and max(
                        float(np.max(np.abs(record["panels"][panel_index])))
                        for record in signals.values()
                    )
                    < 1e-8
                ):
                    # Symmetric roads leave roundoff-sized attitude residuals;
                    # keep raw signals while avoiding an amplified noise plot.
                    ax.set_ylim(-1e-4, 1e-4)
                    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
                    ax.text(
                        0.02,
                        0.94,
                        "Numerically zero (< 1e-8 deg)",
                        transform=ax.transAxes,
                        va="top",
                        fontsize=9,
                        color="#555555",
                    )
            axes[1, 0].axhline(
                data.get("config", {}).get("vehicle", {}).get("travel_limit_m", 0.09) * 1000,
                color="#666666",
                linewidth=0.8,
                linestyle=":",
                label="Research travel threshold",
            )
            axes[1, 1].axhline(0, color="#555555", linewidth=0.8, linestyle=":")
            for ax in axes[1]:
                ax.set_xlabel("Time / s")
            handles, labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="outside upper right", ncols=3, frameon=False)
            fig.suptitle(
                case["id"] + "\n" + _timeseries_subtitle(case),
                x=0.01,
                ha="left",
                fontsize=12,
                fontweight="bold",
            )
            sampling_ms = data.get("sampling", {}).get("time_series_dt_s", 0.01) * 1000
            fig.supxlabel(
                f"Preselected input case, not the worst response. Display samples every {sampling_ms:g} ms; "
                "substep summary metrics retain finer peaks. Dashed gray travel line = research threshold.",
                fontsize=9,
            )
            fig.savefig(path, dpi=180, facecolor="white")
            plt.close(fig)
        paths["timeseries_" + kind] = path
        notes.append(f"- [{name}：`{case['id']}`](timeseries/{kind}.png)")
    return paths, notes


def _wheel_table(test_results: dict) -> list[str]:
    lines = [
        "## 四轮指标",
        "",
        "轮序为 FL/FR/RL/RR。RMS、卸载占比、机械功按七类道路等权汇总；"
        "行程峰值与最小轮荷采用所有已观测工况的最坏值，包含提前终止的已观测部分。",
        "",
        "|控制器|轮位|行程 RMS / mm|行程峰值 / mm|最小轮荷 / N|动载比 RMS|卸载占比 / %|正功 / J|负功幅值 / J|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    columns = (
        ("suspension_travel_rms_per_wheel_m", 1000.0, "mean"),
        ("suspension_travel_peak_per_wheel_m", 1000.0, "max"),
        ("tire_load_min_per_wheel_n", 1.0, "min"),
        ("tire_dynamic_load_rms_ratio_per_wheel", 1.0, "mean"),
        ("wheel_unloaded_fraction_per_wheel", 100.0, "mean"),
        ("positive_mechanical_work_per_wheel_j", 1.0, "mean"),
        ("negative_mechanical_work_magnitude_per_wheel_j", 1.0, "mean"),
    )
    for controller in CONTROLLERS:
        rows = test_results.get(controller, [])
        for wheel_index, wheel in enumerate(WHEELS):
            cells = [CONTROLLER_NAMES[controller], wheel]
            for key, scale, operation in columns:
                values = []
                for row in rows:
                    array = _array(row.get("metrics", {}).get(key))
                    if array is not None:
                        values.append(array[wheel_index])
                if operation == "mean":
                    road_means = []
                    for kind in ROAD_NAMES:
                        selected = [
                            _array(row.get("metrics", {}).get(key))
                            for row in rows
                            if row["case"]["kind"] == kind and row.get("status") == "completed"
                        ]
                        road_means.append(
                            _mean([None if a is None else a[wheel_index] for a in selected])
                        )
                    value = _mean(road_means)
                else:
                    value = (min(values) if operation == "min" else max(values)) if values else None
                cells.append(_fmt(value, scale))
            lines.append("|" + "|".join(cells) + "|")
    return lines


def _design_section(data: dict) -> list[str]:
    design = data.get("design", {})
    factors = design.get("factor_levels", {})
    lengths = design.get("length_parameter_by_road", {})
    lines = [
        "## 验证集选择与试验设计",
        "",
        "正弦、阶跃、凹坑、凸包的多样性来自幅值、空间长度和车速，不能由更换随机种子替代。"
        "验证集使用 2 × 2 × 2 因子的一半组合，每类 4 工况；测试集使用完整组合，每类 8 工况。"
        "B/C 随机路面分别使用验证集 2 个、测试集 4 个独立路面实现，每个实现以两个车速通过。"
        "全部参数在运行前保存于 [plan.json](plan.json)。",
        "",
        "|道路|验证幅值 / mm|测试幅值 / mm|空间长度含义|验证长度 / m|测试长度 / m|",
        "|---|---|---|---|---|---|",
    ]

    def levels(split: str, key: str, scale: float = 1.0) -> str:
        values = factors.get(split, {}).get(key)
        return "缺失" if values is None else "、".join(f"{v * scale:g}" for v in values)

    for kind in ("sine", "step", "pothole", "bump", "composite"):
        length_key = lengths.get(kind, "length_m")
        length_label = {
            "wavelength_m": "波长",
            "transition_length_m": "有限阶跃过渡长度",
            "length_m": "单个局部凹坑/凸包长度",
        }.get(length_key, length_key)
        lines.append(
            "|"
            + "|".join(
                [
                    ROAD_NAMES[kind],
                    levels("validation", "height_m", 1000),
                    levels("test", "height_m", 1000),
                    length_label,
                    levels("validation", length_key),
                    levels("test", length_key),
                ]
            )
            + "|"
        )
    lines.extend(
        [
            "",
            f"各道路验证车速：{levels('validation', 'speed_mps')} m/s；"
            f"测试车速：{levels('test', 'speed_mps')} m/s。"
            "单边/双边激励预先固定：一半工况为双边，其余左右各半；侧别没有完整交叉，"
            "因此此矩阵适合总体控制器比较，不能单独辨识侧别与全部因子的交互效应。"
            "侧别与具体因素组合相关，不能用此矩阵独立估计幅值、长度、车速的因果主效应；"
            "同工况控制器之间仍共享完全相同的路面与车速。"
            "叠加路面为 B 级背景加凸包及凹坑，局部事件改变幅值、长度、车速和作用侧；"
            "单边叠加工况的凸包与凹坑位于相反侧。阶跃有有限平滑过渡，并保持新高度。",
            "",
            "验证集选择规则：先比较失败数量，再比较预先固定的工程综合分数；"
            "每个控制器族选择一组全局参数，测试结果不参与选择。综合分数为：",
            "",
            "```text",
            data.get("selection_protocol", {}).get("score_formula", "缺失：未记录调参目标"),
            "```",
            "",
            "|候选|验证工况数|失败数|验证综合分数|选中|",
            "|---|---:|---:|---:|---|",
        ]
    )
    selected = {value.get("id") for value in data.get("selected_controllers", {}).values()}
    for name, score in data.get("validation_scores", {}).items():
        lines.append(
            f"|{name}|{score.get('case_count', '缺失')}|{score.get('failed_cases', '缺失')}|"
            f"{_fmt(score.get('score'), digits=6)}|{'是' if name in selected else ''}|"
        )
    if not data.get("validation_scores"):
        lines.append("|缺失：没有验证候选分数|||||")
    lines.extend(
        [
            "",
            "冻结参数与对应验证分数：",
            "",
            "```json",
            json.dumps(data.get("selected_controllers", {}), ensure_ascii=False, indent=2),
            "```",
            "",
            f"冻结时间（UTC）：`{data.get('frozen_utc', '缺失')}`。"
            f"冻结记录 SHA-256：`{data.get('freeze_sha256', '缺失')}`。"
            "完整因子组合与种子见 [plan.json](plan.json)，完整冻结记录见 "
            "[frozen_controllers.json](frozen_controllers.json)。",
        ]
    )
    return lines


def write_comparison_report(data: dict, output_dir: Path) -> dict[str, Path]:
    """Write report/CSV/PNG/SVG from stored results, without rerunning or tuning."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        name: output_dir / filename
        for name, filename in (
            ("report", "report.md"),
            ("summary", "summary.csv"),
            ("cases", "cases.csv"),
            ("plot", "comparison.png"),
            ("svg", "comparison.svg"),
        )
    }
    test_results = data.get("test_results", {})
    threshold = float(data.get("config", {}).get("vehicle", {}).get("travel_limit_m", 0.09))
    expected = data.get("test_cases", [])
    expected_ids = {case["id"] for case in expected}
    integrity_warnings = []
    summaries = []
    for controller in CONTROLLERS:
        rows = test_results.get(controller, [])
        observed_ids = {row["case"]["id"] for row in rows}
        if len(observed_ids) != len(rows):
            raise ValueError(f"Duplicate case IDs in {controller} test results")
        if expected_ids:
            for case_id in sorted(expected_ids - observed_ids):
                integrity_warnings.append(f"{controller}/{case_id}: 计划工况尚无结果")
            for case_id in sorted(observed_ids - expected_ids):
                integrity_warnings.append(f"{controller}/{case_id}: 结果不在计划工况内")
        for road in ("all", *ROAD_NAMES):
            selected = rows if road == "all" else [r for r in rows if r["case"]["kind"] == road]
            summary = _summary(selected, controller, road, threshold)
            planned = [case for case in expected if road == "all" or case["kind"] == road]
            missing_count = len({case["id"] for case in planned} - observed_ids)
            summary.update(
                planned_cases=len(planned) if expected else None,
                missing_planned_cases=missing_count,
            )
            summary["failed_or_incomplete"] += missing_count
            summaries.append(summary)
    _write_csv(paths["summary"], summaries)
    _case_csv(test_results, paths["cases"])
    _make_plot(summaries, paths["plot"])
    timeseries_paths, timeseries_notes = _make_timeseries(data, output_dir)
    paths.update(timeseries_paths)
    lookup = {(row["controller"], row["road"]): row for row in summaries}
    lines = [
        "# 七类道路上的被动、Skyhook、LQR 对比",
        "",
        "本报告使用 CarSim D-Class 参数构建的 Chrono 严格七自由度模型。"
        "控制器候选仅使用验证集选择；以下物理指标来自参数冻结后的独立测试集。",
        "",
        "## 汇总口径与完整性",
        "",
        "先对每类道路的完整工况求均值，再对七类道路等权平均。"
        "均值不包含提前终止或未完成工况，其数量在表中单列；"
        "缺少某类道路或必要指标时整体均值标为“缺失”，不以零填充。"
        "峰值、最小轮荷、超限及卸载工况计数覆盖全部已观测工况，包含终止前的数据。",
        "物理汇总采样间隔为 "
        f"{_fmt(data.get('sampling', {}).get('metrics_dt_s'), 1000)} ms；"
        "保存的时序 NPZ 采样间隔为 "
        f"{_fmt(data.get('sampling', {}).get('time_series_dt_s'), 1000)} ms。"
        "汇总采用整个 10 秒片段，包括静止进入路面阶段与事件后的衰减响应；初始 t=0 点不参与积分。",
        "局部凹坑、凸包、阶跃的整窗 RMS 包含事件前后静稳段，不能作为事件峰值，"
        "也不能直接用于判断其与持续随机或正弦路面的输入严重程度。各道路另列响应峰值。",
        "",
        "|控制器|完整 / 已运行|失败或未完成|加速度 RMS / m/s²|俯仰 RMS / °|侧倾 RMS / °|行程 RMS / mm|最大行程 / mm|卸载占比 / %|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for controller in CONTROLLERS:
        s = lookup[(controller, "all")]
        values = [
            CONTROLLER_NAMES[controller],
            f"{s['completed']} / {s['cases']}",
            str(s["failed_or_incomplete"]),
        ]
        for key, scale, aggregation in (
            ("body_acc_rms_mps2", 1.0, "mean"),
            ("pitch_rms_rad", 180 / math.pi, "mean"),
            ("roll_rms_rad", 180 / math.pi, "mean"),
            ("suspension_travel_rms_m", 1000.0, "mean"),
            ("suspension_travel_peak_m", 1000.0, "worst"),
            ("wheel_unloaded_fraction", 100.0, "mean"),
        ):
            values.append(_fmt(s[aggregation + "_" + key], scale))
        lines.append("|" + "|".join(values) + "|")
    lines.extend(
        [
            "",
            "![七类道路对比](comparison.png)",
            "",
            "## 相对被动的改善",
            "",
            "仅使用同一工况下两控制器均完整完成的配对结果。"
            "改善率为每类道路的 `100 × (1 − 控制器均值 / 被动均值)`，再对七类等权平均；"
            "正值表示降低，负值表示变差。被动均值接近零时不定义相对改善率。"
            "这与先求总均值再相除不同，也不代表统计显著性。",
            "",
            "|控制器|配对工况数|加速度改善 / %|俯仰改善 / %|侧倾改善 / %|行程 RMS 改善 / %|",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for controller in ("skyhook", "lqr"):
        outcomes = [
            _paired_improvement(
                test_results.get(controller, []), test_results.get("passive", []), key
            )
            for key in (
                "body_acc_rms_mps2",
                "pitch_rms_rad",
                "roll_rms_rad",
                "suspension_travel_rms_m",
            )
        ]
        lines.append(
            "|"
            + "|".join(
                [
                    CONTROLLER_NAMES[controller],
                    str(outcomes[0][1]),
                    *[_fmt(v, digits=2) for v, _ in outcomes],
                ]
            )
            + "|"
        )
    lines.extend(
        [
            "",
            "## 各道路的车身响应",
            "",
            "加速度为重力补偿后的车身质心垂向惯性加速度，未做 ISO 2631 频率加权。"
            "姿态相对初始化平衡姿态；以下 RMS 为完整工况均值，峰值为已观测最坏值。",
            "",
            "|道路|控制器|加速度 RMS / m/s²|加速度峰值 / m/s²|俯仰 RMS / °|俯仰峰值 / °|侧倾 RMS / °|侧倾峰值 / °|",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for road, name in ROAD_NAMES.items():
        for controller in CONTROLLERS:
            s = lookup[(controller, road)]
            values = [name, CONTROLLER_NAMES[controller]]
            for key, scale in (
                ("body_acc", 1.0),
                ("pitch", 180 / math.pi),
                ("roll", 180 / math.pi),
            ):
                unit = "mps2" if key == "body_acc" else "rad"
                values.extend(
                    [
                        _fmt(s[f"mean_{key}_rms_{unit}"], scale),
                        _fmt(s[f"worst_{key}_peak_{unit}"], scale),
                    ]
                )
            lines.append("|" + "|".join(values) + "|")
    lines.extend(
        [
            "",
            "## 悬架行程与轮胎卸载",
            "",
            f"研究行程阈值为 {threshold * 1000:g} mm，不等同于非线性限位块开始接触的位置。"
            "卸载定义为 `Fz ≤ 0`；卸载占比为四轮、全仿真时段的轮时比例。"
            "最小轮荷比取 `min(Fz/Fz_static)`；动载比 RMS 为 `sqrt(mean((Fz/Fz_static − 1)²))`，"
            "表中先取每工况最差轮位再求均值。",
            "",
            "|道路|控制器|完整 / 运行|行程 RMS / mm|最大行程 / mm|超阈值工况|最小轮荷 / %静载|动载比 RMS|卸载占比 / %|发生卸载工况|",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for road in ("all", *ROAD_NAMES):
        for controller in CONTROLLERS:
            s = lookup[(controller, road)]
            values = [
                "整体" if road == "all" else ROAD_NAMES[road],
                CONTROLLER_NAMES[controller],
                f"{s['completed']} / {s['cases']}",
                _fmt(s["mean_suspension_travel_rms_m"], 1000),
                _fmt(s["worst_suspension_travel_peak_m"], 1000),
                _fmt(s["travel_threshold_exceeded_cases"], digits=0),
                _fmt(s["worst_tire_load_min_ratio"], 100),
                _fmt(s["mean_tire_dynamic_load_rms_ratio"]),
                _fmt(s["mean_wheel_unloaded_fraction"], 100),
                _fmt(s["wheel_unloaded_cases"], digits=0),
            ]
            lines.append("|" + "|".join(values) + "|")
    lines.extend(
        [
            "",
            "## 执行器机械功",
            "",
            "所有功均为四个主动执行器之和，并对完整工况求均值。"
            "正功 `W+ = ∫max(P,0)dt`，负功幅值 `W− = −∫min(P,0)dt`，"
            "绝对功 `Wabs = W+ + W−`，净功 `Wnet = W+ − W−`。"
            "净功会抵消正负功；负机械功不等于可回收电能，这些指标也不是电池耗电量。",
            "",
            "|道路|控制器|正功 / J|负功幅值 / J|绝对功 / J|净功 / J|",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for road in ("all", *ROAD_NAMES):
        for controller in CONTROLLERS:
            s = lookup[(controller, road)]
            values = ["整体" if road == "all" else ROAD_NAMES[road], CONTROLLER_NAMES[controller]]
            values.extend(
                _fmt(s["mean_" + key])
                for key in (
                    "positive_mechanical_work_j",
                    "negative_mechanical_work_magnitude_j",
                    "absolute_mechanical_work_j",
                    "net_mechanical_work_j",
                )
            )
            lines.append("|" + "|".join(values) + "|")
    lines.extend(
        [
            "",
            *_wheel_table(test_results),
            "",
            "## 固定代表工况的响应时序",
            "",
            "代表工况只按运行前的输入参数选择，不查看控制效果：B/C 选择最小随机种子与最高车速；"
            "正弦、阶跃、凹坑、凸包及叠加选择最高幅值、最短对应长度、最高车速，"
            "本设计均为 `a1_l0_v1`。这些是固定代表输入，不是响应最差的样本。"
            "每图同时显示三种控制器的车身加速度、俯仰、侧倾、四轮最大绝对行程、"
            "四轮最小静载归一化轮荷、四执行器机械功率之和。",
            "",
            "时序使用 10 ms 控制采样；图中峰值及接触失效持续时间可能低估 0.5 ms 物理子步结果。"
            "这些图仅用于查看冲击与衰减过程，不替代前述子步积分、峰值和卸载统计。"
            "浅灰虚线为研究行程阈值，功率保留符号，负值表示执行器吸收机械能。",
            "",
            *timeseries_notes,
            "",
            "## 失败、缺失与可复核数据",
            "",
        ]
    )
    failures = [
        (controller, row)
        for controller in CONTROLLERS
        for row in test_results.get(controller, [])
        if row.get("status") != "completed"
    ]
    if failures:
        lines.extend(["|控制器|工况|状态|终止原因|已模拟 / s|", "|---|---|---|---|---:|"])
        for controller, row in failures:
            reason = str(
                row.get("termination_reason") or row.get("metric_error") or "未提供"
            ).replace("|", "/")
            lines.append(
                f"|{CONTROLLER_NAMES[controller]}|{row['case']['id']}|{row.get('status')}|{reason}|{_fmt(row.get('duration_s'))}|"
            )
    else:
        lines.append("已记录的测试工况均完整结束，没有提前终止记录。")
    missing = integrity_warnings
    for controller in CONTROLLERS:
        for key in SCALAR_METRICS:
            count = sum(_metric(row, key) is None for row in test_results.get(controller, []))
            if count:
                missing.append(f"{controller}/{key}: {count} 个工况缺失")
        for road in ROAD_NAMES:
            if lookup[(controller, road)]["cases"] == 0:
                missing.append(f"{controller}/{road}: 无已记录工况")
    if missing:
        lines.extend(["", "缺失字段不会以零代替：", "", *[f"- {item}" for item in missing]])
    lines.extend(
        [
            "",
            "[summary.csv](summary.csv) 保留 SI 单位、七类等权整体均值、各道路均值及最坏值。"
            "[cases.csv](cases.csv) 保留逐工况四轮指标和路面参数 JSON。"
            "[comparison.json](comparison.json) 保留完整数值、配置、调参记录与冻结参数。",
            "",
            *_design_section(data),
            "",
            "## 解释边界",
            "",
            "这是在等效七自由度模型上的有限工况比较，不是 CarSim 完整多体模型的动态一致性证明。"
            "七自由度包括车身升沉、俯仰、侧倾与四轮垂向运动，车速预设，不包含转向、制动、"
            "纵横向轮胎耦合或悬架完整几何运动学。Skyhook 与 LQR 使用模型真值状态反馈；"
            "结果不代表包含状态估计误差的实车闭环表现，也不与噪声传感器 RL 构成同信息条件的比较。"
            "LQR 为平衡点线性化设计，仿真包含非线性减振器、限位、防倾杆、单边轮胎接触和执行器限制。"
            "验证集最优仅指预先给定候选及目标函数中的最优，不能证明全局最优或未见道路上的性能。",
            "",
        ]
    )
    paths["report"].write_text("\n".join(lines), encoding="utf-8")
    return paths
