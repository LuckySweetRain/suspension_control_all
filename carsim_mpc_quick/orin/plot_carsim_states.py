"""Live CarSim state dashboard in a separate Matplotlib runtime, plus PNG/PDF export."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

from runtime.paths import FONT_PATH, ROOT
from state_telemetry import FIELDS, StateTelemetry


def read_states(folder):
    path = folder / "carsim_states.csv"
    limited = not path.exists()
    if limited:
        path = folder / "video_telemetry.csv"
    if not path.exists():
        return [], limited
    data = []
    with path.open(encoding="utf-8-sig", newline="") as incoming:
        for row in csv.DictReader(incoming):
            try:
                if limited:
                    actual = dict.fromkeys(FIELDS, float("nan"))
                    actual.update(
                        sim_time_s=float(row["sim_time_s"]),
                        vx_kph=float(row["speed_kph"]),
                        mpc_front_N=float(row["mpc_front_N"]),
                        mpc_rear_N=float(row["mpc_rear_N"]),
                    )
                else:
                    actual = {key: float(row[key]) for key in FIELDS}
                data.append(actual)
            except (ValueError, TypeError, KeyError):
                # The producer may currently be writing its final CSV line.
                continue
    return data, limited


def configure_fonts():
    import matplotlib
    from matplotlib import font_manager

    font = FONT_PATH
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    matplotlib.rcParams["axes.unicode_minus"] = False


def draw(figure, axes, data, title, limited=False):
    if not data:
        figure.suptitle(title + " · 等待数据")
        return
    times = [r["sim_time_s"] for r in data]
    specs = [
        ("CarSim 车速", "km/h", [("vx_kph", "纵向"), ("vy_kph", "侧向")]),
        ("质心垂向位移（相对首个样本）", "mm", [("zcg_m", "ΔZ")]),
        ("车身垂向加速度", "m/s²", [("az_mps2", "Az")]),
        ("车身姿态", "deg", [("pitch_deg", "俯仰"), ("roll_deg", "侧倾")]),
        ("俯仰角速度", "deg/s", [("pitch_rate_deg_s", "AVy")]),
        ("四轮悬架行程", "mm", [("jounce_" + w + "_mm", w) for w in ("L1", "L2", "R1", "R2")]),
        ("质心轨迹（CarSim 坐标）", "Y / m", []),
        ("四轮垂向载荷", "N", [("load_" + w + "_N", w) for w in ("L1", "L2", "R1", "R2")]),
        ("MPC 主动力（控制器输出）", "N", [("mpc_front_N", "前轴"), ("mpc_rear_N", "后轴")]),
    ]
    for ax, (label, unit, series) in zip(axes.flat, specs):
        ax.clear()
        ax.set_title(label, fontsize=10)
        ax.set_ylabel(unit)
        ax.grid(alpha=0.25)
        has_data = False
        if not series:
            x, y = [r["x_m"] for r in data], [r["y_m"] for r in data]
            if any(math.isfinite(v) for v in x):
                ax.plot(x, y, linewidth=1.2)
                ax.set_aspect("equal", adjustable="datalim")
                has_data = True
            ax.set_xlabel("X / m")
        else:
            ax.set_xlabel("仿真时间 / s")
            for key, name in series:
                values = [r[key] for r in data]
                if key == "zcg_m":
                    values = [(v - data[0][key]) * 1000 for v in values]
                if any(math.isfinite(v) for v in values):
                    ax.plot(times, values, label=name, linewidth=1)
                    has_data = True
            ax.set_xlim(0, max(1, times[-1]))
            if has_data:
                ax.legend(loc="upper right", fontsize=8, ncol=2)
        if not has_data:
            ax.text(
                0.5,
                0.5,
                "本段结束后回填完整状态",
                transform=ax.transAxes,
                ha="center",
                color="#64748b",
            )
    note = (
        "本段实时流仅含车速/主动力，完整状态结束后回填"
        if limited
        else "CarSim 状态 100 Hz；绘图在独立进程，不参与控制"
    )
    figure.suptitle(title + f" · {times[-1]:.2f} s\n" + note, fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.94))


def export(folder):
    import matplotlib.pyplot as plt

    if not (folder / "carsim_states.csv").exists():
        trace = json.loads((folder / "trace.json").read_text())
        log = StateTelemetry(folder, "carsim_states.backfill.csv")
        try:
            for row in trace:
                log.write(row["time"], row["values"], row["reply"])
        finally:
            log.close()
        (folder / "carsim_states.backfill.csv").replace(folder / "carsim_states.csv")
    report = json.loads((folder / "report.json").read_text())
    data, limited = read_states(folder)
    figure, axes = plt.subplots(3, 3, figsize=(15, 10))
    draw(figure, axes, data, report["map"].split("/")[-1] + " · CarSim 车辆状态", limited)
    figure.savefig(folder / "carsim_states.png", dpi=160)
    figure.savefig(folder / "carsim_states.pdf")
    plt.close(figure)


def watch(suite, live_folder=None):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 3, figsize=(15, 9))
    figure.canvas.manager.set_window_title("CarSim 车辆状态 · 自动驾驶闭环实时验证")
    completed, last_preview = set(), [0.0]

    def update():
        try:
            if live_folder is not None:
                data, limited = read_states(live_folder)
                draw(figure, axes, data, "CarSim 自动驾驶闭环状态", limited)
                figure.canvas.draw_idle()
                if (live_folder / "report.json").exists() and str(live_folder) not in completed:
                    export(live_folder)
                    completed.add(str(live_folder))
                if data and time.monotonic() - last_preview[0] > 10:
                    figure.savefig(live_folder / "live_state_preview.png", dpi=100)
                    last_preview[0] = time.monotonic()
                return
            manifest = json.loads((suite / "manifest.json").read_text())
            for entry in manifest["runs"]:
                if "artifacts" in entry and entry["artifacts"] not in completed:
                    export(Path(entry["artifacts"]))
                    completed.add(entry["artifacts"])
            entry = manifest["runs"][-1]
            started = entry["started"].replace("-", "").replace(":", "").replace(" ", "_")
            candidates = sorted((ROOT / "results/carla_environment").iterdir(), reverse=True)
            folder = (
                Path(entry["artifacts"])
                if "artifacts" in entry
                else next((p for p in candidates if p.is_dir() and p.name >= started), None)
            )
            if folder is None:
                return
            data, limited = read_states(folder)
            draw(figure, axes, data, entry["town"] + " · CarSim 车辆状态", limited)
            figure.canvas.draw_idle()
            if data and time.monotonic() - last_preview[0] > 10:
                figure.savefig(folder / "live_state_preview.png", dpi=100)
                last_preview[0] = time.monotonic()
        except (OSError, ValueError, KeyError) as error:
            print("Waiting for state stream:", error, flush=True)

    timer = figure.canvas.new_timer(interval=1500)
    timer.add_callback(update)
    timer.start()
    update()
    print("CarSim live state dashboard opened", flush=True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--folder", type=Path)
    parser.add_argument("--live-folder", type=Path)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("TkAgg" if args.suite or args.live_folder else "Agg")
    configure_fonts()
    if args.live_folder:
        watch(None, args.live_folder.resolve())
    elif args.suite:
        watch(args.suite.resolve())
    elif args.folder:
        export(args.folder.resolve())
    else:
        parser.error("Specify --suite for live view or --folder for export")
