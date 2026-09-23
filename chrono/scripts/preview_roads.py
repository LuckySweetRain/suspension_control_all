"""Export seven reproducible four-wheel road previews without running a vehicle."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chrono_suspension.config import load_config  # noqa: E402

RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results"
CORNER_NAMES = ("FL", "FR", "RL", "RR")
COLORS = ("#0072B2", "#009E73", "#D55E00", "#CC79A7")
LINESTYLES = ("-", "-", "--", "--")
ROAD_TITLES = {
    "iso_b": "ISO B reference spectrum",
    "iso_c": "ISO C reference spectrum",
    "sine": "Sinusoidal road",
    "step": "Smoothed step",
    "pothole": "Local pothole",
    "bump": "Local bump",
    "composite": "Random road + local defects",
}


def new_output_directory(value: str | None, seed: int) -> Path:
    """Reserve a new results directory, refusing overwrite and out-of-tree output."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = (
        Path(value).expanduser()
        if value
        else RESULTS_ROOT / "chrono" / "road_preview" / f"seed{seed}_{timestamp}"
    ).resolve()
    if not path.is_relative_to(RESULTS_ROOT.resolve()):
        raise ValueError(f"Output directory must be inside {RESULTS_ROOT}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)
    return path


def sample_times(duration: float, sample_dt: float) -> np.ndarray:
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("--duration must be finite and positive")
    if not math.isfinite(sample_dt) or not 0 < sample_dt <= duration:
        raise ValueError("--sample-dt must be finite, positive and no larger than --duration")
    steps = math.floor(duration / sample_dt + 1e-10)
    if steps > 2_000_000:
        raise ValueError("Preview is limited to 2,000,001 samples per road")
    times = np.arange(steps + 1, dtype=float) * sample_dt
    if duration - times[-1] > max(1e-12, sample_dt * 1e-9):
        times = np.r_[times, duration]
    return times


def style_axis(ax: Any) -> None:
    ax.set_facecolor("#FBFCFE")
    ax.grid(True, color="#DCE3EB", linewidth=0.6)
    ax.axhline(0, color="#8C9CAA", linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#BCC8D4")
    ax.tick_params(colors="#455669")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Road height (mm)")


def plot_corners(ax: Any, times: np.ndarray, heights: np.ndarray, corners: tuple[int, ...]) -> None:
    for corner in corners:
        ax.plot(
            times,
            heights[:, corner] * 1000,
            color=COLORS[corner],
            linestyle=LINESTYLES[corner],
            linewidth=1.25,
            label=CORNER_NAMES[corner],
        )
    style_axis(ax)


def save_figure(fig: Any, path: Path) -> list[str]:
    names = []
    for extension in ("png", "pdf"):
        destination = path.with_suffix(f".{extension}")
        fig.savefig(destination, dpi=220, facecolor="white", bbox_inches="tight")
        names.append(destination.name)
    plt.close(fig)
    return names


def detail_window(times: np.ndarray, velocities: np.ndarray, delay: float) -> tuple[float, float]:
    """Focus on the strongest sampled excitation, including the rear-wheel delay."""
    duration = float(times[-1])
    centre = float(times[int(np.argmax(np.max(np.abs(velocities), axis=1)))])
    width = min(duration, max(1.0, 4 * delay))
    start = max(0.0, min(centre - width / 2, duration - width))
    return start, start + width


def plot_individual(
    name: str,
    times: np.ndarray,
    heights: np.ndarray,
    velocities: np.ndarray,
    delay: float,
    speed: float,
    seed: int,
    output: Path,
) -> dict:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), layout="constrained", sharey=True)
    zoom = detail_window(times, velocities, delay)
    for column, (track, corners) in enumerate((("Left", (0, 2)), ("Right", (1, 3)))):
        for row in range(2):
            ax = axes[row, column]
            plot_corners(ax, times, heights, corners)
            ax.set_xlim((0, times[-1]) if row == 0 else zoom)
            ax.set_title(f"{track} track | {'Full duration' if row == 0 else 'Excitation detail'}")
            ax.legend(loc="upper right", ncols=2)
    fig.suptitle(
        f"{ROAD_TITLES.get(name, name)}\n"
        f"Speed {speed:g} m/s  |  Axle delay {delay:.4f} s  |  Seed {seed}",
        fontsize=16,
        fontweight="bold",
        color="#17344C",
    )
    files = save_figure(fig, output / name)
    return {"figures": files, "detail_time_window_s": list(zoom)}


def plot_overview(samples: dict, times: np.ndarray, seed: int, output: Path) -> list[str]:
    fig, axes = plt.subplots(4, 2, figsize=(13.5, 13), layout="constrained")
    for ax, (name, road) in zip(axes.flat, samples.items(), strict=False):
        plot_corners(ax, times, road["heights"], (0, 1, 2, 3))
        ax.set_xlim(0, times[-1])
        ax.set_title(ROAD_TITLES.get(name, name), fontweight="bold")
    guide = axes.flat[-1]
    guide.set_axis_off()
    for corner in range(4):
        guide.plot(
            [],
            [],
            color=COLORS[corner],
            linestyle=LINESTYLES[corner],
            label=f"{CORNER_NAMES[corner]} - {'front' if corner < 2 else 'rear'} "
            f"{'left' if corner % 2 == 0 else 'right'}",
        )
    guide.legend(loc="upper left", frameon=False, ncols=2, fontsize=11)
    guide.text(
        0.03,
        0.64,
        "Front / rear inputs follow the same spatial track.\n"
        "Rear wheels see the road after the wheelbase delay.\n\n"
        "Shared left/right deterministic roads have overlapping lines.\n"
        "Each individual figure includes an excitation detail window.\n\n"
        "Preview only: no vehicle or suspension simulation.\n"
        "CSV files retain SI units: m and m/s.\n"
        "See metadata.json for the actual spectrum and input settings.",
        transform=guide.transAxes,
        va="top",
        color="#455669",
        fontsize=10.5,
        linespacing=1.55,
    )
    fig.suptitle(
        f"Seven road inputs | Four-wheel previews | Seed {seed}",
        fontsize=18,
        fontweight="bold",
        color="#17344C",
    )
    return save_figure(fig, output / "seven_roads")


def plot_reference_psd(entries: dict, output: Path) -> list[str]:
    """Plot the actual configured B/C spatial PSD targets, not a trace estimate."""
    fig, ax = plt.subplots(figsize=(9.2, 5.8), layout="constrained")
    for name, color in (("iso_b", "#0072B2"), ("iso_c", "#D55E00")):
        spectrum = entries[name]["road_metadata"]["spectral"]
        reference = float(spectrum["reference_frequency_per_m"])
        gdn0 = float(spectrum["gdn0_m3"])
        frequencies = np.geomspace(spectrum["n_min_per_m"], spectrum["n_max_per_m"], 500)
        target = gdn0 * (frequencies / reference) ** -2
        ax.loglog(
            frequencies,
            target,
            color=color,
            linewidth=2,
            label=f"{name.upper().replace('_', ' ')}: Gd(n0) = {gdn0:.3g} m3",
        )
        ax.scatter([reference], [gdn0], color=color, s=35, zorder=4)
    ax.set_facecolor("#FBFCFE")
    ax.grid(True, which="major", color="#DCE3EB", linewidth=0.7)
    ax.grid(True, which="minor", color="#E9EEF3", linewidth=0.4)
    ax.set_xlabel("Spatial frequency n (cycles/m)")
    ax.set_ylabel(r"One-sided spatial height PSD $G_d(n)$ ($\mathrm{m}^3$)")
    ax.set_title(
        "B/C reference spectra over the configured synthesis band\n"
        r"$G_d(n) = G_d(n_0)\,(n/n_0)^{-2}$",
        fontweight="bold",
        pad=14,
    )
    ax.legend(loc="lower left")
    fig.text(
        0.5,
        -0.035,
        "Theoretical targets read from road metadata; this is not an estimated PSD of the preview.",
        ha="center",
        fontsize=9,
        color="#455669",
    )
    return save_figure(fig, output / "iso_reference_psd")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional simulation YAML for geometry and road settings")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--sample-dt", type=float, default=0.002)
    parser.add_argument(
        "--output-dir", help="New directory under repository results/; never overwrite"
    )
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    from chrono_suspension.roads import FourWheelRoad, seven_road_scenarios

    cfg = load_config(args.config)
    times = sample_times(args.duration, args.sample_dt)
    scenarios = seven_road_scenarios()
    if set(scenarios) != set(ROAD_TITLES):
        raise ValueError(f"Expected seven road scenarios, received {list(scenarios)}")
    vehicle = cfg["vehicle"]
    wheelbase = vehicle["front_distance_m"] + vehicle["rear_distance_m"]
    samples, entries = {}, {}
    for name, preset in scenarios.items():
        scenario = {**cfg["road"], **preset, "seed": args.seed}
        road = FourWheelRoad(
            scenario, vehicle["front_distance_m"], vehicle["rear_distance_m"], args.seed
        )
        values = np.asarray([road.sample(float(t)) for t in times], dtype=float)
        if values.shape != (len(times), 2, 4) or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid four-wheel road samples for {name}")
        heights, velocities = values[:, 0], values[:, 1]
        samples[name] = {"heights": heights, "velocities": velocities}
        entries[name] = {
            "scenario": scenario,
            "road_metadata": road.metadata(),
            "sampled_height_min_m": heights.min(axis=0).tolist(),
            "sampled_height_max_m": heights.max(axis=0).tolist(),
            "sampled_height_rms_m": np.sqrt(np.mean(heights**2, axis=0)).tolist(),
            "sampled_peak_absolute_velocity_mps": np.max(np.abs(velocities), axis=0).tolist(),
        }
    report = {
        "schema": "chrono-seven-road-preview-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "config": cfg,
        "requested_duration_s": args.duration,
        "sample_dt_s": args.sample_dt,
        "last_sample_interval_s": float(times[-1] - times[-2]),
        "sample_count_per_road": len(times),
        "corner_order": list(CORNER_NAMES),
        "csv_units": {"time": "s", "height": "m", "velocity": "m/s"},
        "figure_height_unit": "mm",
        "configuration_precedence": "base config road, then named preset, then command-line seed",
        "interpretation": "Prescribed road inputs only; finite traces do not certify an ISO road class.",
        "roads": entries,
    }
    # Check metadata serializability before reserving an output directory.
    json.dumps(report, allow_nan=False)
    output = new_output_directory(args.output_dir, args.seed)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlecolor": "#17344C",
            "axes.labelcolor": "#455669",
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#E2E8EF",
            "pdf.fonttype": 42,
        }
    )
    header = ",".join(
        ["time_s"]
        + [f"height_{corner}_m" for corner in CORNER_NAMES]
        + [f"velocity_{corner}_mps" for corner in CORNER_NAMES]
    )
    for name, data in samples.items():
        destination = output / f"{name}.csv"
        np.savetxt(
            destination,
            np.column_stack([times, data["heights"], data["velocities"]]),
            delimiter=",",
            header=header,
            comments="",
            fmt="%.12e",
        )
        speed = float(entries[name]["scenario"]["speed_mps"])
        entries[name].update(
            plot_individual(
                name,
                times,
                data["heights"],
                data["velocities"],
                wheelbase / speed,
                speed,
                args.seed,
                output,
            )
        )
        entries[name]["csv"] = destination.name
    report["overview_figures"] = plot_overview(samples, times, args.seed, output)
    report["theoretical_psd_figures"] = plot_reference_psd(entries, output)
    (output / "metadata.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    print(f"Road previews: {output}")
    print(f"Seven scenarios, {len(times)} samples each; PNG/PDF, SI-unit CSV and metadata.json")


if __name__ == "__main__":
    main()
