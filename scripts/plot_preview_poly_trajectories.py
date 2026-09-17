from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from models.half_car import HalfCarParams
from roads.road_profiles import RoadProfileFactory


def preview_poly_reference(
    road,
    t_values: np.ndarray,
    *,
    speed: float,
    wheelbase: float,
    preview_steps: int,
    preview_dt: float,
    lookahead_steps: int,
    degree: int,
    a: float,
    b: float,
    beta_z: float,
    beta_theta: float,
) -> dict[str, np.ndarray]:
    tau = np.arange(1, preview_steps + 1, dtype=np.float64) * preview_dt
    degree = min(degree, preview_steps - 1)
    lookahead_steps = int(np.clip(lookahead_steps, 1, preview_steps))
    tau_l = lookahead_steps * preview_dt

    vander = np.vander(tau, N=degree + 1, increasing=True)
    vander_pinv = np.linalg.pinv(vander)
    basis0 = np.asarray([tau_l**j for j in range(degree + 1)], dtype=np.float64)
    basis1 = np.asarray(
        [0.0] + [j * tau_l ** (j - 1) for j in range(1, degree + 1)], dtype=np.float64
    )
    basis2 = np.asarray(
        [0.0, 0.0] + [j * (j - 1) * tau_l ** (j - 2) for j in range(2, degree + 1)],
        dtype=np.float64,
    )

    zf = np.zeros_like(t_values)
    zr = np.zeros_like(t_values)
    x1_ref = np.zeros_like(t_values)
    dx1_ref = np.zeros_like(t_values)
    ddx1_ref = np.zeros_like(t_values)
    x3_ref = np.zeros_like(t_values)
    dx3_ref = np.zeros_like(t_values)
    ddx3_ref = np.zeros_like(t_values)

    for idx, t in enumerate(t_values):
        zf_now, _, zr_now, _ = road.pair(float(t), speed, wheelbase)
        zf[idx] = zf_now
        zr[idx] = zr_now

        preview = road.preview(float(t), speed, wheelbase, preview_steps, preview_dt)
        cf = vander_pinv @ preview[:, 0]
        cr = vander_pinv @ preview[:, 1]

        zf_q = np.asarray([basis0 @ cf, basis1 @ cf, basis2 @ cf], dtype=np.float64)
        zr_q = np.asarray([basis0 @ cr, basis1 @ cr, basis2 @ cr], dtype=np.float64)

        x1 = beta_z * (b * zf_q + a * zr_q) / (a + b)
        x3 = beta_theta * (zf_q - zr_q) / (a + b)
        x1_ref[idx], dx1_ref[idx], ddx1_ref[idx] = x1
        x3_ref[idx], dx3_ref[idx], ddx3_ref[idx] = x3

    return {
        "zf": zf,
        "zr": zr,
        "x1_ref": x1_ref,
        "dx1_ref": dx1_ref,
        "ddx1_ref": ddx1_ref,
        "x3_ref": x3_ref,
        "dx3_ref": dx3_ref,
        "ddx3_ref": ddx3_ref,
    }


def make_summary_plot(results: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(
        len(results),
        2,
        figsize=(10.8, 12.6),
        sharex=True,
        constrained_layout=True,
    )

    for row, item in enumerate(results):
        t = item["t"]
        data = item["data"]
        label = item["label"]
        ax_l, ax_r = axes[row]

        ax_l.plot(t, data["zf"] * 1000.0, color="#6b7280", linewidth=0.9, label="front road")
        ax_l.plot(
            t,
            data["zr"] * 1000.0,
            color="#9ca3af",
            linewidth=0.9,
            linestyle="--",
            label="rear road",
        )
        ax_l.plot(t, data["x1_ref"] * 1000.0, color="#2563eb", linewidth=1.25, label="heave ref")
        ax_l.set_ylabel(f"{label}\nmm")
        ax_l.grid(True, alpha=0.28)

        ax_r.plot(t, np.rad2deg(data["x3_ref"]), color="#dc2626", linewidth=1.25)
        ax_r.set_ylabel("deg")
        ax_r.grid(True, alpha=0.28)

        if row == 0:
            ax_l.set_title("Road input and heave preview reference")
            ax_r.set_title("Pitch preview reference")
            ax_l.legend(loc="upper right", ncols=3, fontsize=7, frameon=False)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle("Polynomial Preview Reference Trajectories under Seven Road Profiles", fontsize=13)

    for suffix in (".png", ".pdf", ".eps"):
        fig.savefig(out_path.with_suffix(suffix), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_single_plot(item: dict, out_path: Path) -> None:
    t = item["t"]
    data = item["data"]
    label = item["label"]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True, constrained_layout=True)
    axes[0].plot(t, data["zf"] * 1000.0, color="#6b7280", linewidth=1.0, label="front road")
    axes[0].plot(
        t, data["zr"] * 1000.0, color="#9ca3af", linewidth=1.0, linestyle="--", label="rear road"
    )
    axes[0].plot(t, data["x1_ref"] * 1000.0, color="#2563eb", linewidth=1.4, label="heave ref")
    axes[0].set_ylabel("Height (mm)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", ncols=3, fontsize=8, frameon=False)

    axes[1].plot(t, np.rad2deg(data["x3_ref"]), color="#dc2626", linewidth=1.4)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Pitch ref (deg)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"{label}: Polynomial Preview Reference")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot polynomial preview reference trajectories for all road scenarios."
    )
    parser.add_argument(
        "--config", default="configs/default.yaml", help="YAML config with seven road scenarios."
    )
    parser.add_argument(
        "--out-dir",
        default=r"E:\Desktop\workspace\active-suspension-midterm-report\figures",
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--plot-seconds", type=float, default=4.0, help="Time span shown in the figures."
    )
    parser.add_argument(
        "--sample-dt",
        type=float,
        default=None,
        help="Output sample interval; defaults to preview_dt.",
    )
    parser.add_argument(
        "--preview-steps", type=int, default=450, help="Number of preview samples N_p."
    )
    parser.add_argument(
        "--preview-dt", type=float, default=0.002, help="Preview sampling period T_p."
    )
    parser.add_argument(
        "--lookahead-steps", type=int, default=300, help="Lookahead index N_L used for reference."
    )
    parser.add_argument("--degree", type=int, default=3, help="Polynomial degree m.")
    parser.add_argument(
        "--beta-z", type=float, default=0.35, help="Heave reference amplitude coefficient."
    )
    parser.add_argument(
        "--beta-theta", type=float, default=0.35, help="Pitch reference amplitude coefficient."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(ROOT / args.config)
    params = HalfCarParams.from_config(config)
    sample_dt = float(args.sample_dt or args.preview_dt)
    t_values = np.arange(0.0, args.plot_seconds + 0.5 * sample_dt, sample_dt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for scenario in config["scenarios"]:
        label = str(scenario.get("label", scenario["name"]))
        speed = float(scenario.get("speed", config.get("speed", 20.0)))
        road = RoadProfileFactory.create(
            scenario,
            duration=args.plot_seconds
            + args.preview_steps * args.preview_dt
            + config.get("wheelbase", 2.7) / speed,
            dt=float(config.get("dt", 0.001)),
        )
        data = preview_poly_reference(
            road,
            t_values,
            speed=speed,
            wheelbase=float(config.get("wheelbase", params.a + params.b)),
            preview_steps=args.preview_steps,
            preview_dt=args.preview_dt,
            lookahead_steps=args.lookahead_steps,
            degree=args.degree,
            a=params.a,
            b=params.b,
            beta_z=args.beta_z,
            beta_theta=args.beta_theta,
        )
        item = {"name": scenario["name"], "label": label, "t": t_values, "data": data}
        results.append(item)
        make_single_plot(item, out_dir / f"preview_poly_{scenario['name']}.png")

    make_summary_plot(results, out_dir / "preview_poly_seven_roads")
    print(f"Generated summary: {out_dir / 'preview_poly_seven_roads.png'}")
    print(f"Generated EPS:     {out_dir / 'preview_poly_seven_roads.eps'}")
    print(f"Generated singles: {out_dir / 'preview_poly_<scenario>.png'}")


if __name__ == "__main__":
    main()
