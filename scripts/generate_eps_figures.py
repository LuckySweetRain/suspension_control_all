"""Generate Chapter 6 EPS figures with the same style as Chapter 5 MATLAB plots."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ijrnc_figure_style import (
    CTRL_STYLE,
    apply_ijrnc_style,
    compact_legend,
    ieee_figure,
    plot_controller,
    save_ieee,
    style_axis,
)

from config import apply_algorithm_overrides, load_config
from experiments.evaluation import make_controller, make_env, rollout
from visualization.spectra import sampling_frequency

OUT_DIR = ROOT.parent / "active-suspension-midterm-report" / "figures"
SI_CKPT = ROOT / "results/20260621_182309_ppo/checkpoints/best.pt"
SCRATCH_CKPT = ROOT / "results/20260621_191152_ppo/checkpoints/best.pt"


def _checkpoint_matched_config(config: dict, checkpoint: Path) -> dict:
    """Align plotting config with old PPO checkpoints without changing training configs."""
    cfg = copy.deepcopy(config)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved = ckpt.get("cfg", {})
    obs_dim_base = int(saved.get("obs_dim_base", cfg["rl"].get("obs_dim_base", 14)))
    preview_steps = int(saved.get("preview_steps", cfg["preview"].get("steps", 10)))
    preview_token_dim = int(saved.get("preview_token_dim", cfg["rl"].get("preview_token_dim", 2)))
    cfg["rl"]["obs_dim_base"] = obs_dim_base
    cfg["rl"]["preview_token_dim"] = preview_token_dim
    cfg["preview"]["steps"] = preview_steps
    cfg["action_in_observation"] = obs_dim_base >= 16
    return cfg


def _run_rl(config: dict, checkpoint: Path, scenario: dict) -> dict:
    matched = _checkpoint_matched_config(config, checkpoint)
    env = make_env(matched, scenario=scenario, use_preview=True)
    ctrl = make_controller("PPO", env, matched, checkpoint=str(checkpoint), algorithm="ppo")
    run = rollout(env, ctrl, use_preview=True)
    env.close()
    return run


def _run_classical(
    config: dict, controller: str, scenario: dict, use_preview: bool = False
) -> dict:
    env = make_env(config, scenario=scenario, use_preview=use_preview)
    ctrl = make_controller(controller, env, config)
    run = rollout(env, ctrl, use_preview=use_preview)
    env.close()
    return run


def _shared_xlabel(fig, text: str, y: float = 0.025) -> None:
    fig.text(0.5, y, text, ha="center", va="center", fontsize=8, fontname="Times New Roman")


def metric_bar(df: pd.DataFrame) -> None:
    key_scenarios = ["class_b", "class_c", "bump", "sine_long", "mixed"]
    labels = ["Class B", "Class C", "Bump", "Sine", "Mixed"]
    controllers = ["PASSIVE", "PID", "SPDF", "MPC", "PPO_scratch", "SI-RPPO"]
    plot_df = df[df["Scenario"].isin(key_scenarios)]

    fig = ieee_figure("double", height=2.6)
    ax = fig.add_axes([0.075, 0.24, 0.91, 0.68])
    x = np.arange(len(key_scenarios))
    width = 0.12
    offsets = (np.arange(len(controllers)) - (len(controllers) - 1) / 2) * width
    for offset, ctrl in zip(offsets, controllers, strict=True):
        vals = [
            plot_df[(plot_df["Scenario"] == scenario) & (plot_df["Controller"] == ctrl)][
                "BodyAccRMS_mps2"
            ].values[0]
            for scenario in key_scenarios
        ]
        style = CTRL_STYLE[ctrl]
        ax.bar(
            x + offset,
            vals,
            width=width * 0.92,
            color=style["color"],
            edgecolor="k",
            linewidth=0.35,
            label=style["label"],
        )
    ax.set_ylabel(r"Body acc. RMS ($\mathrm{m/s^2}$)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    style_axis(ax)
    compact_legend(ax, ncol=3, loc="upper left")
    save_ieee(fig, OUT_DIR, "allmethods_metric_bar.eps")
    print("1. allmethods_metric_bar.eps")


def training_curves() -> None:
    with (ROOT / "results/20260621_182309_ppo/training_history.json").open() as f:
        history = json.load(f)
    with (ROOT / "results/20260621_182309_ppo/eval_history.json").open() as f:
        eval_history = json.load(f)

    fig = ieee_figure("single", height=3.15)
    gs = fig.add_gridspec(3, 1, left=0.18, right=0.97, bottom=0.12, top=0.97, hspace=0.28)
    eps = [h["episode"] for h in history]
    eval_eps = [e["episode"] for e in eval_history]

    ax = fig.add_subplot(gs[0])
    ax.plot(eval_eps, [e["mean_return"] for e in eval_history], "b-", linewidth=0.5)
    ax.set_ylabel("Eval return")
    style_axis(ax)

    ax = fig.add_subplot(gs[1])
    ax.plot(eps, [h.get("mean_improvement_gate", 0.0) for h in history], "g-", linewidth=0.5)
    ax.set_ylabel("Gate")
    ax.set_ylim(-0.05, 1.05)
    style_axis(ax)

    ax = fig.add_subplot(gs[2])
    ax.plot(eps, [h.get("mean_action_delta", 0.0) for h in history], "r-", linewidth=0.5)
    ax.set_ylabel(r"$\Delta u$")
    ax.set_xlabel("Episode")
    style_axis(ax)
    save_ieee(fig, OUT_DIR, "sirppo_training_curves.eps")
    print("2. sirppo_training_curves.eps")


def comfort_energy(df: pd.DataFrame) -> None:
    mean_df = (
        df.groupby("Controller")
        .agg({"BodyAccRMS_mps2": "mean", "ControlAbsEnergy_J": "mean"})
        .reset_index()
    )
    order = ["PASSIVE", "PID", "SPDF", "MPC", "PPO_scratch", "SI-RPPO"]

    fig = ieee_figure("single", height=2.25)
    ax = fig.add_axes([0.18, 0.2, 0.78, 0.72])
    for ctrl in order:
        row = mean_df[mean_df["Controller"] == ctrl]
        if row.empty:
            continue
        style = CTRL_STYLE[ctrl]
        ax.plot(
            row["ControlAbsEnergy_J"].values[0],
            row["BodyAccRMS_mps2"].values[0],
            marker="o",
            markersize=3.2,
            color=style["color"],
            linestyle="none",
            label=style["label"],
        )
    ax.set_xscale("log")
    ax.set_xlabel("Control energy (J)")
    ax.set_ylabel(r"Body acc. RMS ($\mathrm{m/s^2}$)")
    style_axis(ax)
    compact_legend(ax, loc="best", ncol=1)
    save_ieee(fig, OUT_DIR, "comfort_energy_tradeoff.eps")
    print("3. comfort_energy_tradeoff.eps")


def rollout_set(config: dict, scenario: dict) -> dict[str, dict]:
    scratch_cfg = apply_algorithm_overrides(load_config("configs/ppo_scratch.yaml"), "ppo")
    return {
        "PASSIVE": _run_classical(config, "PASSIVE", scenario, use_preview=False),
        "PPO_scratch": _run_rl(scratch_cfg, SCRATCH_CKPT, scenario),
        "SI-RPPO": _run_rl(config, SI_CKPT, scenario),
    }


def time_domain_accel(runs: dict[str, dict]) -> None:
    fig = ieee_figure("single", height=2.25)
    gs = fig.add_gridspec(2, 1, left=0.18, right=0.97, bottom=0.16, top=0.96, hspace=0.20)
    signals = [
        ("ddzb", r"$\ddot z_b\ (\mathrm{m/s^2})$"),
        ("ddtheta", r"$\ddot\theta_b\ (\mathrm{rad/s^2})$"),
    ]
    for idx, (key, ylabel) in enumerate(signals):
        ax = fig.add_subplot(gs[idx])
        for ctrl, run in runs.items():
            data = run["data"]
            mask = (data["time"] >= 1.0) & (data["time"] <= 4.0)
            plot_controller(ax, data["time"].values[mask], data[key].values[mask], ctrl)
        ax.set_ylabel(ylabel)
        style_axis(ax)
        if idx == 0:
            compact_legend(ax, loc="upper right")
        else:
            ax.set_xlabel("Time (s)")
    save_ieee(fig, OUT_DIR, "allmethods_ts_accel.eps")
    print("4. allmethods_ts_accel.eps")


def psd_plot(runs: dict[str, dict]) -> None:
    fig = ieee_figure("single", height=2.1)
    ax = fig.add_axes([0.18, 0.18, 0.78, 0.76])
    for ctrl, run in runs.items():
        steady = run["data"][run["data"]["time"] >= 1.0]
        fs = sampling_frequency(steady["time"].to_numpy())
        nperseg = min(2048, max(128, len(steady) // 2))
        f, pxx = signal.welch(steady["ddzb"].values, fs=fs, nperseg=nperseg)
        style = CTRL_STYLE[ctrl]
        ax.loglog(
            f,
            pxx,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=0.5,
            label=style["label"],
        )
    ax.set_xlim(1e-1, 30)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"PSD of $\ddot z_b$")
    style_axis(ax)
    compact_legend(ax, loc="lower left")
    save_ieee(fig, OUT_DIR, "allmethods_psd.eps")
    print("5. allmethods_psd.eps")


def gate_behavior(sirppo_run: dict) -> None:
    data = sirppo_run["data"]
    mask = (data["time"] >= 1.0) & (data["time"] <= 6.0)
    t = data["time"].values[mask]

    fig = ieee_figure("single", height=2.8)
    gs = fig.add_gridspec(3, 1, left=0.18, right=0.97, bottom=0.12, top=0.97, hspace=0.28)

    ax = fig.add_subplot(gs[0])
    ax.plot(t, data["ddzb"].values[mask], "b-", linewidth=0.5)
    ax.set_ylabel(r"$\ddot z_b$")
    style_axis(ax)

    ax = fig.add_subplot(gs[1])
    gate_col = "policy_improvement_gate"
    if gate_col in data.columns:
        ax.plot(t, data[gate_col].values[mask], "g-", linewidth=0.5)
    ax.set_ylabel("Gate")
    ax.set_ylim(-0.05, 1.05)
    style_axis(ax)

    ax = fig.add_subplot(gs[2])
    force = data["Uaf"].values[mask] + data["Uar"].values[mask]
    ax.plot(t, force, "r-", linewidth=0.5)
    ax.set_ylabel("Force (N)")
    ax.set_xlabel("Time (s)")
    style_axis(ax)
    save_ieee(fig, OUT_DIR, "sirppo_gate_behavior.eps")
    print("6. sirppo_gate_behavior.eps")


def motion_plot(runs: dict[str, dict]) -> None:
    fig = ieee_figure("single", height=2.25)
    gs = fig.add_gridspec(2, 1, left=0.18, right=0.97, bottom=0.16, top=0.96, hspace=0.20)
    for idx, (key, ylabel) in enumerate(
        [("zb", r"$z_b\ (\mathrm{m})$"), ("theta", r"$\theta_b\ (\mathrm{rad})$")]
    ):
        ax = fig.add_subplot(gs[idx])
        for ctrl, run in runs.items():
            data = run["data"]
            mask = (data["time"] >= 1.0) & (data["time"] <= 4.0)
            plot_controller(ax, data["time"].values[mask], data[key].values[mask], ctrl)
        ax.set_ylabel(ylabel)
        style_axis(ax)
        if idx == 0:
            compact_legend(ax, loc="upper right")
        else:
            ax.set_xlabel("Time (s)")
    save_ieee(fig, OUT_DIR, "allmethods_ts_motion.eps")
    print("7. allmethods_ts_motion.eps")


def convergence_comparison() -> None:
    with (ROOT / "results/20260621_182309_ppo/eval_history.json").open() as f:
        sirppo_eval = json.load(f)
    with (ROOT / "results/20260621_191152_ppo/eval_history.json").open() as f:
        scratch_eval = json.load(f)

    fig = ieee_figure("single", height=2.05)
    ax = fig.add_axes([0.18, 0.19, 0.78, 0.73])
    ax.plot(
        [e["episode"] for e in sirppo_eval],
        [e["mean_return"] for e in sirppo_eval],
        color=CTRL_STYLE["SI-RPPO"]["color"],
        linestyle=CTRL_STYLE["SI-RPPO"]["linestyle"],
        linewidth=0.5,
        label=CTRL_STYLE["SI-RPPO"]["label"],
    )
    ax.plot(
        [e["episode"] for e in scratch_eval],
        [e["mean_return"] for e in scratch_eval],
        color=CTRL_STYLE["PPO_scratch"]["color"],
        linestyle=CTRL_STYLE["PPO_scratch"]["linestyle"],
        linewidth=0.5,
        label=CTRL_STYLE["PPO_scratch"]["label"],
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Eval return")
    style_axis(ax)
    compact_legend(ax, loc="lower right")
    save_ieee(fig, OUT_DIR, "convergence_comparison.eps")
    print("8. convergence_comparison.eps")


def main() -> None:
    apply_ijrnc_style()
    df = pd.read_csv(ROOT / "results/comprehensive_comparison.csv")
    config = apply_algorithm_overrides(load_config("configs/half_car_safe_ppo.yaml"), "ppo")
    scenario = config["scenarios"][0]

    metric_bar(df)
    training_curves()
    comfort_energy(df)
    runs = rollout_set(config, scenario)
    time_domain_accel(runs)
    psd_plot(runs)
    gate_behavior(runs["SI-RPPO"])
    motion_plot(runs)
    convergence_comparison()
    print(f"\nAll EPS figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
