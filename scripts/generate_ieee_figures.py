"""Generate IEEE-style figures matching SPDF paper conventions.

Style rules from SPDF paper (IJRNC):
  - Figure size: single 3.5x2.5 in, double 7.16x3 in
  - Font: Times New Roman, 7-8pt
  - Line styles: Passive=blue solid, PID=black dash-dot, SPDF=red dash-dot
  - Grid: on
  - PSD: loglog scale
  - Output: PDF vector
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import apply_algorithm_overrides, load_config
from experiments.evaluation import rollout, make_controller, make_env

# === IEEE style settings ===
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6,
    "lines.linewidth": 0.5,
    "grid.alpha": 0.3,
    "text.usetex": False,  # set True if LaTeX is available
})
# Color/line convention matching SPDF paper:
# Passive=blue solid, PID=black dash-dot, SPDF=red dash-dot
STYLES = {
    "PASSIVE": {"color": "b", "linestyle": "-"},
    "PID": {"color": "k", "linestyle": "-."},
    "SPDF": {"color": "r", "linestyle": "-."},
    "MPC": {"color": "m", "linestyle": "--"},
    "PPO_scratch": {"color": "#C44E52", "linestyle": "-"},
    "SI-RPPO": {"color": "g", "linestyle": "-"},
}
ORDER = ["PASSIVE", "PID", "SPDF", "MPC", "PPO_scratch", "SI-RPPO"]

out_dir = ROOT.parent / "active-suspension-midterm-report" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(ROOT / "results/comprehensive_comparison.csv")


def ieee_fig(single=True, height=None):
    """Create IEEE-sized figure."""
    w = 3.5 if single else 7.16
    h = height or (1.8 if single else 3.0)
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.18, top=0.92)
    return fig, ax


def ieee_save(fig, name):
    """Save as PDF with vector output."""
    path = out_dir / name
    fig.savefig(str(path), format="eps", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


def plot_with_style(ax, data, key, label):
    """Plot with SPDF paper line conventions."""
    s = STYLES.get(label, {"color": "k", "linestyle": "-"})
    ax.plot(data, color=s["color"], linestyle=s["linestyle"],
            linewidth=0.5, label=label)


# ===== 1. PSD: Body Acc - Class B (matching PLOT_rank_PSD_SPDF_PID.m) =====
print("1. PSD plot...")
# Need to run simulation first to get time-series data
config = apply_algorithm_overrides(load_config("configs/half_car_safe_ppo.yaml"), "ppo")
scenario = config["scenarios"][0]  # class_b

# Run all controllers
all_runs = {}
for ctrl_name in ["PASSIVE", "PID", "SPDF"]:
    env = make_env(config, scenario=scenario, use_preview=(ctrl_name == "MPC"))
    ctrl = make_controller(ctrl_name, env, config)
    run = rollout(env, ctrl, use_preview=(ctrl_name == "MPC"))
    all_runs[ctrl_name] = run
    env.close()

# PPO_scratch
scratch_cfg = apply_algorithm_overrides(load_config("configs/ppo_scratch.yaml"), "ppo")
env = make_env(scratch_cfg, scenario=scenario, use_preview=True)
run = rollout(env, make_controller("PPO", env, scratch_cfg,
              checkpoint="results/20260621_191152_ppo/checkpoints/best.pt", algorithm="ppo"), use_preview=True)
all_runs["PPO_scratch"] = run
env.close()

# SI-RPPO
env = make_env(config, scenario=scenario, use_preview=True)
run = rollout(env, make_controller("PPO", env, config,
              checkpoint="results/20260621_182309_ppo/checkpoints/best.pt", algorithm="ppo"), use_preview=True)
all_runs["SI-RPPO"] = run
env.close()

# PSD: 2x2 subplot (body acc Class B/C, pitch acc Class B/C) like SPDF paper
fig = plt.figure(figsize=(3.5, 2.8))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.4,
                       left=0.14, right=0.96, bottom=0.14, top=0.92)
fs = 1.0 / 0.001

scenario_b = config["scenarios"][0]
scenario_c = config["scenarios"][1]

for idx, (sc, sc_label) in enumerate([(scenario_b, "ClassB"), (scenario_c, "ClassC")]):
    for j, (signal_name, ylabel) in enumerate([("ddzb", r"$\ddot{z}_b$"), ("ddtheta", r"$\ddot{\theta}_b$")]):
        ax = fig.add_subplot(gs[j, idx])
        for ctrl_name in ["PASSIVE", "PID", "SPDF"]:
            env_c = make_env(config, scenario=sc, use_preview=(ctrl_name == "MPC"))
            ctrl_c = make_controller(ctrl_name, env_c, config)
            run_c = rollout(env_c, ctrl_c, use_preview=(ctrl_name == "MPC"))
            data = run_c["data"]
            steady = data[data["time"] >= 1.0]
            f, Pxx = signal.welch(steady[signal_name].values, fs=fs,
                                  nperseg=min(2048, len(steady)//4))
            s = STYLES[ctrl_name]
            ax.loglog(f, Pxx, color=s["color"], linestyle=s["linestyle"],
                      linewidth=0.5, label=ctrl_name)
            env_c.close()

        # Also plot SI-RPPO
        env_c = make_env(config, scenario=sc, use_preview=True)
        run_c = rollout(env_c, make_controller("PPO", env_c, config,
                       checkpoint="results/20260621_182309_ppo/checkpoints/best.pt", algorithm="ppo"), use_preview=True)
        data = run_c["data"]
        steady = data[data["time"] >= 1.0]
        f, Pxx = signal.welch(steady[signal_name].values, fs=fs,
                              nperseg=min(2048, len(steady)//4))
        s = STYLES["SI-RPPO"]
        ax.loglog(f, Pxx, color=s["color"], linestyle=s["linestyle"],
                  linewidth=0.5, label="SI-RPPO")
        env_c.close()

        ax.set_xlim(1e-1, 30)
        ax.set_xlabel("Frequency (Hz)", fontsize=7)
        ax.set_ylabel(f"PSD of {ylabel}", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
        if idx == 0 and j == 0:
            ax.legend(fontsize=5, loc="lower left")

# Global xlabel
fig.text(0.5, 0.01, "Frequency (Hz)", ha="center", fontsize=8,
         fontname="Times New Roman")
ieee_save(fig, "allmethods_psd.eps")
print("  PSD done")


# ===== 2. Time-domain: Body Acc + Motion (matching PLOT_rank_acc_SPDF_PID.m) =====
print("2. Time-domain plots...")
# 2x2: body acc class B/C, pitch acc class B/C
fig = plt.figure(figsize=(3.5, 2.8))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.4,
                       left=0.14, right=0.96, bottom=0.14, top=0.92)

for idx, (sc, sc_label) in enumerate([(scenario_b, "ClassB"), (scenario_c, "ClassC")]):
    for j, (signal_name, ylabel) in enumerate([("ddzb", r"$\ddot{z}_b\ (\mathrm{m/s^2})$"),
                                                ("ddtheta", r"$\ddot{\theta}_b\ (\mathrm{rad/s^2})$")]):
        ax = fig.add_subplot(gs[j, idx])
        for ctrl_name in ["PASSIVE", "PID", "SPDF"]:
            env_c = make_env(config, scenario=sc, use_preview=(ctrl_name == "MPC"))
            ctrl_c = make_controller(ctrl_name, env_c, config)
            run_c = rollout(env_c, ctrl_c, use_preview=(ctrl_name == "MPC"))
            data = run_c["data"]
            mask = (data["time"] >= 1.0) & (data["time"] <= 4.0)
            s = STYLES[ctrl_name]
            ax.plot(data["time"].values[mask], data[signal_name].values[mask],
                    color=s["color"], linestyle=s["linestyle"], linewidth=0.5, label=ctrl_name)
            env_c.close()
        # SI-RPPO
        env_c = make_env(config, scenario=sc, use_preview=True)
        run_c = rollout(env_c, make_controller("PPO", env_c, config,
                       checkpoint="results/20260621_182309_ppo/checkpoints/best.pt", algorithm="ppo"), use_preview=True)
        data = run_c["data"]
        mask = (data["time"] >= 1.0) & (data["time"] <= 4.0)
        s = STYLES["SI-RPPO"]
        ax.plot(data["time"].values[mask], data[signal_name].values[mask],
                color=s["color"], linestyle=s["linestyle"], linewidth=0.5, label="SI-RPPO")
        env_c.close()

        ax.set_ylabel(ylabel, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
        if idx == 0 and j == 0:
            ax.legend(fontsize=5, loc="upper right")

fig.text(0.5, 0.01, "Time (s)", ha="center", fontsize=8,
         fontname="Times New Roman")
ieee_save(fig, "allmethods_ts_accel.eps")
print("  Time-domain done")


# ===== 3. Bump/Special road response (matching PLOT_special_acc_SPDF_PID.m) =====
print("3. Bump response...")
# Find the bump scenario
bump_sc = [s for s in config["scenarios"] if s["name"] == "bump"][0]
fig = plt.figure(figsize=(3.5, 1.8))
gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.4,
                       left=0.14, right=0.96, bottom=0.22, top=0.90)

# Use SPDF code to collect data
from controllers import SPDFController as SPDFCtrl

for j, (signal_name, ylabel) in enumerate([("ddzb", r"$\ddot{z}_b\ (\mathrm{m/s^2})$"),
                                            ("ddtheta", r"$\ddot{\theta}_b\ (\mathrm{rad/s^2})$")]):
    ax = fig.add_subplot(gs[0, j])
    for ctrl_name in ["PASSIVE", "SPDF"]:
        env_c = make_env(config, scenario=bump_sc, use_preview=False)
        if ctrl_name == "SPDF":
            ctrl_c = SPDFCtrl(env_c.params, env_c.control_dt)
        else:
            ctrl_c = make_controller(ctrl_name, env_c, config)
        run_c = rollout(env_c, ctrl_c, use_preview=False)
        data = run_c["data"]
        mask = data["time"] >= 0.5
        s = STYLES[ctrl_name]
        ax.plot(data["time"].values[mask], data[signal_name].values[mask],
                color=s["color"], linestyle=s["linestyle"], linewidth=0.5, label=ctrl_name)
        env_c.close()
    # SI-RPPO
    env_c = make_env(config, scenario=bump_sc, use_preview=True)
    run_c = rollout(env_c, make_controller("PPO", env_c, config,
                   checkpoint="results/20260621_182309_ppo/checkpoints/best.pt", algorithm="ppo"), use_preview=True)
    data = run_c["data"]
    mask = data["time"] >= 0.5
    s = STYLES["SI-RPPO"]
    ax.plot(data["time"].values[mask], data[signal_name].values[mask],
            color=s["color"], linestyle=s["linestyle"], linewidth=0.5, label="SI-RPPO")
    env_c.close()

    ax.set_ylabel(ylabel, fontsize=7)
    ax.set_xlim(0.5, 3.0)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3)
    if j == 0:
        ax.legend(fontsize=5, loc="upper right")

fig.text(0.5, 0.03, "Time (s)", ha="center", fontsize=8,
         fontname="Times New Roman")
ieee_save(fig, "bump_response.eps")
print("  Bump response done")


# ===== 4. Comfort vs Energy bar chart (summary) =====
print("4. Bar chart...")
fig = plt.figure(figsize=(3.5, 2.2))
ax = fig.subplots()
fig.subplots_adjust(left=0.16, right=0.97, bottom=0.22, top=0.88)

mean_df = df.groupby("Controller").agg({"BodyAccRMS_mps2": "mean",
                                         "ControlAbsEnergy_J": "mean"}).reset_index()
ctrl_order = [c for c in ORDER if c in mean_df["Controller"].values]
x = np.arange(len(ctrl_order))
width = 0.6

colors_bar = [STYLES[c]["color"] for c in ctrl_order]
ax.bar(x, [mean_df[mean_df["Controller"]==c]["BodyAccRMS_mps2"].values[0] for c in ctrl_order],
       width, color=colors_bar, edgecolor="k", linewidth=0.3)
ax.set_xticks(x)
ax.set_xticklabels(ctrl_order, fontsize=6, rotation=30)
ax.set_ylabel("Body Acc RMS (m/s$^2$)", fontsize=7)
ax.grid(True, alpha=0.3, axis="y")
ax.tick_params(labelsize=6)
ieee_save(fig, "allmethods_metric_bar.eps")


# ===== 5. Training curves (using eval history for stable convergence) =====
print("5. Training curves...")
with open(ROOT / "results/20260621_182309_ppo/training_history.json") as f:
    history = json.load(f)
with open(ROOT / "results/20260621_182309_ppo/eval_history.json") as f:
    eval_history = json.load(f)

fig = plt.figure(figsize=(3.5, 3.5))
gs_train = fig.add_gridspec(3, 1, hspace=0.40, left=0.16, right=0.96,
                             bottom=0.10, top=0.95)
eps_list = [h["episode"] for h in history]
gates = [h.get("mean_improvement_gate", 0) for h in history]
act_deltas = [h.get("mean_action_delta", 0) for h in history]
eval_eps = [e["episode"] for e in eval_history]
eval_returns = [e["mean_return"] for e in eval_history]

# Top: eval return (stable, averaged over all scenarios)
ax1 = fig.add_subplot(gs_train[0])
ax1.plot(eval_eps, eval_returns, "b-", linewidth=0.8, marker=".", markersize=3)
ax1.axhline(y=-863, color="gray", linestyle="--", linewidth=0.4, label="Final best")
ax1.set_ylabel("Eval Mean Return", fontsize=7)
ax1.set_ylim(-7000, 0)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=5)
ax1.tick_params(labelsize=6)
# Annotate convergence
ax1.annotate("Converged", xy=(100, -900), fontsize=5, color="gray",
             fontstyle="italic")

# Middle: gate
ax2 = fig.add_subplot(gs_train[1])
ax2.plot(eps_list, gates, "g", linewidth=0.5)
ax2.set_ylabel("Mean Gate", fontsize=7)
ax2.set_ylim(0, 1.1)
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=6)

# Bottom: action delta
ax3 = fig.add_subplot(gs_train[2])
ax3.plot(eps_list, act_deltas, "r", linewidth=0.5)
ax3.set_ylabel("Action Delta RMS", fontsize=7)
ax3.set_xlabel("Episode", fontsize=7)
ax3.grid(True, alpha=0.3)
ax3.tick_params(labelsize=6)

ieee_save(fig, "sirppo_training_curves.eps")


# ===== 6. Gate behavior =====
print("6. Gate behavior...")
sirppo_data = all_runs["SI-RPPO"]["data"]
mask_gate = (sirppo_data["time"] >= 1.0) & (sirppo_data["time"] <= 6.0)

fig = plt.figure(figsize=(3.5, 2.5))
gs_gate = fig.add_gridspec(3, 1, hspace=0.3, left=0.16, right=0.96,
                            bottom=0.10, top=0.95)
tt = sirppo_data["time"].values[mask_gate]

ax1 = fig.add_subplot(gs_gate[0])
ax1.plot(tt, sirppo_data["ddzb"].values[mask_gate], "b", linewidth=0.3)
ax1.set_ylabel("Body Acc", fontsize=7)
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=6)

ax2 = fig.add_subplot(gs_gate[1])
gate_col = "policy_improvement_gate"
if gate_col in sirppo_data.columns:
    ax2.plot(tt, sirppo_data[gate_col].values[mask_gate], "g", linewidth=0.5)
    ax2.set_ylabel("Gate", fontsize=7)
    ax2.set_ylim(-0.05, 1.05)
    ax2.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.3)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=6)

ax3 = fig.add_subplot(gs_gate[2])
force = sirppo_data["Uaf"].values[mask_gate] + sirppo_data["Uar"].values[mask_gate]
ax3.plot(tt, force, "r", linewidth=0.3)
ax3.set_ylabel("Force (N)", fontsize=7)
ax3.set_xlabel("Time (s)", fontsize=7)
ax3.grid(True, alpha=0.3)
ax3.tick_params(labelsize=6)

ieee_save(fig, "sirppo_gate_behavior.eps")


# ===== 7. Convergence comparison: SI-RPPO vs PPO scratch =====
print("7. Convergence comparison...")
with open(ROOT / "results/20260621_191152_ppo/eval_history.json") as f:
    scratch_eval = json.load(f)

fig, ax = plt.subplots(figsize=(3.5, 2.0))
fig.subplots_adjust(left=0.16, right=0.97, bottom=0.18, top=0.90)

ax.plot([e["episode"] for e in eval_history], [e["mean_return"] for e in eval_history],
        "b-", linewidth=0.7, marker=".", markersize=2, label="SI-RPPO")
ax.plot([e["episode"] for e in scratch_eval], [e["mean_return"] for e in scratch_eval],
        "r--", linewidth=0.7, marker="x", markersize=2, label="PPO Scratch")
ax.axhline(y=-869, color="gray", linestyle=":", linewidth=0.4, label="PASSIVE")
ax.set_xlabel("Episode", fontsize=7)
ax.set_ylabel("Eval Mean Return", fontsize=7)
ax.legend(fontsize=6)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=6)
ieee_save(fig, "convergence_comparison.eps")


print(f"\nAll figures saved to {out_dir}")
