"""IJRNC/SPDF paper plotting style helpers.

These settings mirror the MATLAB plotting utilities used for Chapter 5:
``saveIEEEFig.m``, ``setIEEEStyle.m`` and the tight subplot scripts.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


SINGLE_WIDTH = 3.5
DOUBLE_WIDTH = 7.16

CTRL_STYLE = {
    "PASSIVE": {"color": "b", "linestyle": "-", "label": "Passive"},
    "PID": {"color": "k", "linestyle": "-.", "label": "PID"},
    "SPDF": {"color": "r", "linestyle": "-.", "label": "SPDF"},
    "MPC": {"color": "m", "linestyle": "--", "label": "MPC"},
    "PPO_scratch": {"color": "#7f7f7f", "linestyle": "--", "label": "PPO scratch"},
    "SI-RPPO": {"color": "g", "linestyle": "-", "label": "SI-RPPO"},
}


def apply_ijrnc_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "lines.linewidth": 0.5,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "grid.linewidth": 0.35,
            "grid.alpha": 0.35,
            "axes.unicode_minus": False,
            "mathtext.fontset": "stix",
            "ps.fonttype": 42,
            "pdf.fonttype": 42,
        }
    )


def ieee_figure(kind: str = "single", height: float | None = None):
    width = DOUBLE_WIDTH if kind == "double" else SINGLE_WIDTH
    if height is None:
        height = 3.0 if kind == "double" else 1.8
    return plt.figure(figsize=(width, height))


def style_axis(ax, grid: bool = True) -> None:
    ax.tick_params(direction="in", width=0.5, length=2.5, pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    if grid:
        ax.grid(True)


def plot_controller(ax, x, y, controller: str, **kwargs):
    style = dict(CTRL_STYLE.get(controller, {"color": "k", "linestyle": "-", "label": controller}))
    style.update(kwargs)
    label = style.pop("label")
    return ax.plot(x, y, label=label, **style)


def compact_legend(ax, **kwargs):
    legend = ax.legend(frameon=False, handlelength=1.4, borderpad=0.2, labelspacing=0.25, **kwargs)
    if legend is not None:
        legend.get_frame().set_linewidth(0.0)
    return legend


def save_ieee(fig, out_dir: Path, filename: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, format=path.suffix.lstrip("."), bbox_inches="tight")
    plt.close(fig)
    return path
