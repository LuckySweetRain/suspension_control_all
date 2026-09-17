from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_GAINS = {
    "k_z": 1200.0,
    "k_dz": 1800.0,
    "k_pitch": 950.0,
    "k_dpitch": 700.0,
    "k_roll": 850.0,
    "k_droll": 650.0,
    "k_preview": 18000.0,
}


def _format(value) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.6g}"


def _trajectory_rows(result_dir: Path) -> list[dict]:
    rows = []
    for csv_path in sorted(result_dir.glob("BP_NMPS_*.csv")):
        df = pd.read_csv(csv_path)
        action_cols = [c for c in df.columns if c.startswith("mpc_param_action_")]
        gain_cols = [c for c in df.columns if c.startswith("mpc_param_gain_")]
        if df.empty or not gain_cols:
            continue
        first = df.iloc[0]
        scenario = csv_path.stem.removeprefix("BP_NMPS_")
        row = {"Scenario": scenario, "Trajectory": str(csv_path)}
        for col in action_cols:
            row[col] = float(first[col])
        for col in gain_cols:
            gain_name = col.removeprefix("mpc_param_gain_")
            gain_value = float(first[col])
            row[gain_name] = gain_value
            if gain_name in BASE_GAINS and BASE_GAINS[gain_name] != 0.0:
                scale = gain_value / BASE_GAINS[gain_name]
                row[f"scale_{gain_name}"] = scale
                row[f"delta_pct_{gain_name}"] = 100.0 * (scale - 1.0)
        rows.append(row)
    return rows


def _plot_gains(summary: pd.DataFrame, out_dir: Path) -> Path | None:
    gain_cols = [c for c in summary.columns if c.startswith("k_")]
    if summary.empty or not gain_cols:
        return None
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(summary)), 4.5))
    x = range(len(summary))
    width = 0.8 / max(1, len(gain_cols))
    for idx, col in enumerate(gain_cols):
        offset = (idx - 0.5 * (len(gain_cols) - 1)) * width
        ax.bar([v + offset for v in x], summary[col].to_numpy(), width=width, label=col)
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary["Scenario"], rotation=20, ha="right")
    ax.set_ylabel("Scheduled gain")
    ax.set_title("BP-NMPS scheduled MPC-lite gains")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    path = fig_dir / "bp_nmps_gain_schedule.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _plot_delta_pct(summary: pd.DataFrame, out_dir: Path) -> Path | None:
    delta_cols = [c for c in summary.columns if c.startswith("delta_pct_k_")]
    if summary.empty or not delta_cols:
        return None
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(summary)), 4.5))
    x = range(len(summary))
    width = 0.8 / max(1, len(delta_cols))
    for idx, col in enumerate(delta_cols):
        label = col.removeprefix("delta_pct_")
        offset = (idx - 0.5 * (len(delta_cols) - 1)) * width
        ax.bar([v + offset for v in x], summary[col].to_numpy(), width=width, label=label)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary["Scenario"], rotation=20, ha="right")
    ax.set_ylabel("Delta vs base gain (%)")
    ax.set_title("BP-NMPS MPC-lite gain percentage change")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    path = fig_dir / "bp_nmps_gain_delta_pct.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def summarize_mpc_parameter_schedule(result_dir: Path) -> dict:
    result_dir = Path(result_dir)
    rows = _trajectory_rows(result_dir)
    if not rows:
        raise FileNotFoundError(
            f"No BP_NMPS trajectories with mpc_param_gain_* columns found in {result_dir}"
        )
    summary = pd.DataFrame(rows)
    csv_path = result_dir / "mpc_parameter_schedule_summary.csv"
    summary.to_csv(csv_path, index=False)
    figure_path = _plot_gains(summary, result_dir)
    delta_figure_path = _plot_delta_pct(summary, result_dir)
    report_path = result_dir / "mpc_parameter_schedule_report.md"
    gain_cols = [c for c in summary.columns if c.startswith("k_")]
    delta_cols = [c for c in summary.columns if c.startswith("delta_pct_k_")]
    lines = [
        "# BP-NMPS MPC Parameter Schedule",
        "",
        f"- Result directory: `{result_dir}`",
        f"- Scenarios: `{', '.join(summary['Scenario'])}`",
        "",
        "| Scenario | " + " | ".join(gain_cols) + " |",
        "| --- | " + " | ".join(["---:"] * len(gain_cols)) + " |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| "
            + " | ".join([str(row["Scenario"]), *[_format(row[col]) for col in gain_cols]])
            + " |"
        )
    if delta_cols:
        lines += [
            "",
            "## Percent Change Vs Base Gains",
            "",
            "| Scenario | " + " | ".join(delta_cols) + " |",
            "| --- | " + " | ".join(["---:"] * len(delta_cols)) + " |",
        ]
        for _, row in summary.iterrows():
            lines.append(
                "| "
                + " | ".join([str(row["Scenario"]), *[_format(row[col]) for col in delta_cols]])
                + " |"
            )
    if figure_path:
        rel = figure_path.resolve().relative_to(result_dir.resolve())
        lines += ["", "## Figure", "", f"![BP-NMPS gain schedule]({rel.as_posix()})", ""]
    if delta_figure_path:
        rel = delta_figure_path.resolve().relative_to(result_dir.resolve())
        lines += ["", f"![BP-NMPS gain percentage change]({rel.as_posix()})", ""]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "summary_csv": str(csv_path),
        "report": str(report_path),
        "figure": str(figure_path) if figure_path else None,
        "delta_pct_figure": str(delta_figure_path) if delta_figure_path else None,
        "rows": len(summary),
        "gain_columns": gain_cols,
        "delta_pct_columns": delta_cols,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()
    manifest = summarize_mpc_parameter_schedule(Path(args.result_dir))
    print(f"Saved BP-NMPS parameter schedule report to {manifest['report']}")


if __name__ == "__main__":
    main()
