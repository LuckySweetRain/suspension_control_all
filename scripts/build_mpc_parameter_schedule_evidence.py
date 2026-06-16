"""Aggregate BP-NMPS gain schedules from a seed-sweep result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from envs.mpc_parameter_env import GAIN_KEYS


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return data


def _format(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _base_gains(config_path: Path) -> dict[str, float]:
    config = load_config(config_path)
    gains = config.get("reduced_full_car_mpc", {})
    return {key: float(gains[key]) for key in GAIN_KEYS}


def _policy_actions_csv(seed_row: dict[str, Any]) -> Path | None:
    manifest_path = seed_row.get("benchmark_manifest")
    if not manifest_path:
        return None
    manifest = _load_json(Path(manifest_path))
    for row in manifest.get("rows", []):
        if row.get("variant") == "neural_parameter_policy":
            path = row.get("policy_actions_csv")
            return Path(path) if path else None
    return None


def build_mpc_parameter_schedule_evidence(summary_path: Path, out_dir: Path) -> dict[str, Any]:
    summary_path = Path(summary_path)
    out_dir = Path(out_dir)
    summary = _load_json(summary_path)
    base = _base_gains(Path(summary["config"]))
    rows: list[dict[str, Any]] = []

    for seed_row in summary.get("seeds", []):
        actions_csv = _policy_actions_csv(seed_row)
        if actions_csv is None or not actions_csv.exists():
            continue
        actions = pd.read_csv(actions_csv)
        for _, action_row in actions.iterrows():
            row: dict[str, Any] = {
                "seed": int(seed_row["seed"]),
                "scenario": action_row["scenario"],
                "score": float(action_row["score"]),
                "score_delta_vs_fixed": float(seed_row.get("neural_policy_delta_vs_fixed", 0.0)),
                "deployment_gate_active": bool(action_row.get("deployment_gate_active", seed_row.get("deployment_gate_active", False))),
                "policy_actions_csv": str(actions_csv),
            }
            for key in GAIN_KEYS:
                gain = float(action_row[key])
                action = float(action_row.get(f"action_{key}", 0.0))
                scale = gain / base[key] if base[key] else 0.0
                row[f"action_{key}"] = action
                row[key] = gain
                row[f"scale_{key}"] = scale
                row[f"delta_pct_{key}"] = 100.0 * (scale - 1.0)
            rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No BP-NMPS final action CSV files found from {summary_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = out_dir / "bp_nmps_schedule_evidence.csv"
    df.to_csv(csv_path, index=False)

    aggregate = (
        df.groupby(["scenario", "deployment_gate_active"], as_index=False)
        [[f"delta_pct_{key}" for key in GAIN_KEYS] + ["score_delta_vs_fixed"]]
        .mean()
    )
    aggregate_path = out_dir / "bp_nmps_schedule_evidence_by_scenario.csv"
    aggregate.to_csv(aggregate_path, index=False)

    report_path = out_dir / "bp_nmps_schedule_evidence.md"
    delta_cols = [f"delta_pct_{key}" for key in GAIN_KEYS]
    lines = [
        "# BP-NMPS Schedule Evidence",
        "",
        "This report aggregates the deployed BP-NMPS MPC-lite gain schedule from the formal seed sweep.",
        "",
        f"- Seed-sweep summary: `{summary_path}`",
        f"- Seeds: `{summary.get('seed_count')}`",
        f"- Held-out scenarios: `{', '.join(summary.get('eval_scenarios', []))}`",
        f"- Deployment gate rate: `{_format(float(df['deployment_gate_active'].mean()), 3)}`",
        "",
        "## Per-Seed Schedule",
        "",
        "| Seed | Scenario | Gate | Score delta | " + " | ".join(delta_cols) + " |",
        "| ---: | --- | --- | ---: | " + " | ".join(["---:"] * len(delta_cols)) + " |",
    ]
    for _, row in df.iterrows():
        values = [
            str(int(row["seed"])),
            str(row["scenario"]),
            "on" if bool(row["deployment_gate_active"]) else "off",
            _format(row["score_delta_vs_fixed"], 6),
            *[_format(row[col], 3) for col in delta_cols],
        ]
        lines.append("| " + " | ".join(values) + " |")

    lines += [
        "",
        "## Scenario Mean Schedule",
        "",
        "| Scenario | Gate | Mean score delta | " + " | ".join(delta_cols) + " |",
        "| --- | --- | ---: | " + " | ".join(["---:"] * len(delta_cols)) + " |",
    ]
    for _, row in aggregate.iterrows():
        values = [
            str(row["scenario"]),
            "on" if bool(row["deployment_gate_active"]) else "off",
            _format(row["score_delta_vs_fixed"], 6),
            *[_format(row[col], 3) for col in delta_cols],
        ]
        lines.append("| " + " | ".join(values) + " |")

    lines += [
        "",
        "## Interpretation",
        "",
        "- Gate-on rows keep all MPC-lite gain deltas at zero and deploy the fixed baseline.",
        "- Gate-off rows expose the learned neural gain schedule, so the adaptive behavior can be inspected as small percentage changes to physical MPC-lite gains.",
        "- This evidence supports describing BP-NMPS as an interpretable, baseline-preserving MPC parameter scheduler.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "summary_path": str(summary_path),
        "csv": str(csv_path),
        "aggregate_csv": str(aggregate_path),
        "report": str(report_path),
        "rows": int(len(df)),
        "gate_rate": float(df["deployment_gate_active"].mean()),
        "scenarios": sorted(str(v) for v in df["scenario"].unique()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/mpc_parameter_seed_sweep_formal_gate/mpc_parameter_seed_sweep_summary.json"),
    )
    parser.add_argument("--out", type=Path, default=Path("results/bp_nmps_schedule_evidence"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_mpc_parameter_schedule_evidence(args.summary, args.out)
    print(f"Saved BP-NMPS schedule evidence to {manifest['report']}")


if __name__ == "__main__":
    main()
