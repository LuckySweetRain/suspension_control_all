from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _format(value) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _latex_escape(value) -> str:
    return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_")


def _summary_path(path: Path) -> Path:
    if path.is_dir():
        return path / "mpc_parameter_seed_sweep_summary.json"
    return path


def _load_summary(path: Path) -> dict:
    summary_path = _summary_path(path)
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    data["_summary_path"] = str(summary_path.resolve())
    return data


def _row(label: str, summary: dict) -> dict:
    seeds = list(summary.get("seeds", []))
    gate_count = sum(1 for row in seeds if row.get("deployment_gate_active"))
    raw_deltas = [
        float(row.get("neural_raw_policy_score", row.get("neural_policy_score", 0.0))) - float(row.get("fixed_score", 0.0))
        for row in seeds
    ]
    mean_raw_delta = float(sum(raw_deltas) / max(1, len(raw_deltas)))
    return {
        "label": label,
        "summary_path": summary["_summary_path"],
        "seed_count": int(summary.get("seed_count", len(seeds))),
        "support_rate": float(summary.get("support_rate", 0.0)),
        "mean_policy_delta_vs_fixed": float(summary.get("mean_policy_delta_vs_fixed", 0.0)),
        "mean_raw_policy_delta_vs_fixed": mean_raw_delta,
        "mean_unsafe_delta_vs_fixed": float(summary.get("mean_unsafe_delta_vs_fixed", 0.0)),
        "gate_count": int(gate_count),
        "gate_rate": float(gate_count / max(1, len(seeds))),
        "noninferiority_status": str(summary.get("noninferiority_status", "")),
        "improvement_status": str(summary.get("improvement_status", "")),
        "train_scenarios": ", ".join(summary.get("train_scenarios", [])),
        "eval_scenarios": ", ".join(summary.get("eval_scenarios", [])),
    }


def _write_markdown(rows: list[dict], path: Path) -> None:
    lines = [
        "# BP-NMPS Evidence Table",
        "",
        "| Experiment | Seeds | Support | Mean Delta | Raw Mean Delta | Unsafe Delta | Gate Rate | Non-inferiority | Improvement |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["label"],
                    str(row["seed_count"]),
                    _format(row["support_rate"]),
                    _format(row["mean_policy_delta_vs_fixed"]),
                    _format(row["mean_raw_policy_delta_vs_fixed"]),
                    _format(row["mean_unsafe_delta_vs_fixed"]),
                    _format(row["gate_rate"]),
                    row["noninferiority_status"],
                    row["improvement_status"],
                ]
            )
            + " |"
        )
    lines += [
        "",
        "Mean Delta is the deployed BP-NMPS policy score minus fixed MPC-lite score; lower is better.",
        "Raw Mean Delta ignores deployment fallback when the source summary reports raw policy scores.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_latex(rows: list[dict], path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{BP-NMPS held-out evidence across seed-sweep probes. Lower delta is better.}",
        r"\label{tab:bp_nmps_evidence}",
        r"\begin{tabular}{lrrrrll}",
        r"\toprule",
        r"Experiment & Seeds & Support & Mean delta & Gate rate & Non-inf. & Improve \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_latex_escape(row['label'])} & "
            f"{row['seed_count']} & "
            f"{_format(row['support_rate'])} & "
            f"{_format(row['mean_policy_delta_vs_fixed'])} & "
            f"{_format(row['gate_rate'])} & "
            f"{_latex_escape(row['noninferiority_status'])} & "
            f"{_latex_escape(row['improvement_status'])} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_mpc_parameter_evidence_table(paths: list[Path], out_dir: Path, labels: list[str] | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = labels or [path.name for path in paths]
    if len(labels) != len(paths):
        raise ValueError("labels and paths must have the same length")
    rows = [_row(label, _load_summary(path)) for label, path in zip(labels, paths)]
    csv_path = out_dir / "bp_nmps_evidence_table.csv"
    markdown_path = out_dir / "bp_nmps_evidence_table.md"
    latex_path = out_dir / "bp_nmps_evidence_table.tex"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    _write_markdown(rows, markdown_path)
    _write_latex(rows, latex_path)
    manifest = {
        "inputs": [str(path.resolve()) for path in paths],
        "labels": labels,
        "csv": str(csv_path.resolve()),
        "markdown": str(markdown_path.resolve()),
        "latex": str(latex_path.resolve()),
        "rows": rows,
    }
    manifest_path = out_dir / "bp_nmps_evidence_table.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path.resolve())
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--labels", default=None, help="Comma-separated labels matching --inputs.")
    parser.add_argument("--out", default="results/bp_nmps_evidence_table")
    args = parser.parse_args()
    paths = [Path(item) if Path(item).is_absolute() else ROOT / item for item in args.inputs]
    labels = [item.strip() for item in args.labels.split(",")] if args.labels else None
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    manifest = build_mpc_parameter_evidence_table(paths, out_dir, labels=labels)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
