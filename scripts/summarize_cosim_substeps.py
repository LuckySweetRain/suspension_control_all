"""Audit every native 1 ms integration result and compare it with 10 ms sampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    for run in args.runs:
        folder = run / "episode_0000"
        assert (folder / "summary.json").exists() and not (folder / "failure.json").exists()
        rows = [json.loads(s) for s in (folder / "transitions.jsonl").read_text().splitlines()]
        log = (folder / "simulator.log").read_text(encoding="utf-8")
        simulation = Path(
            [s[11:].strip() for s in log.splitlines() if s.startswith("Artifacts: ")][-1]
        )
        trace = json.loads((simulation / "trace.json").read_text())
        report = json.loads((simulation / "report.json").read_text())
        assert report["passed"] and len(trace) == len(rows) == 1000
        samples = []
        for actual, row in zip(trace, rows, strict=True):
            substeps = actual["substeps"]
            assert len(substeps) == 10
            assert np.allclose(substeps[-1]["values"], row["exports"], rtol=0, atol=1e-9)
            assert np.allclose(
                np.tile(row["corner_forces"], 2), actual["forces"], rtol=0, atol=1e-6
            )
            samples.extend(substeps)
        times = np.asarray([s["time"] for s in samples])
        values = np.asarray([s["values"] for s in samples])
        assert np.isfinite(values).all() and np.allclose(np.diff(times), 0.001, rtol=0, atol=1e-9)
        origin = (
            np.asarray(rows[0]["exports"])[44:48] - 80 * np.asarray(rows[1]["observation"])[4:8]
        )
        az = values[:, 40] * 9.80665
        travel = (values[:, 44:48] - origin) / 1000
        coarse = np.asarray([r["exports"] for r in rows])[:, 40] * 9.80665
        result = dict(
            passed=True,
            samples=len(samples),
            interval_s=0.001,
            simulator_report=str(simulation / "report.json"),
            vertical_accel_rms_mps2=float(np.sqrt(np.mean(az**2))),
            vertical_accel_peak_mps2=float(abs(az).max()),
            coarse_vertical_accel_peak_mps2=float(abs(coarse).max()),
            max_relative_suspension_travel_m=float(abs(travel).max()),
            relative_travel_over_80mm_fraction=float(np.mean(abs(travel) > 0.08)),
            min_native_tire_load_N=float(values[:, 52:56].min()),
            scope="all 1 ms solver outputs in this run; not a continuous-time bound",
        )
        (run / "substep_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(dict(run=str(run), **result)))


if __name__ == "__main__":
    main()
