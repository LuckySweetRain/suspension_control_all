"""Verify passive-force equivalence, independent solver clocks and fair warmup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from controller import Controller


def verify(folder):
    folder = Path(folder)
    report = json.loads((folder / "report.json").read_text())
    trace = json.loads((folder / "trace.json").read_text())
    initial = [
        json.loads((folder / role / "initial.json").read_text())["values"]
        for role in ["active", "passive"]
    ]
    active = json.loads((folder / "active/trace.json").read_text())
    passive = json.loads((folder / "passive/trace.json").read_text())
    assert report["passed"], report
    assert len(trace) == len(active) == len(passive) == report["steps"]
    np.testing.assert_array_equal(initial[0], initial[1])
    assert len(set(report["vehicle_parameter_hashes"])) == 1
    warmup_error = 0.0
    baseline_error = 0.0
    previous_passive = initial[1]
    reference = Controller(
        str(Path(__file__).resolve().parents[2] / "results/orin/quick_mpc_data.mat")
    )
    warmup_steps = 0
    for seq, (row, a, p) in enumerate(zip(trace, active, passive)):
        assert a["seq"] == p["seq"] == seq
        assert abs(a["time"] - p["time"]) < 1e-10
        assert abs(a["time"] - (seq + 1) * 0.01) < 1e-8
        baseline = row["reply"]["passive_baseline_config"]
        assert baseline["damper_gain_N_per_mmps"] == reference.c["carsim_damper_gain"]
        assert baseline["corner_force_limit_N"] == reference.c["corner_force_limit"]
        # Compare to the existing MATLAB-parity controller's passive branch,
        # independently of the comparison runner's force helper under test.
        expected, diagnostics, fallback = reference.step(previous_passive[:77], contact_active=True)
        assert not fallback and diagnostics[:2] == [0.0, 0.0]
        np.testing.assert_allclose(row["passive_forces"], expected, atol=1e-10, rtol=0)
        baseline_error = max(
            baseline_error, float(np.max(np.abs(np.array(row["passive_forces"]) - expected)))
        )
        np.testing.assert_array_equal(p["forces"], row["passive_forces"])
        np.testing.assert_array_equal(a["values"], row["active"])
        np.testing.assert_array_equal(p["values"], row["passive"])
        if row["reply"]["preview_warmup"]:
            warmup_steps += 1
            np.testing.assert_array_equal(row["reply"]["diagnostics"][:2], [0, 0])
            warmup_error = max(
                warmup_error, float(np.max(np.abs(np.asarray(row["active"]) - row["passive"])))
            )
        previous_passive = row["passive"]
    assert warmup_steps > 0 and warmup_error < 1e-8, warmup_error
    assert sum("camera_frame" in row for row in trace) == report["camera_updates"]
    result = dict(
        passed=True,
        steps=len(trace),
        independent_clocks_matched=True,
        passive_damping_formula_max_error_N=baseline_error,
        equal_vehicle_parameter_files=True,
        equal_initial_states=True,
        passive_warmup_steps=warmup_steps,
        warmup_state_max_difference=warmup_error,
    )
    (folder / "fairness_check.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    verify(parser.parse_args().folder)
