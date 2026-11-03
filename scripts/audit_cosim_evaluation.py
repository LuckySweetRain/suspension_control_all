"""Verify evaluated policy outputs against the exported actor and native simulator trace."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    model_path = args.bundle / "policy.ts"
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == manifest["policy_sha256"]
    torch.set_num_threads(1)
    actor = torch.jit.load(str(model_path), map_location="cpu").eval()
    for run in args.runs:
        folder = run / "episode_0000"
        assert (folder / "summary.json").is_file() and not (folder / "failure.json").exists()
        rows = [
            json.loads(line) for line in (folder / "transitions.jsonl").read_text().splitlines()
        ]
        log = (folder / "simulator.log").read_text(encoding="utf-8")
        artifacts = [s[11:].strip() for s in log.splitlines() if s.startswith("Artifacts: ")]
        simulation = Path(artifacts[-1])
        report = json.loads((simulation / "report.json").read_text())
        physical = json.loads((simulation / "trace.json").read_text())
        assert report["passed"] and report["completed_steps"] == len(rows) == len(physical) == 1000
        assert report["terrain_callback_calls"] > 0 and report["image_updates"] > 0
        maximum_error = 0.0
        for step, (row, actual) in enumerate(zip(rows, physical, strict=True)):
            assert not row["terminated"] and row["truncated"] == (step == 999)
            assert np.isfinite(row["exports"]).all() and np.isfinite(row["observation"]).all()
            assert np.allclose(row["exports"], actual["values"], rtol=0, atol=1e-9)
            assert np.allclose(
                np.tile(row["corner_forces"], 2), actual["forces"], rtol=0, atol=1e-6
            )
            with torch.inference_mode():
                expected = actor(torch.tensor([row["observation"]], dtype=torch.float32)).numpy()[0]
            maximum_error = max(maximum_error, float(abs(expected - row["raw_action"]).max()))
        assert maximum_error < 0.01, maximum_error
        result = dict(
            passed=True,
            steps=len(rows),
            policy_sha256=manifest["policy_sha256"],
            max_actor_error_N=maximum_error,
            remote_actions=sum(row.get("inference") is not None for row in rows),
            simulator_report=str(simulation / "report.json"),
            wall_seconds=report["wall_seconds"],
            timing=report["timing"],
            async_planner=report.get("async_planner"),
            scope="policy parity and native physical trace; no convergence or hard-real-time claim",
        )
        (run / "evaluation_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(dict(run=str(run), **result)))


if __name__ == "__main__":
    main()
