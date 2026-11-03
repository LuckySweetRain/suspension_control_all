"""Measure batch-one CPU inference on saved real observations, without driving actuators."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    policy = args.bundle / "policy.ts"
    if hashlib.sha256(policy.read_bytes()).hexdigest() != manifest["policy_sha256"]:
        raise ValueError("Policy SHA256 mismatch")
    model = torch.jit.load(str(policy), map_location="cpu").eval()
    observations = np.load(args.bundle / "observations.npy")
    expected = np.load(args.bundle / "expected_forces.npy")
    outputs, durations = [], []
    with torch.inference_mode():
        for _ in range(30):
            model(torch.from_numpy(observations[:1]))
        for row in observations:
            obs = torch.from_numpy(row[None])
            start = time.perf_counter_ns()
            outputs.append(model(obs).numpy().copy())
            durations.append((time.perf_counter_ns() - start) / 1e6)
    error = float(np.max(abs(np.concatenate(outputs) - expected)))
    result = dict(
        platform=platform.platform(),
        architecture=platform.machine(),
        torch=torch.__version__,
        device="cpu",
        threads=args.threads,
        samples=len(outputs),
        max_force_error_N=error,
        parity_passed=bool(np.isfinite(outputs).all() and error < 0.01),
        inference_ms_p50_p95_p99_max=np.percentile(durations, [50, 95, 99, 100]).tolist(),
        over_10ms_fraction=float(np.mean(np.asarray(durations) > 10)),
        scope="batch-one warm CPU inference only; excludes sensing/RPC/projection/actuation",
    )
    (args.bundle / "benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["parity_passed"]:
        raise RuntimeError("Exported policy parity failed")


if __name__ == "__main__":
    main()
