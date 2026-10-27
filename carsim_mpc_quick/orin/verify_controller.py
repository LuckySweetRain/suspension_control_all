"""Parity and timing checks against MATLAB-generated numerical fixtures."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.io import loadmat

from controller import Controller


def verify(data, fixture):
    ref = loadmat(fixture, simplify_cells=True)
    controller = Controller(data)
    values, timings, fallback = [], [], []
    for sample in ref["exports"].T:
        start = time.perf_counter()
        forces, _, failed = controller.step(sample)
        timings.append(1000 * (time.perf_counter() - start))
        values.append(forces)
        fallback.append(failed)
    error = float(np.max(np.abs(np.array(values).T - ref["reference"])))
    assert error < 1e-5, f"MATLAB/Python force mismatch: {error} N"
    assert not any(fallback), "Unexpected passive fallback"
    assert np.max(np.abs(ref["diagnostics"][:2])) > 1, "Fixture must exercise active control"
    controller.reset()
    force, _, _ = controller.step(ref["exports"][:, 0])
    assert np.allclose(force, ref["reference"][:, 0], atol=1e-5)
    try:
        controller.step([float("nan")] * 39)
    except ValueError:
        pass
    else:
        raise AssertionError("Nonfinite input was accepted")
    bad = ref["exports"][:, 0].copy()
    bad[0] += 2
    _, _, latched = controller.step(bad)
    assert latched, "Infeasible state did not latch passive damping"
    force, _, latched = controller.step(ref["exports"][:, 0])
    baseline = np.clip(3 * ref["exports"][22:26, 0], -5000, 5000)
    assert latched and np.allclose(force, np.tile(baseline, 2))
    result = {
        "backend": "native" if controller.native is not None else "numpy",
        "samples": len(values),
        "max_force_error_N": error,
        "compute_ms_p50_p95_max": np.percentile(timings, [50, 95, 100]).tolist(),
        "reset_nonfinite_and_latched_fallback": "passed",
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="quick_mpc_data.mat")
    parser.add_argument("--fixture", default="parity_fixture.mat")
    args = parser.parse_args()
    verify(args.data, args.fixture)
