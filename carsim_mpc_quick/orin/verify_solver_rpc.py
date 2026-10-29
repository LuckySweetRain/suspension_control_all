"""Compare a real Windows DLL run with WSL RPC, including non-flat terrain.

Run --case local in Windows, then --case remote in WSL with the same folder.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import json
import math
import os
from pathlib import Path

import numpy as np

from carla_terrain import CarlaTerrain
from runtime.carsim import CarSim as LocalCarSim
from runtime.paths import ROOT
from runtime.solver_rpc import RemoteCarSim, send_frame


def test_terrain():
    # Keep the original terrain query implementation as the independent baseline.
    terrain = CarlaTerrain.__new__(CarlaTerrain)
    terrain.calls, terrain.rays = 0, 0
    terrain.errors, terrain.cache = [], {}
    terrain.origin_cs, terrain.origin_calibrated = (0, 0), True

    def height(x, y):
        x, y = round(x / 0.05) * 0.05, round(y / 0.05) * 0.05
        return 0.003 * math.sin(x / 3) + 0.002 * y

    terrain.height = height
    pointer = ct.POINTER(ct.c_double)
    signature = ct.CFUNCTYPE(
        ct.c_int, ct.c_double, ct.c_double, ct.c_double, *([pointer] * 5), ct.c_void_p
    )
    terrain.callback = signature(terrain.query)
    return terrain


def verify(case, folder):
    folder = Path(folder).resolve()
    (folder / case).mkdir(parents=True, exist_ok=True)
    terrain = test_terrain()
    cls = LocalCarSim if case == "local" else RemoteCarSim
    car = cls(folder / case, terrain=terrain, contact=True, driving=True, stop_time=2)
    states = [car.initial]
    try:
        for seq in range(100):
            force = np.tile(3 * np.asarray(states[-1])[60:64], 2).tolist()
            values = car.step(
                force, driving_inputs=[0.5, 0.15 if seq < 50 else 0, 0 if seq < 50 else 1]
            )
            states.append(values)
        assert car.n_import == 15 and car.n_export == 82
        assert abs(car.t - 1) < 1e-8
        assert terrain.calls > 100
        result = dict(
            case=case,
            platform=os.name,
            coordinator_pid=os.getpid(),
            dll_owner_pid=getattr(car, "owner_pid", os.getpid()),
            steps=100,
            terrain_callbacks=terrain.calls,
            passed=True,
        )
        (folder / case / "states.json").write_text(json.dumps(states))
        if case == "remote":
            expected = np.asarray(json.loads((folder / "local/states.json").read_text()))
            error = float(np.max(np.abs(np.asarray(states) - expected)))
            assert error < 1e-9, "Cross-platform state mismatch: " + str(error)
            result["max_state_error"] = error
            try:
                send_frame(None, dict(value=float("nan")))
            except ValueError:
                result["nonfinite_packet_rejected"] = True
            else:
                raise AssertionError("Nonfinite packet accepted")
        (folder / case / "report.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return result
    finally:
        car.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("local", "remote"), required=True)
    parser.add_argument(
        "--folder", type=Path, default=ROOT / "results/wsl_split_20261029/solver_check"
    )
    args = parser.parse_args()
    verify(args.case, args.folder)
