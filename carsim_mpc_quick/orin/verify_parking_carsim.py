"""Real CarSim braking/parking regression through Orin's shared MPC controller.

This isolated actuator test does not change the autonomous driving policy. It
uses the legacy v1 endpoint to exclude traffic-light and model variability;
zero-speed image geometry is checked separately by verify_stop_start.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time

import numpy as np

from runtime.carsim import CarSim
from runtime.paths import ROOT
from runtime.protocol import OrinConnection


def verify(host):
    output = ROOT / "results/standstill_carsim" / time.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True)
    digest = hashlib.sha256((ROOT / "results/orin/quick_mpc_data.mat").read_bytes()).hexdigest()
    car, connection = None, None
    rows, report = [], dict(passed=False, scope="CarSim + Orin braking/parking; static road v1")
    try:
        car = CarSim(output, contact=True, driving=True, stop_time=8)
        connection = OrinConnection(host, 55001)
        values = car.initial
        for seq in range(800):
            before = values[:]
            reply, _ = connection.exchange(
                dict(version=1, seq=seq, time=seq * 0.01, exports=before[:77])
            )
            assert reply["seq"] == seq and reply["sha256"] == digest
            assert not reply["fallback"], "QP fallback invalidates the parking regression"
            brake = 0.0 if seq < 50 else 3.0  # MPa; dedicated test stimulus only.
            values = car.step(reply["forces"], driving_inputs=[0, 0, brake])
            assert all(math.isfinite(v) for v in values)
            rows.append(dict(time=car.t, before=before, values=values, reply=reply))
        speeds = np.array([abs(r["before"][76]) / 3.6 for r in rows])
        forces = np.array([r["reply"]["diagnostics"][:2] for r in rows])
        # Require sustained actual standstill, with >0.5 s for the force exit.
        stopped_steps, settled = 0, []
        for i, speed in enumerate(speeds):
            stopped_steps = stopped_steps + 1 if speed <= 0.2 else 0
            if stopped_steps > 50:
                settled.append(i)
        assert len(settled) >= 200, "Vehicle did not remain parked long enough"
        assert np.max(np.abs(forces[settled])) < 1e-9
        assert np.max(np.abs(forces[:50])) > 1, "Test never exercised active MPC"
        damping_error = 0.0
        for i in settled:
            expected = np.tile(np.clip(3 * np.asarray(rows[i]["before"])[60:64], -5000, 5000), 2)
            damping_error = max(
                damping_error, float(np.max(np.abs(expected - rows[i]["reply"]["forces"])))
            )
        assert damping_error < 1e-9
        parked = np.array([rows[i]["values"] for i in settled])
        report.update(
            passed=True,
            steps=len(rows),
            stopped_first_time_s=next(r["time"] for r, s in zip(rows, speeds) if s <= 0.2),
            settled_standstill_samples=len(settled),
            parked_active_force_max_N=np.max(np.abs(forces[settled]), axis=0).tolist(),
            passive_damping_max_error_N=damping_error,
            parked_vz_rms_mps=float(np.sqrt(np.mean((parked[:, 39] / 3.6) ** 2))),
            parked_az_rms_mps2=float(np.sqrt(np.mean((parked[:, 40] * 9.80665) ** 2))),
            final_speed_kph=values[76],
        )
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        if connection is not None:
            connection.close()
        if car is not None:
            car.close()
        (output / "trace.json").write_text(json.dumps(rows), encoding="utf-8")
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        print("Artifacts:", output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.50.2")
    verify(parser.parse_args().host)
