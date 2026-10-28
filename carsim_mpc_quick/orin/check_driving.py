"""Check PID signs/frame handling and optionally real VehicleSim actuator response.

These are integration probes, not learned-model or driving-performance results.
Run with the CARLA Python environment and --carsim for the Windows DLL checks.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from driving import PID, DrivingSession, Pose, Trajectory, TrajectoryTracker


def check_tracker():
    origin = Pose(10.0, -3.0, math.pi / 2)
    points = ((0.5, 2.0, 0.0), (1.0, 4.0, 0.0), (1.5, 6.0, 0.0))
    plan = Trajectory("integration_reference", "synthetic_not_a_model", 1, 4.0, origin, points)
    tracker = TrajectoryTracker()
    control = tracker.step(plan, origin, 0, 4.0)
    assert abs(control.steer_rad) < 1e-10 and control.throttle > 0 and control.brake == 0
    assert control.target_speed_mps == 4
    deceleration = TrajectoryTracker().step(plan, origin, 9.7, 4.0)
    assert deceleration.target_speed_mps == 4 and deceleration.brake > 0
    # Moving left of the anchored path requires a right correction, even after yaw rotation.
    x, y = origin.world_point(0, 0.5)
    control = tracker.step(plan, Pose(x, y, origin.yaw), 5, 4.01)
    assert control.steer_rad < 0 and control.brake > 0 and control.throttle == 0
    assert tracker.step(plan, origin, 4, 4.51).reason == "stale_plan"
    assert tracker.step(plan, origin, 4, 3.9).reason == "future_plan"
    assert tracker.step(None, origin, 4, 4).brake == 1
    for bad in (float("nan"), float("inf")):
        try:
            Trajectory("test", "test", 1, 0, origin, ((0.5, bad, 0), (1, 2, 0)))
        except ValueError:
            pass
        else:
            raise AssertionError("Nonfinite trajectory was accepted")
    pid = PID(1, 1, 0, 1)
    for _ in range(100):
        pid.step(10, 0.01)
    assert pid.integral == 0 and pid.step(-0.2, 0.01) < 0
    return {"status": "passed", "scope": "frame, pedal signs, stale/future/missing, finite, windup"}


def check_tcp_native():
    session = DrivingSession()
    action = dict(
        policy="tcp_official_agent",
        steer_right_positive=0.2,
        throttle=0.4,
        brake=0.3,
        desired_speed_mps=4,
    )
    packet = dict(
        model="tcp-third-party-reproduction",
        checkpoint="synthetic_adapter_check",
        seq=0,
        captured_at=0,
        origin=[0, 0, 0],
        points=[[0.5, 2, 0], [1, 4, 0]],
        tcp_control=action,
    )
    request = dict(pose=[0, 0, 0], speed_mps=10, plan=packet)
    with patch.object(
        TrajectoryTracker, "step", side_effect=AssertionError("Extra PID must not run for TCP")
    ):
        result = session.step(request, 0)
        assert result["throttle"] == 0.4 and result["brake"] == 0.3
        assert result["carsim_inputs"][0] < 0 and result["control_source"] == "tcp_official_agent"
        held = session.step(dict(request, plan=None), 0.01)
        assert held == result
        assert session.step(dict(request, plan=None), -1e-15) == result
        transfuser = DrivingSession()
        tf_action = dict(action, policy="transfuser_cvpr2021_agent")
        tf_packet = dict(packet, model="transfuser-cvpr2021-official", transfuser_control=tf_action)
        tf_result = transfuser.step(dict(request, plan=tf_packet), 0)
        assert tf_result["control_source"] == "transfuser_cvpr2021_agent"
        assert tf_result["throttle"] == 0.4 and tf_result["brake"] == 0.3
    try:
        session.step(dict(request, plan=None, cruise_speed_mps=11), 0.02)
    except ValueError:
        pass
    else:
        raise AssertionError("Removed cruise override was accepted")
    return dict(
        status="passed",
        extra_pid_bypassed=True,
        native_pedals_preserved=True,
        steering_sign="right_to_negative_wheel_deg",
    )


def check_carsim(output, case):
    from runtime.carsim import CarSim

    commands = {
        "coast": (0, 0, 0),
        "left": (60, 0, 0),
        "right": (-60, 0, 0),
        "throttle": (0, 1, 0),
        "brake": (0, 0, 3),
    }
    directory = output / case
    directory.mkdir(parents=True, exist_ok=True)
    car = CarSim(directory, driving=True)
    try:
        initial = car.initial
        for _ in range(150):
            forces = [max(-5000, min(5000, 3 * v)) for v in car.values[22:26]] * 2
            values = car.step(forces, driving_inputs=commands[case])
        report = {
            "imports": car.n_import,
            "exports": car.n_export,
            "initial_speed_kph": initial[76],
            "final_speed_kph": values[76],
            "yaw_deg": values[79],
            "lateral_m": values[78] - initial[78],
            "simulation_s": car.t,
            "steering_wheel_deg_throttle_brake_MPa": commands[case],
        }
        (directory / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    finally:
        car.close()


def check_spatial_preview():
    import numpy as np

    from depth_preview import DepthPreview

    axis = np.arange(-6, 6.001, 0.05)
    x, y = np.meshgrid(axis, axis)
    cloud = np.column_stack((x.ravel(), y.ravel(), (0.01 * x + 0.02 * y).ravel()))
    cfg = {
        "horizon": 20,
        "control_dt": 0.01,
        "preview": {"minimum_speed_mps": 0.1},
        "vehicle": {"a": 1.4, "b": 1.4},
    }
    errors = []
    for yaw in (0, math.pi / 4, math.pi / 2):
        pose = Pose(0, 0, yaw)
        wheels = np.array(
            [
                pose.world_point(a, b)
                for a, b in [(1.4, 0.8), (-1.4, 0.8), (1.4, -0.8), (-1.4, -0.8)]
            ]
        )
        state = np.zeros(39)
        state[30:34], state[34:38], state[38] = wheels[:, 0], wheels[:, 1], 36
        preview = DepthPreview()
        preview.observed_time, preview.points = 0, cloud
        preview.reference_plane, preview.initial_road = np.zeros(3), np.zeros(2)
        result = preview.preview(state, cfg, 0, {"pose": [0, 0, yaw], "steer_rad": 0})
        front = result[0].reshape(-1, 4)[:, 0]
        expected = (1.4 + np.arange(20) * 0.1) * (0.01 * math.cos(yaw) + 0.02 * math.sin(yaw))
        errors.append(float(np.max(np.abs(front - expected))))
        assert errors[-1] < 0.0015 and preview.coverage_fractions == [1, 1]
    # A derivative needs BOTH ends of its interval, including the horizon endpoint.
    preview = DepthPreview()
    preview.observed_time, preview.points = 0, cloud[cloud[:, 0] <= 3.201]
    preview.reference_plane, preview.initial_road = np.zeros(3), np.zeros(2)
    state[30:34], state[34:38] = [1.4, -1.4, 1.4, -1.4], [0.8, 0.8, -0.8, -0.8]
    preview.preview(state, cfg, 0, {"pose": [0, 0, 0], "steer_rad": 0})
    assert preview.coverage_fractions[0] < 1
    return {
        "status": "passed",
        "yaw_deg": [0, 45, 90],
        "height_error_m": errors,
        "unseen_horizon_endpoint_rejected": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carsim", action="store_true")
    parser.add_argument("--spatial-preview", action="store_true")
    parser.add_argument("--case", choices=("coast", "left", "right", "throttle", "brake"))
    parser.add_argument("--output", type=Path, default=Path("results/autodrive_checks"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.case:
        check_carsim(output, args.case)
        return
    report = {
        "tracker": check_tracker(),
        "tcp_native_adapter": check_tcp_native(),
        "learned_models_tested": False,
    }
    if args.spatial_preview:
        report["spatial_depth_preview"] = check_spatial_preview()
    if args.carsim:
        cases = {}
        # VehicleSim owns process-global state: isolate each initial condition.
        for case in ("coast", "left", "right", "throttle", "brake"):
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--case",
                    case,
                    "--output",
                    str(output),
                ],
                check=True,
            )
            cases[case] = json.loads((output / case / "report.json").read_text(encoding="utf-8"))
        assert cases["left"]["lateral_m"] > 1 and cases["right"]["lateral_m"] < -1
        coast = cases["coast"]["final_speed_kph"]
        assert cases["throttle"]["final_speed_kph"] > coast + 5
        assert cases["brake"]["final_speed_kph"] < coast - 5
        report["carsim_actuators"] = {"status": "passed", "cases": cases}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
