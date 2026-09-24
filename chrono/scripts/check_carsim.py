"""Audit the CarSim D-class parameter reduction and headless Chrono mechanics.

This is a physical/API smoke audit, not CarSim trajectory equivalence or proof
of controller performance. No CarSim solver or GUI is started.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from check_terrain import (  # noqa: E402
    _assert_finite,
    _block_visual_construction,
    _project_visualization_loaded,
    check_vector,
)
from chrono_suspension.config import PROJECT, load_config  # noqa: E402
from chrono_suspension.constitutive import PiecewiseLinearCurve, damper_tangent  # noqa: E402
from chrono_suspension.controllers import linear_matrices  # noqa: E402
from chrono_suspension.physics import SevenDOFModel  # noqa: E402
from chrono_suspension.roads import FourWheelRoad, seven_road_scenarios  # noqa: E402
from chrono_suspension.training import RESULTS_ROOT, limit_worker_threads  # noqa: E402

# Independently transcribed from Run_all.par FD_TABLE, not generated from the
# converted configuration. Source velocities are damper mm/s, forces are N.
SOURCE_VELOCITY_MMPS = np.array(
    [-1410, -720, -390, -210, -90, -20, 0, 20, 90, 200, 390, 760, 1160], dtype=float
)
SOURCE_DAMPER_FORCE_N = np.array(
    [-5008, -3436, -2324, -1792, -1008, -228, 0, 228, 596, 784, 1100, 1796, 2560],
    dtype=float,
)
MOTION_RATIO = np.array([0.65144, 0.65144, 0.797, 0.797])


def check_sources_and_inheritance(config: dict) -> dict:
    metadata = config["metadata"]
    assert metadata["vehicle_preset"] == "carsim_dclass"
    assert metadata["corner_order"] == ["FL", "FR", "RL", "RR"]
    mixed = load_config(PROJECT / "configs" / "carsim_dclass_seven_roads.yaml")
    assert mixed["vehicle"] == config["vehicle"]
    assert mixed["physics"]["gravity_mps2"] == config["physics"]["gravity_mps2"]
    assert {item["kind"] for item in mixed["road"]["scenarios"]} == set(seven_road_scenarios())
    roundtrip = load_config(json.loads(json.dumps(config, allow_nan=False)))
    assert roundtrip == config
    audit_path = PROJECT.parent / metadata["source_audit"]
    if not audit_path.exists():
        return {
            "seven_road_inheritance": "passed",
            "source_files": "not checked: local source audit artifact is unavailable",
            "audit_path": str(audit_path),
        }
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["schema"] == "chrono-carsim-static-input-audit-v1"
    for row in audit["parameters"]:
        section, key = row["target"].split(".")
        assert config[section][key] == row["mapped_value"], row["target"]
        assert row["source_lines"] and row["source_unit"] and row["target_unit"]
    sources = {}
    for name, source in audit["sources"].items():
        path = Path(source["path"])
        # Do not turn a moved/removed proprietary CarSim installation into a
        # requirement for running the self-contained Chrono physical audit.
        if not path.exists():
            sources[name] = {"status": "unavailable", "path": str(path)}
            continue
        content = path.read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        assert actual == source["sha256"], f"Source has changed since import: {path}"
        assert len(content) == source["size_bytes"]
        sources[name] = {"status": "hash_verified", "sha256": actual, "path": str(path)}
    return {
        "seven_road_inheritance": "passed",
        "resolved_config_json_roundtrip": "passed",
        "mapped_parameters_crosschecked": len(audit["parameters"]),
        "sources": sources,
    }


def plant(config: dict) -> SevenDOFModel:
    vehicle = config["vehicle"]
    road = FourWheelRoad(
        {**config["road"], "kind": "flat"},
        vehicle["front_distance_m"],
        vehicle["rear_distance_m"],
        3017,
    )
    return SevenDOFModel(config, road)


def check_parameter_mapping(config: dict) -> dict:
    """Check SI conversion against independent source values and mass balance."""
    vehicle = config["vehicle"]
    expected = {
        "sprung_mass_kg": 1370.0,
        "inertia_kgm2": [671.3, 1972.8, 2315.3],
        "front_distance_m": 1.110,
        "rear_distance_m": 1.756,
        "front_half_track_m": 0.775,
        "rear_half_track_m": 0.775,
        "unsprung_mass_kg": [66.95, 66.95, 54.6, 54.6],
        "tire_stiffness_npm": [268000.0] * 4,
        "tire_damping_nspm": [0.0] * 4,
        "tire_radius_m": 0.334,
        "suspension_stiffness_npm": np.array([153.0, 153.0, 82.0, 82.0]) * 1000 * MOTION_RATIO**2,
        "tire_force_limit_n": [100000.0] * 4,
        "anti_roll_stiffness_nmprad": np.array([384.0, 344.0]) * 180 / np.pi,
    }
    for key, value in expected.items():
        np.testing.assert_allclose(vehicle[key], value, rtol=1e-11, atol=1e-10, err_msg=key)
    np.testing.assert_allclose(config["physics"]["gravity_mps2"], 9.80665, rtol=0, atol=1e-12)
    assert config["physics"]["dt"] == 0.0005
    assert vehicle["suspension_mount_height_m"] == 0.0
    g, mass = 9.80665, 1370.0
    wheel_mass = np.array([66.95, 66.95, 54.6, 54.6])
    preload = mass * g * np.array([1.756, 1.756, 1.110, 1.110]) / (2 * 2.866)
    static_load = preload + wheel_mass * g
    model = plant(config)
    try:
        np.testing.assert_allclose(model.preload, preload, rtol=0, atol=1e-9)
        np.testing.assert_allclose(model.static_load, static_load, rtol=0, atol=1e-9)
        np.testing.assert_allclose(np.sum(model.static_load), 1613.1 * g, rtol=0, atol=1e-8)
        np.testing.assert_allclose(model.wheel_z0, 0.334 - static_load / 268000, atol=1e-12)
        assert model.mechanical_dof == 7
        assert model.system.GetNumCoordsVelLevel() == 30
        assert model.system.GetNumConstraints() == 23
    finally:
        model.close()
    return {
        "total_mass_kg": 1613.1,
        "sprung_corner_preload_n": preload.tolist(),
        "static_tire_load_n": static_load.tolist(),
        "physics_dt_s": config["physics"]["dt"],
        "gravity_mps2": g,
        "mechanical_dof": 7,
        "anti_roll_conversion": "source N m/deg multiplied by 180/pi gives N m/rad",
    }


def check_damper_callbacks(config: dict) -> dict:
    """Check every source node, dissipation, interpolation and end extrapolation."""
    model = plant(config)
    report = []
    try:
        for i, ratio in enumerate(MOTION_RATIO):
            velocity = SOURCE_VELOCITY_MMPS / (1000 * ratio)
            force = SOURCE_DAMPER_FORCE_N * ratio
            table = config["vehicle"]["damper_tables"][i]
            np.testing.assert_allclose(table["velocity_mps"], velocity, rtol=2e-12, atol=1e-13)
            np.testing.assert_allclose(table["force_n"], force, rtol=2e-12, atol=1e-10)
            curve = PiecewiseLinearCurve(table, "velocity_mps")
            spring, callback = model.suspensions[i], model.functors[2 * i]
            rest = spring.GetRestLength()
            node_errors = []
            for speed, expected_force in zip(velocity, force, strict=True):
                actual = callback.evaluate(0, rest, rest, -speed, spring) - model.preload[i]
                node_errors.append(abs(actual - expected_force))
                np.testing.assert_allclose(actual, expected_force, rtol=2e-12, atol=1e-8)
            # Discrete and continuous passivity: F(v_jounce) * v_jounce >= 0.
            probe = np.linspace(2 * velocity[0], 2 * velocity[-1], 301)
            assert np.all(curve(probe) * probe >= -1e-9)
            for end, neighbor in ((0, 1), (-1, -2)):
                query = velocity[end] + (velocity[end] - velocity[neighbor]) * 0.6
                expected_force = force[end] + (force[end] - force[neighbor]) * 0.6
                actual = callback.evaluate(0, rest, rest, -query, spring) - model.preload[i]
                np.testing.assert_allclose(actual, expected_force, rtol=2e-12, atol=1e-8)
            midpoint = (velocity[:-1] + velocity[1:]) / 2
            np.testing.assert_allclose(curve(midpoint), (force[:-1] + force[1:]) / 2, atol=1e-8)
            report.append(
                {"corner": i, "motion_ratio": ratio, "max_node_error_n": max(node_errors)}
            )
        expected_tangent = 11400.0 * MOTION_RATIO**2
        np.testing.assert_allclose(damper_tangent(config["vehicle"]), expected_tangent, atol=1e-7)
        # Changing the fallback linear damper must have no effect when tables exist.
        altered = deepcopy(config)
        altered["vehicle"]["suspension_damping_nspm"] = [1e8] * 4
        altered["vehicle"]["compression_damping_ratio"] = 9.0
        altered["vehicle"]["rebound_damping_ratio"] = 8.0
        other = plant(load_config(altered))
        try:
            for i in range(4):
                for speed in (-0.7, 0.0, 0.4):
                    spring, changed = model.suspensions[i], other.suspensions[i]
                    first = model.functors[2 * i].evaluate(
                        0, spring.GetRestLength(), spring.GetRestLength(), speed, spring
                    )
                    second = other.functors[2 * i].evaluate(
                        0, changed.GetRestLength(), changed.GetRestLength(), speed, changed
                    )
                    np.testing.assert_allclose(first, second, rtol=0, atol=1e-10)
        finally:
            other.close()
    finally:
        model.close()
    return {
        "corners": report,
        "linear_fallback_not_double_counted": True,
        "extrapolation": "linear",
    }


def check_stop_callbacks(config: dict) -> dict:
    isolated = deepcopy(config)
    isolated["vehicle"]["anti_roll_stiffness_nmprad"] = [0.0, 0.0]
    # Intentionally large generic stops: table-equipped force laws must ignore them.
    isolated["vehicle"]["bump_stop_npm2"] = 9e9
    model = plant(load_config(isolated))
    report = []
    try:
        for i, ratio in enumerate(MOTION_RATIO):
            source_jounce = np.array([58, 59, 60] if i < 2 else [68, 69, 70]) / 1000
            source_rebound = np.array([-40, -39, -38] if i < 2 else [-50, -49, -48]) / 1000
            for key, displacement, forces in (
                ("jounce_stop_tables", source_jounce, np.array([0, 0, 10000.0])),
                ("rebound_stop_tables", source_rebound, np.array([-10000.0, 0, 0])),
            ):
                table = config["vehicle"][key][i]
                np.testing.assert_allclose(table["compression_m"], displacement / ratio, atol=1e-12)
                np.testing.assert_allclose(table["force_n"], forces * ratio, atol=1e-9)
            spring, callback = model.suspensions[i], model.functors[2 * i]
            rest = spring.GetRestLength()
            queries = np.r_[source_jounce / ratio, source_rebound / ratio, -0.11, 0, 0.12]
            errors = []
            for compression in queries:
                # Independent analytic linear ramp; inactive ends remain zero.
                jounce = max(0, compression - source_jounce[1] / ratio) * 1e7 * ratio**2
                rebound = min(0, compression - source_rebound[1] / ratio) * 1e7 * ratio**2
                expected = jounce + rebound
                total = callback.evaluate(0, rest, rest - compression, 0, spring)
                actual = total - model.preload[i] - model.ks[i] * compression
                errors.append(abs(actual - expected))
                np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=1e-7)
                assert actual * compression >= -1e-7
            report.append({"corner": i, "max_stop_force_error_n": max(errors)})
    finally:
        model.close()
    return {"corners": report, "generic_stops_not_double_counted": True, "force_cap": False}


def _set_state(model: SevenDOFModel, q: np.ndarray, qd: np.ndarray) -> None:
    import pychrono as ch

    model.chassis.SetPos(ch.ChVector3d(0, 0, model.body_z0 + q[0]))
    model.chassis.SetRot(ch.QuatFromAngleY(q[2]) * ch.QuatFromAngleX(q[1]))
    model.chassis.SetPosDt(ch.ChVector3d(0, 0, qd[0]))
    model.chassis.SetAngVelParent(ch.ChVector3d(qd[1] * np.cos(q[2]), qd[2], -qd[1] * np.sin(q[2])))
    for i, wheel in enumerate(model.wheels):
        wheel.SetPos(ch.ChVector3d(*model.xy[i], model.wheel_z0[i] + q[i + 3]))
        wheel.SetPosDt(ch.ChVector3d(0, 0, qd[i + 3]))
    model.system.Update(0.0)


def _suspension_scalars(model: SevenDOFModel) -> np.ndarray:
    return np.array(
        [
            model.functors[2 * i].evaluate(
                0, spring.GetRestLength(), spring.GetLength(), spring.GetVelocity(), spring
            )
            for i, spring in enumerate(model.suspensions)
        ]
    )


def check_anti_roll(config: dict) -> dict:
    without = deepcopy(config)
    without["vehicle"]["anti_roll_stiffness_nmprad"] = [0.0, 0.0]
    model, baseline = plant(config), plant(load_config(without))
    widths = 2 * np.array([0.775, 0.775])
    wheel_rate = np.array([384.0, 344.0]) * 180 / np.pi / widths**2
    report = []
    try:
        for axle in range(2):
            left, right = 2 * axle, 2 * axle + 1
            for pattern, travel in (
                ("same_phase", [0.01, 0.01]),
                ("opposite_phase", [0.01, -0.01]),
            ):
                q = np.zeros(7)
                q[[left + 3, right + 3]] = travel
                for item in (model, baseline):
                    _set_state(item, q, np.zeros(7))
                actual = _suspension_scalars(model) - _suspension_scalars(baseline)
                expected = np.zeros(4)
                expected[left] = wheel_rate[axle] * (travel[0] - travel[1])
                expected[right] = -expected[left]
                np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=1e-8)
                assert abs(actual.sum()) < 1e-8
                assert float(np.dot(-actual, q[3:])) <= 1e-9
                report.append(
                    {"axle": axle + 1, "pattern": pattern, "delta_force_n": actual.tolist()}
                )
    finally:
        model.close()
        baseline.close()
    return {"axle_wheel_cross_rates_npm": wheel_rate.tolist(), "cases": report}


def _generalized_force(model: SevenDOFModel, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
    _set_state(model, q, qd)
    body = model.chassis.GetPos()
    center = np.array([body.x, body.y, body.z])
    forces, torque, tire_forces = [], np.zeros(3), []
    scalars = _suspension_scalars(model)
    for i, spring in enumerate(model.suspensions):
        first, second = spring.GetPoint1Abs(), spring.GetPoint2Abs()
        top, bottom = (
            np.array([first.x, first.y, first.z]),
            np.array([second.x, second.y, second.z]),
        )
        force = scalars[i] * (top - bottom) / spring.GetLength()
        forces.append(force)
        torque += np.cross(top - center, force)
        tire = model.tires[i]
        tire_forces.append(
            model.functors[2 * i + 1].evaluate(
                0, tire.GetRestLength(), tire.GetLength(), tire.GetVelocity(), tire
            )
        )
    forces = np.asarray(forces)
    gravity = model.config["physics"]["gravity_mps2"]
    return np.r_[
        np.sum(forces[:, 2]) - model.config["vehicle"]["sprung_mass_kg"] * gravity,
        torque[0] * np.cos(q[2]) - torque[2] * np.sin(q[2]),
        torque[1],
        -forces[:, 2] + np.asarray(tire_forces) - model.mu * gravity,
    ]


def check_linearization(config: dict) -> dict:
    model = plant(config)
    try:
        a, b, _ = linear_matrices(config)
        vehicle = config["vehicle"]
        mass = np.r_[vehicle["sprung_mass_kg"], vehicle["inertia_kgm2"][:2], model.mu]
        analytic_k, analytic_c = -mass[:, None] * a[7:, :7], -mass[:, None] * a[7:, 7:]
        finite_k, finite_c, finite_b = np.zeros((7, 7)), np.zeros((7, 7)), np.zeros((7, 4))
        zero, epsilon = np.zeros(7), 1e-6
        np.testing.assert_allclose(_generalized_force(model, zero, zero), 0, atol=1e-8)
        for column in range(7):
            perturbation = np.eye(7)[column] * epsilon
            finite_k[:, column] = -(
                _generalized_force(model, perturbation, zero)
                - _generalized_force(model, -perturbation, zero)
            ) / (2 * epsilon)
            finite_c[:, column] = -(
                _generalized_force(model, zero, perturbation)
                - _generalized_force(model, zero, -perturbation)
            ) / (2 * epsilon)
        for column in range(4):
            model.actuator_force = np.eye(4)[column] * 1e-3
            positive = _generalized_force(model, zero, zero)
            model.actuator_force = -np.eye(4)[column] * 1e-3
            negative = _generalized_force(model, zero, zero)
            finite_b[:, column] = (positive - negative) / 2e-3
        np.testing.assert_allclose(finite_k, analytic_k, rtol=2e-8, atol=2e-5)
        np.testing.assert_allclose(finite_c, analytic_c, rtol=2e-8, atol=2e-5)
        np.testing.assert_allclose(finite_b, mass[:, None] * b[7:], rtol=1e-6, atol=1e-7)
        return {
            "max_stiffness_jacobian_error": float(np.max(np.abs(finite_k - analytic_k))),
            "max_damping_jacobian_error": float(np.max(np.abs(finite_c - analytic_c))),
            "max_force_jacobian_error": float(np.max(np.abs(finite_b - mass[:, None] * b[7:]))),
            "coordinate_perturbation": epsilon,
            "interpretation": "Mechanical static tangent; actuator dynamics remain separate states.",
        }
    finally:
        model.close()


def check_static(config: dict) -> dict:
    model = plant(config)
    peak_q, peak_qd, peak_acc, peak_load_error = 0.0, 0.0, 0.0, 0.0
    try:
        for index in range(round(2.0 / model.dt)):
            model.step(np.zeros(4))
            if index % 20 == 0:
                q, qd = model.coordinates()
                metrics = model.metrics()
                _assert_finite((q, qd, metrics))
                peak_q = max(peak_q, float(np.max(np.abs(q))))
                peak_qd = max(peak_qd, float(np.max(np.abs(qd))))
                peak_acc = max(peak_acc, abs(float(metrics["body_acc_mps2"])))
                peak_load_error = max(
                    peak_load_error,
                    float(np.max(np.abs(metrics["tire_load_n"] - model.static_load))),
                )
        assert peak_q < 1e-5 and peak_qd < 1e-4 and peak_acc < 1e-3 and peak_load_error < 0.1
    finally:
        model.close()
    assert len(model.system.GetBodies()) == 0
    return {
        "duration_s": 2.0,
        "peak_q": peak_q,
        "peak_qd": peak_qd,
        "peak_acc_mps2": peak_acc,
        "peak_static_load_error_n": peak_load_error,
        "bodies_after_close": 0,
    }


def check_seven_roads(config: dict) -> dict:
    from chrono_suspension.env import ChronoSuspensionEnv

    results = {}
    short_config = deepcopy(config)
    short_config["physics"]["episode_seconds"] = 2.3
    for name, scenario in seven_road_scenarios().items():
        for controller in ("passive", "skyhook"):
            env = ChronoSuspensionEnv(short_config, observation_mode="privileged", render_mode=None)
            travel_peak, minimum_load, reward_sum = 0.0, float("inf"), 0.0
            try:
                obs, info = env.reset(seed=3017, options={"scenario": scenario})
                _assert_finite((obs, info))
                terminated = truncated = False
                while not (terminated or truncated):
                    obs, reward, terminated, truncated, info = env.step(
                        env.baseline_action(controller)
                    )
                    _assert_finite((obs, reward, info, env.model.metrics()))
                    assert env.model.mechanical_dof == 7 and env.viewer is None
                    metrics = env.model.metrics()
                    travel_peak = max(
                        travel_peak, float(np.max(np.abs(metrics["suspension_travel_m"])))
                    )
                    minimum_load = min(minimum_load, float(np.min(metrics["tire_load_n"])))
                    reward_sum += reward
                assert not terminated, f"{name}/{controller} terminated: {info}"
                np.testing.assert_allclose(env.model.time, 2.3, atol=1e-8)
                results[f"{name}/{controller}"] = {
                    "duration_s": env.model.time,
                    "terminated": terminated,
                    "truncated": truncated,
                    "travel_peak_m": travel_peak,
                    "minimum_tire_load_n": minimum_load,
                    "reward_sum": reward_sum,
                }
            finally:
                env.close()
    return {
        "cases": results,
        "interpretation": "Finite short rollouts; not a performance comparison.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs" / "carsim_dclass.yaml")
    parser.add_argument(
        "--with-vector", action="store_true", help="Also exercise two spawn workers"
    )
    args = parser.parse_args()
    limit_worker_threads()
    config = load_config(args.config)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = RESULTS_ROOT / "carsim" / f"checks_{timestamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "created_utc": timestamp,
        "config": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "checks": {},
    }
    checks = [
        ("sources_and_inheritance", check_sources_and_inheritance),
        ("parameter_mapping", check_parameter_mapping),
        ("damper_source_nodes", check_damper_callbacks),
        ("stop_source_nodes", check_stop_callbacks),
        ("anti_roll_coupling", check_anti_roll),
        ("static_linearization", check_linearization),
        ("static_equilibrium", check_static),
        ("seven_road_finite_rollouts", check_seven_roads),
    ]
    if args.with_vector:
        checks.append(("two_worker_spawn", check_vector))
    guard, blocked = _block_visual_construction()
    with guard:
        for name, check in checks:
            started = time.monotonic()
            try:
                detail = check(config)
                report["checks"][name] = {"status": "passed", "detail": detail}
            except Exception as error:
                report["checks"][name] = {
                    "status": "failed",
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
            report["checks"][name]["wall_seconds"] = time.monotonic() - started
            print(f"{name}: {report['checks'][name]['status']}", flush=True)
    report["blocked_visual_constructors"] = blocked
    report["project_visualization_loaded"] = _project_visualization_loaded()
    report["all_requested_checks_passed"] = not _project_visualization_loaded() and all(
        item["status"] == "passed" for item in report["checks"].values()
    )
    report["skipped"] = [] if args.with_vector else ["two_worker_spawn"]
    report["limitations"] = (
        "No CarSim solver invocation, trajectory equivalence, full suspension kinematics, "
        "training convergence, or real-vehicle validation is claimed."
    )
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Report: {output}")
    if not report["all_requested_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
