"""Run reproducible physical/API checks without pytest; write a JSON audit."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chrono_suspension.config import load_config  # noqa: E402
from chrono_suspension.physics import SevenDOFModel  # noqa: E402
from chrono_suspension.roads import FourWheelRoad  # noqa: E402
from chrono_suspension.visualization import result_path, validate_frame  # noqa: E402

_OPEN_MODELS: list[SevenDOFModel] = []


def plant(cfg: dict, kind: str = "flat", seed: int = 42) -> SevenDOFModel:
    road_cfg = {**cfg["road"], "kind": kind}
    vehicle = cfg["vehicle"]
    road = FourWheelRoad(road_cfg, vehicle["front_distance_m"], vehicle["rear_distance_m"], seed)
    model = SevenDOFModel(cfg, road)
    _OPEN_MODELS.append(model)
    return model


def rollout(model: SevenDOFModel, duration: float, force_n: float = 0) -> dict[str, np.ndarray]:
    sample_every = round(0.01 / model.dt)
    frames = {"q": [], "qd": [], "acceleration": [], "travel": [], "tire_load": []}
    for index in range(round(duration / model.dt)):
        model.step(np.full(4, force_n))
        if (index + 1) % sample_every == 0:
            q, qd = model.coordinates()
            metrics = model.metrics()
            frames["q"].append(q)
            frames["qd"].append(qd)
            frames["acceleration"].append(metrics["body_acc_mps2"])
            frames["travel"].append(metrics["suspension_travel_m"])
            frames["tire_load"].append(metrics["tire_load_n"])
    arrays = {key: np.asarray(value) for key, value in frames.items()}
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise AssertionError("Nonfinite rollout state or metric")
    return arrays


def check_dof(cfg: dict) -> dict:
    model = plant(cfg)
    velocity_coordinates = model.system.GetNumCoordsVelLevel()
    constraints = model.system.GetNumConstraints()
    assert model.mechanical_dof == 7, model.mechanical_dof
    assert velocity_coordinates == 30 and constraints == 23
    return {
        "mechanical_dof": 7,
        "velocity_coordinates": velocity_coordinates,
        "constraints": constraints,
        "definition": "heave, roll, pitch, four wheel heights",
    }


def check_static(cfg: dict) -> dict:
    model = plant(cfg)
    data = rollout(model, 10.0)
    max_displacement = float(np.max(np.abs(data["q"])))
    max_velocity = float(np.max(np.abs(data["qd"])))
    load_error = float(np.max(np.abs(data["tire_load"] - model.static_load)))
    max_acc = float(np.max(np.abs(data["acceleration"])))
    assert max_displacement < 1e-5 and max_velocity < 1e-4
    assert max_acc < 1e-3 and load_error < 0.1
    return {
        "duration_s": 10,
        "max_abs_generalized_displacement": max_displacement,
        "max_abs_generalized_velocity": max_velocity,
        "max_body_acc_mps2": max_acc,
        "max_static_tire_load_error_n": load_error,
        "limits": {"displacement": 1e-5, "velocity": 1e-4, "acceleration": 1e-3, "load_n": 0.1},
    }


def check_symmetry(cfg: dict) -> dict:
    data = rollout(plant(cfg, "bump"), 2.0)
    peak_roll = float(np.max(np.abs(data["q"][:, 1])))
    peak_pitch = float(np.max(np.abs(data["q"][:, 2])))
    assert peak_roll < 1e-7, peak_roll
    assert peak_pitch > 1e-5, "Symmetric bump produced no pitch excitation"
    return {"peak_roll_rad": peak_roll, "peak_pitch_rad": peak_pitch, "roll_limit_rad": 1e-7}


def check_single_side(cfg: dict) -> dict:
    data = rollout(plant(cfg, "left_bump"), 2.0)
    peak_roll = float(np.max(np.abs(data["q"][:, 1])))
    assert peak_roll > 1e-4, peak_roll
    return {"peak_roll_rad": peak_roll, "minimum_expected_rad": 1e-4}


def check_force_direction(cfg: dict) -> dict:
    model = plant(cfg)
    # Cover the configured input delay plus a short actuator transient.
    duration = cfg["actuator"]["delay_s"] + 0.01
    for _ in range(round(duration / model.dt)):
        model.step(np.full(4, 500.0))
    q, qd = model.coordinates()
    assert q[0] > 0 and qd[0] > 0 and np.all(q[3:] < 0) and np.all(qd[3:] < 0)
    assert np.all(model.actuator_force > 0)
    return {
        "command_n": 500,
        "duration_s": duration,
        "chassis_displacement_m": float(q[0]),
        "wheel_displacements_m": q[3:].tolist(),
        "actual_actuator_force_n": model.actuator_force.tolist(),
    }


def check_actuator_limits(cfg: dict) -> dict:
    """Exercise delayed over-range and sign-reversing demands at physics frequency."""
    model = plant(cfg)
    actuator = cfg["actuator"]
    limit = float(actuator["force_limit_n"])
    request = np.full(4, 2 * limit)
    for _ in range(model.delay_steps):
        model.step(request)
        assert np.array_equal(model.actuator_force, np.zeros(4)), (
            "Actuation before configured delay"
        )
    model.step(request)
    assert np.all(model.actuator_force > 0), "No actuation on first step after delay"
    first_response_end = model.time
    previous = model.actuator_force.copy()
    peak_force, peak_slew, peak_power = float(np.max(np.abs(previous))), 0.0, 0.0
    for index in range(round(0.4 / model.dt)):
        # Alternating unreachable demands ensure both saturation and slew limits
        # are exercised, including a reversal rather than only a zero-force start.
        sign = 1 if (index // round(0.1 / model.dt)) % 2 == 0 else -1
        model.step(sign * request)
        actual = model.actuator_force.copy()
        peak_force = max(peak_force, float(np.max(np.abs(actual))))
        peak_slew = max(peak_slew, float(np.max(np.abs(actual - previous))) / model.dt)
        peak_power = max(peak_power, float(np.max(np.abs(model.metrics()["mechanical_power_w"]))))
        assert np.all(np.abs(actual) <= limit + 1e-8), "Actual force exceeds hardware force bound"
        assert np.all(np.abs(actual - previous) <= actuator["rate_limit_nps"] * model.dt + 1e-8)
        previous = actual
    # An imposed chassis-velocity change represents a severe transient. With a
    # deliberately low nominal power, actual force must decay through the actuator
    # dynamics rather than jump instantaneously to the new target-power bound.
    import pychrono as chrono

    transient_cfg = load_config(
        {**cfg, "actuator": {**actuator, "power_limit_w": 1.0, "delay_s": 0.0}}
    )
    transient = plant(transient_cfg)
    for _ in range(round(0.04 / transient.dt)):
        transient.step(np.full(4, 500.0))
    force_before = transient.actuator_force.copy()
    transient.chassis.SetPosDt(chrono.ChVector3d(0, 0, 1.0))
    transient.system.Update(transient.time)
    transient.step(np.full(4, 500.0))
    force_after = transient.actuator_force.copy()
    observed_power = np.abs(transient.metrics()["mechanical_power_w"])
    assert np.all(force_after < force_before), "Power derating did not reduce actual force"
    assert np.all(
        np.abs(force_after - force_before) <= actuator["rate_limit_nps"] * transient.dt + 1e-8
    )
    return {
        "configured_delay_s": actuator["delay_s"],
        "first_response_step_ends_at_s": first_response_end,
        "force_limit_n": limit,
        "peak_actual_force_n": peak_force,
        "slew_limit_nps": actuator["rate_limit_nps"],
        "peak_actual_slew_nps": peak_slew,
        "nominal_power_limit_w": actuator["power_limit_w"],
        "peak_actual_power_w": peak_power,
        "imposed_velocity_transient": {
            "nominal_power_w": 1.0,
            "imposed_chassis_speed_mps": 1.0,
            "force_before_n": force_before.tolist(),
            "force_after_n": force_after.tolist(),
            "actual_power_after_step_w": observed_power.tolist(),
        },
        "power_interpretation": "Power derates the target; actuator lag and slew can produce transient overshoot.",
    }


def check_cleanup(cfg: dict) -> dict:
    model = plant(cfg)
    functors = list(model.functors)
    model.step(np.zeros(4))
    model.close()
    assert all(functor.model is None for functor in functors)
    assert len(model.system.GetBodies()) == 0
    assert not model.functors and not model.suspensions and not model.tires and not model.wheels
    model.close()  # Required for safe repeated close/finally paths.
    return {
        "remaining_bodies": 0,
        "callback_model_references_cleared": True,
        "repeated_close": "passed",
    }


def check_unilateral_tires(cfg: dict) -> dict:
    model = plant(cfg, "bump")
    data = rollout(model, 2.0)
    minimum_load = float(np.min(data["tire_load"]))
    assert minimum_load >= 0
    # Lift the actual wheel body above its tire radius, then update geometry.
    import pychrono as chrono

    detached = plant(cfg, "flat")
    position = detached.wheels[0].GetPos()
    detached.wheels[0].SetPos(chrono.ChVector3d(position.x, position.y, detached.radius + 0.05))
    detached.system.Update(0.0)
    detached_load = float(detached.metrics()["tire_load_n"][0])
    assert detached_load == 0, detached_load
    return {
        "minimum_load_over_bump_n": minimum_load,
        "lifted_wheel_clearance_m": 0.05,
        "lifted_wheel_load_n": detached_load,
    }


def check_road_delay(cfg: dict) -> dict:
    v = cfg["vehicle"]
    delay = (v["front_distance_m"] + v["rear_distance_m"]) / cfg["road"]["speed_mps"]
    errors = {}
    for kind in ("random", "bump", "left_bump", "sine", "pothole"):
        road = plant(cfg, kind).road
        maximum = 0.0
        for t in np.linspace(0, 2, 101):
            front_z, front_v = road.sample(float(t))
            rear_z, rear_v = road.sample(float(t + delay))
            maximum = max(
                maximum,
                float(np.max(np.abs(front_z[:2] - rear_z[2:]))),
                float(np.max(np.abs(front_v[:2] - rear_v[2:]))),
            )
        assert maximum < 1e-10, (kind, maximum)
        errors[kind] = maximum
    return {"axle_delay_s": delay, "max_front_vs_delayed_rear_errors": errors, "tolerance": 1e-10}


def check_core_reproducibility(cfg: dict) -> dict:
    first = rollout(plant(cfg, "random", 734), 1.0)
    second = rollout(plant(cfg, "random", 734), 1.0)
    error = max(float(np.max(np.abs(first[key] - second[key]))) for key in first)
    assert error < 1e-12, error
    return {"seed": 734, "duration_s": 1, "maximum_repeat_error": error}


def check_static_tangent(cfg: dict) -> dict:
    """Compare analytic LQR mechanics with finite differences of Chrono geometry/forces."""
    import pychrono as chrono
    from chrono_suspension.controllers import linear_matrices

    def generalized_force(model: SevenDOFModel, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        model.chassis.SetPos(chrono.ChVector3d(0, 0, model.body_z0 + q[0]))
        model.chassis.SetRot(chrono.QuatFromAngleY(q[2]) * chrono.QuatFromAngleX(q[1]))
        model.chassis.SetPosDt(chrono.ChVector3d(0, 0, qd[0]))
        model.chassis.SetAngVelParent(
            chrono.ChVector3d(qd[1] * np.cos(q[2]), qd[2], -qd[1] * np.sin(q[2]))
        )
        for i, wheel in enumerate(model.wheels):
            wheel.SetPos(chrono.ChVector3d(*model.xy[i], model.wheel_z0[i] + q[i + 3]))
            wheel.SetPosDt(chrono.ChVector3d(0, 0, qd[i + 3]))
        model.system.Update(0.0)
        body = model.chassis.GetPos()
        body_position = np.array([body.x, body.y, body.z])
        forces, torque, tire_forces = [], np.zeros(3), []
        for i, spring in enumerate(model.suspensions):
            first, second = spring.GetPoint1Abs(), spring.GetPoint2Abs()
            top = np.array([first.x, first.y, first.z])
            bottom = np.array([second.x, second.y, second.z])
            scalar = model.functors[2 * i].evaluate(
                0, spring.GetRestLength(), spring.GetLength(), spring.GetVelocity(), spring
            )
            force = scalar * (top - bottom) / spring.GetLength()
            forces.append(force)
            torque += np.cross(top - body_position, force)
            tire = model.tires[i]
            tire_forces.append(
                model.functors[2 * i + 1].evaluate(
                    0, tire.GetRestLength(), tire.GetLength(), tire.GetVelocity(), tire
                )
            )
        forces = np.asarray(forces)
        return np.r_[
            np.sum(forces[:, 2]) - model.config["vehicle"]["sprung_mass_kg"] * 9.81,
            torque[0] * np.cos(q[2]) - torque[2] * np.sin(q[2]),
            torque[1],
            -forces[:, 2] + np.asarray(tire_forces) - model.mu * 9.81,
        ]

    cases = {
        "configured": cfg,
        "asymmetric_geometry": load_config(
            {
                **cfg,
                "vehicle": {
                    **cfg["vehicle"],
                    "front_distance_m": 1.2,
                    "rear_distance_m": 1.55,
                    "unsprung_mass_kg": [42, 43, 47, 48],
                    "compression_damping_ratio": 1.7,
                    "rebound_damping_ratio": 1.7,
                },
            }
        ),
    }
    results = {}
    epsilon = 1e-6
    for name, case_cfg in cases.items():
        model = plant(case_cfg)
        a, b, _ = linear_matrices(case_cfg)
        vehicle = case_cfg["vehicle"]
        mass = np.r_[vehicle["sprung_mass_kg"], vehicle["inertia_kgm2"][:2], model.mu]
        analytic_k = -mass[:, None] * a[7:, :7]
        analytic_c = -mass[:, None] * a[7:, 7:]
        finite_k, finite_c = np.zeros((7, 7)), np.zeros((7, 7))
        zero = np.zeros(7)
        for j in range(7):
            perturbation = np.eye(7)[j] * epsilon
            finite_k[:, j] = -(
                generalized_force(model, perturbation, zero)
                - generalized_force(model, -perturbation, zero)
            ) / (2 * epsilon)
            finite_c[:, j] = -(
                generalized_force(model, zero, perturbation)
                - generalized_force(model, zero, -perturbation)
            ) / (2 * epsilon)
        # B excludes actuator dynamics: perturb actual force, not delayed demand.
        finite_b = np.zeros((7, 4))
        force_epsilon = 1e-3
        for j in range(4):
            model.actuator_force = np.eye(4)[j] * force_epsilon
            positive = generalized_force(model, zero, zero)
            model.actuator_force = -np.eye(4)[j] * force_epsilon
            negative = generalized_force(model, zero, zero)
            finite_b[:, j] = (positive - negative) / (2 * force_epsilon)
        model.actuator_force[:] = 0
        generalized_force(model, zero, zero)
        assert np.allclose(finite_k, analytic_k, rtol=2e-8, atol=2e-5)
        assert np.allclose(finite_c, analytic_c, rtol=2e-8, atol=2e-5)
        assert np.allclose(finite_b, mass[:, None] * b[7:], rtol=1e-6, atol=1e-7)
        results[name] = {
            "max_absolute_stiffness_error": float(np.max(np.abs(finite_k - analytic_k))),
            "max_absolute_damping_error": float(np.max(np.abs(finite_c - analytic_c))),
            "max_absolute_force_jacobian_error": float(
                np.max(np.abs(finite_b - mass[:, None] * b[7:]))
            ),
        }
    return {
        "coordinate_perturbation": epsilon,
        "force_perturbation_n": 1e-3,
        "stiffness_damping_rtol": 2e-8,
        "stiffness_damping_atol": 2e-5,
        "cases": results,
        "interpretation": "Static mechanical tangent only; actuator lag/delay remain nonlinear environment states.",
    }


def check_timestep(cfg: dict) -> dict:
    coarse_cfg = load_config({**cfg, "physics": {**cfg["physics"], "dt": 0.001}})
    fine_cfg = load_config({**cfg, "physics": {**cfg["physics"], "dt": 0.0005}})
    coarse = rollout(plant(coarse_cfg, "bump", 829), 3.0)
    fine = rollout(plant(fine_cfg, "bump", 829), 3.0)
    report = {
        "physics_dt_s": [0.001, 0.0005],
        "sample_dt_s": 0.01,
        "duration_s": 3,
        "scenario": "bump",
        "controller": "passive",
        "metrics": {},
    }
    for name in ("acceleration", "travel"):
        coarse_rms = float(np.sqrt(np.mean(coarse[name] ** 2)))
        fine_rms = float(np.sqrt(np.mean(fine[name] ** 2)))
        relative_rms = abs(coarse_rms - fine_rms) / max(fine_rms, 1e-12)
        waveform_error = float(np.sqrt(np.mean((coarse[name] - fine[name]) ** 2)))
        report["metrics"][name] = {
            "rms_at_1ms": coarse_rms,
            "rms_at_0_5ms": fine_rms,
            "relative_rms_difference": relative_rms,
            "waveform_rmse": waveform_error,
            "relative_waveform_rmse": waveform_error / max(fine_rms, 1e-12),
        }
    # This is an audit, not a convergence-order proof: expose the measured gap.
    report["within_10_percent_rms"] = all(
        m["relative_rms_difference"] < 0.10 for m in report["metrics"].values()
    )
    report["interpretation"] = (
        "Two-step-size sensitivity only; does not establish asymptotic convergence, "
        "real-vehicle fidelity, or control performance."
    )
    return report


def check_frame(cfg: dict) -> dict:
    model = plant(cfg, "left_bump")
    rollout(model, 1.0)
    frame = model.export_frame()
    validate_frame(frame)
    restored = json.loads(json.dumps(frame, allow_nan=False))
    validate_frame(restored)
    return {"fields": list(frame), "json_round_trip": True, "quaternion_order": "wxyz"}


def check_environment(cfg: dict) -> dict:
    from chrono_suspension.env import ChronoSuspensionEnv
    from gymnasium.utils.env_checker import check_env as gym_check_env
    from stable_baselines3.common.env_checker import check_env as sb3_check_env

    modes = {}
    for mode in ("privileged", "sensors"):
        env = ChronoSuspensionEnv(cfg, observation_mode=mode)
        try:
            gym_check_env(env, skip_render_check=True)
            sb3_check_env(env, warn=True, skip_render_check=True)
            histories = []
            for _ in range(2):
                obs, _ = env.reset(seed=173)
                history = [obs.copy()]
                for _step in range(20):
                    obs, reward, terminated, truncated, _ = env.step(np.zeros(4, dtype=np.float32))
                    assert np.all(np.isfinite(obs)) and math.isfinite(reward)
                    assert not terminated
                    history.append(obs.copy())
                    if truncated:
                        break
                histories.append(np.asarray(history))
            error = float(np.max(np.abs(histories[0] - histories[1])))
            assert error == 0
            modes[mode] = {
                "observation_shape": list(env.observation_space.shape),
                "action_shape": list(env.action_space.shape),
                "repeat_error": error,
                "gymnasium_check_env": "passed",
                "sb3_check_env": "passed",
            }
        finally:
            env.close()
    return modes


def check_suspension_mount_guard(cfg: dict) -> dict:
    """Reject insufficient mounting clearance and detect unsigned-TSDA inversion."""
    import pychrono as chrono
    from chrono_suspension.env import ChronoSuspensionEnv

    low_vehicle = {**cfg["vehicle"], "body_height_m": 0.5, "travel_limit_m": 0.09}
    try:
        load_config({**cfg, "vehicle": low_vehicle})
    except ValueError as exc:
        rejection = str(exc)
        assert "Static suspension length must exceed" in rejection, rejection
    else:
        raise AssertionError("Insufficient suspension mounting clearance was accepted")

    allowed = load_config(
        {
            **cfg,
            "vehicle": {**low_vehicle, "travel_limit_m": 0.04, "bump_stop_start_m": 0.025},
        }
    )
    env = ChronoSuspensionEnv(allowed)
    try:
        env.reset(seed=713, options={"scenario": {"kind": "flat"}})
        initial_gaps = np.array(
            [s.GetPoint1Abs().z - s.GetPoint2Abs().z for s in env.model.suspensions]
        )
        assert np.all(initial_gaps > allowed["vehicle"]["travel_limit_m"])
        position = env.model.chassis.GetPos()
        env.model.chassis.SetPos(chrono.ChVector3d(position.x, position.y, position.z - 0.09))
        env.model.system.Update(env.model.time)
        crossed_gaps = np.array(
            [s.GetPoint1Abs().z - s.GetPoint2Abs().z for s in env.model.suspensions]
        )
        assert np.all(crossed_gaps < 0), crossed_gaps
        before = env.model.metrics()
        envelope = 2 * allowed["vehicle"]["travel_limit_m"]
        assert np.max(np.abs(before["suspension_travel_m"])) < envelope
        assert max(abs(before["roll_rad"]), abs(before["pitch_rad"])) < 0.5
        obs, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
        assert terminated and not truncated, info
        assert info["failure_reason"] == "suspension_mount_crossed_wheel", info
        assert np.max(info["max_travel_m"]) < envelope, info["max_travel_m"]
        assert env.model.time <= env.dt + 1e-12, "Inversion was not caught on the first substep"
        assert np.all(np.isfinite(obs)) and math.isfinite(reward)
        return {
            "rejected_config": {"body_height_m": 0.5, "travel_limit_m": 0.09},
            "config_rejection": rejection,
            "accepted_config": {
                "body_height_m": 0.5,
                "travel_limit_m": 0.04,
                "bump_stop_start_m": 0.025,
            },
            "imposed_chassis_displacement_m": -0.09,
            "initial_signed_mount_gaps_m": initial_gaps.tolist(),
            "crossed_signed_mount_gaps_m": crossed_gaps.tolist(),
            "unsigned_travel_after_step_m": info["max_travel_m"].tolist(),
            "other_travel_failure_threshold_m": envelope,
            "failure_reason": info["failure_reason"],
            "failure_time_s": env.model.time,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--core-only", action="store_true", help="Skip Gymnasium/SB3 API checks")
    parser.add_argument(
        "--output", help="New results/ JSON path; existing reports are never overwritten"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = result_path(args.output or f"results/chrono/validation/check_model_{timestamp}.json")
    if output.exists():
        raise FileExistsError(output)
    cases: list[tuple[str, Callable]] = [
        ("mechanical_dof", check_dof),
        ("static_equilibrium", check_static),
        ("left_right_symmetry", check_symmetry),
        ("single_side_bump", check_single_side),
        ("positive_force_direction", check_force_direction),
        ("actuator_delay_and_limits", check_actuator_limits),
        ("cleanup", check_cleanup),
        ("unilateral_tires", check_unilateral_tires),
        ("wheelbase_delay", check_road_delay),
        ("core_reproducibility", check_core_reproducibility),
        ("static_linearization_tangent", check_static_tangent),
        ("timestep_sensitivity", check_timestep),
        ("render_frame", check_frame),
    ]
    if not args.core_only:
        cases.append(("gymnasium_and_sb3", check_environment))
        cases.append(("suspension_mount_guard", check_suspension_mount_guard))
    report: dict[str, Any] = {
        "created_utc": timestamp,
        "config": cfg,
        "checks": {},
        "core_only": args.core_only,
        "versions": {},
    }
    for package in ("numpy", "gymnasium", "stable-baselines3"):
        try:
            report["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["versions"][package] = "not installed"
    chrono_metadata = list((Path(sys.prefix) / "conda-meta").glob("pychrono-*.json"))
    report["versions"]["chrono"] = (
        json.loads(chrono_metadata[0].read_text(encoding="utf-8"))["version"]
        if chrono_metadata
        else "unknown (no conda metadata)"
    )
    for name, case in cases:
        started = time.perf_counter()
        try:
            detail = case(cfg)
            report["checks"][name] = {"status": "passed", "detail": detail}
        except Exception as exc:
            report["checks"][name] = {"status": "failed", "error": repr(exc)}
        finally:
            for model in _OPEN_MODELS:
                model.close()
            _OPEN_MODELS.clear()
        report["checks"][name]["wall_seconds"] = time.perf_counter() - started
        print(f"{name}: {report['checks'][name]['status']}", flush=True)
    report["all_requested_checks_passed"] = all(
        case["status"] == "passed" for case in report["checks"].values()
    )
    if args.core_only:
        report["skipped"] = ["gymnasium_and_sb3", "suspension_mount_guard"]
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    print(f"Report: {output}")
    if not report["all_requested_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
