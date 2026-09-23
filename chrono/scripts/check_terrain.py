"""Audit Chrono terrain, road reconstruction and strict 7-DOF headless operation."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chrono_suspension.config import load_config  # noqa: E402
from chrono_suspension.roads import FourWheelRoad, seven_road_scenarios  # noqa: E402
from chrono_suspension.training import RESULTS_ROOT, limit_worker_threads  # noqa: E402


def _assert_finite(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _assert_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_finite(item)
    elif isinstance(value, np.ndarray):
        assert np.isfinite(value).all(), "Nonfinite array"
    elif isinstance(value, (int, float, np.number)):
        assert math.isfinite(float(value)), f"Nonfinite scalar: {value}"


def _irrlicht_modules() -> list[str]:
    return [name for name in sys.modules if name.startswith("pychrono") and "irrlicht" in name]


def _block_visual_construction() -> tuple[ExitStack, list[str]]:
    import pychrono as ch
    import pychrono.irrlicht as irr
    import pychrono.vehicle as veh

    stack, blocked = ExitStack(), []
    try:
        for module, names in (
            (irr, ("ChVisualSystemIrrlicht",)),
            (veh, ("ChVehicleVisualSystemIrrlicht", "ChSuspensionTestRigVisualSystemIRR")),
            (ch, ("ChTriangleMeshConnected", "ChVisualShapeTriangleMesh")),
        ):
            for name in names:
                if not hasattr(module, name):
                    continue
                label = f"{module.__name__}.{name}"
                stack.enter_context(
                    patch.object(
                        getattr(module, name),
                        "__init__",
                        side_effect=AssertionError(
                            f"Visual construction in headless audit: {label}"
                        ),
                    )
                )
                blocked.append(label)
    except BaseException:
        stack.close()
        raise
    return stack, blocked


def _project_visualization_loaded() -> bool:
    return "chrono_suspension.visualization" in sys.modules


def _roads(config: dict) -> dict[str, FourWheelRoad]:
    vehicle = config["vehicle"]
    scenarios = seven_road_scenarios()
    scenarios["random"] = {
        "kind": "random",
        "roughness_m": 0.012,
        "left_right_correlation": 0.35,
        "run_in_m": 4.5,
        "tracks": "right",
    }
    # Explicit component seeds must survive reconstruction, not be regenerated.
    scenarios["composite"]["components"][0]["seed"] = 91007
    return {
        name: FourWheelRoad(
            {**scenario, "seed": 91001 + index, "speed_mps": 15.0},
            vehicle["front_distance_m"],
            vehicle["rear_distance_m"],
            123,
        )
        for index, (name, scenario) in enumerate(scenarios.items())
    }


def check_road_reconstruction(config: dict) -> dict:
    vehicle = config["vehicle"]
    front, rear = vehicle["front_distance_m"], vehicle["rear_distance_m"]
    results = {}
    times = np.linspace(-0.1, 8.0, 129)
    for name, road in _roads(config).items():
        record = json.loads(json.dumps(road.metadata(), allow_nan=False))
        restored = FourWheelRoad.from_metadata(record, front, rear)
        assert restored.metadata() == record, f"Metadata/child seeds changed: {name}"
        error = max(
            float(np.max(np.abs(np.asarray(road.sample(t)) - np.asarray(restored.sample(t)))))
            for t in times
        )
        assert error < 1e-12, (name, error)
        try:
            FourWheelRoad.from_metadata(record, front, rear + 0.1)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Wheelbase mismatch was accepted: {name}")
        results[name] = {
            "max_height_velocity_error": error,
            "seed": record["seed"],
            "component_seeds": [part["seed"] for part in record["components"]],
            "wheelbase_mismatch_rejected": True,
        }
    return {"times_per_road": len(times), "tolerance": 1e-12, "roads": results}


def check_spatial_tracks(config: dict) -> dict:
    stations = np.array([[-0.7, 0.41, 1.61, 3.91], [9.11, 9.41, 30.31, 75.41]])
    epsilon = 1e-5
    results = {}
    for name, road in _roads(config).items():
        height, slope = road.sample_tracks(stations)
        assert height.shape == slope.shape == (*stations.shape, 2)
        scalar = [road.sample_tracks(float(station)) for station in stations.flat]
        expected_height = np.array([value[0] for value in scalar]).reshape(height.shape)
        expected_slope = np.array([value[1] for value in scalar]).reshape(slope.shape)
        scalar_error = max(
            float(np.max(np.abs(height - expected_height))),
            float(np.max(np.abs(slope - expected_slope))),
        )
        assert scalar_error < 1e-12
        finite_difference = (
            road.sample_tracks(stations + epsilon)[0] - road.sample_tracks(stations - epsilon)[0]
        ) / (2 * epsilon)
        derivative_error = float(np.max(np.abs(finite_difference - slope)))
        assert derivative_error < 2e-7, (name, derivative_error)
        four_wheel_error = 0.0
        for station in stations.flat:
            track_height, track_slope = road.sample_tracks(
                np.array([station, station - road.wheelbase])
            )
            expected = (track_height.reshape(4), road.speed * track_slope.reshape(4))
            actual = road.sample(float(station) / road.speed)
            four_wheel_error = max(
                four_wheel_error, float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
            )
        assert four_wheel_error < 1e-10, (name, four_wheel_error)
        assert road.sample_tracks(np.empty((0,)))[0].shape == (0, 2)
        results[name] = {
            "scalar_array_error": scalar_error,
            "slope_finite_difference_error": derivative_error,
            "four_wheel_height_velocity_error": four_wheel_error,
        }
    return {"station_shape": list(stations.shape), "difference_step_m": epsilon, "roads": results}


def _vector(value: Any) -> np.ndarray:
    return np.array([value.x, value.y, value.z], dtype=float)


def check_native_terrain(config: dict) -> dict:
    import pychrono as ch
    import pychrono.vehicle as veh
    from chrono_suspension.terrain import RoadTerrain

    # Deliberately unequal half tracks exercise the constant-height outer strips.
    config = deepcopy(config)
    config["vehicle"].update(front_half_track_m=0.65, rear_half_track_m=0.95)
    front = config["vehicle"]["front_distance_m"]
    epsilon = 1e-5
    results = {}
    for name, road in _roads(config).items():
        terrain = RoadTerrain(road, config)
        assert isinstance(terrain, veh.ChTerrain) and terrain.this is not None
        maximum_normal_error = maximum_native_error = maximum_plateau_error = 0.0
        for station in (0.41, 2.11, 5.31, 9.13, 9.43, 30.31, 75.29):
            track_height, track_slope = road.sample_tracks(station)
            for y in (-1.15, -0.95, -0.65, -0.21, 0.0, 0.33, 0.65, 0.95, 1.15):
                x = front + station
                location = ch.ChVector3d(x, y, 11.0)
                height = terrain.GetHeight(location)
                normal = _vector(terrain.GetNormal(location))
                point = _vector(terrain.GetPoint(location))
                dx = (
                    terrain.GetHeight(ch.ChVector3d(x + epsilon, y, 0))
                    - terrain.GetHeight(ch.ChVector3d(x - epsilon, y, 0))
                ) / (2 * epsilon)
                dy = (
                    terrain.GetHeight(ch.ChVector3d(x, y + epsilon, 0))
                    - terrain.GetHeight(ch.ChVector3d(x, y - epsilon, 0))
                ) / (2 * epsilon)
                expected_normal = np.array([-dx, -dy, 1.0])
                expected_normal /= np.linalg.norm(expected_normal)
                maximum_normal_error = max(
                    maximum_normal_error, float(np.max(np.abs(normal - expected_normal)))
                )
                np.testing.assert_allclose(point, [x, y, height], rtol=0, atol=1e-12)
                assert abs(np.linalg.norm(normal) - 1.0) < 1e-12 and normal[2] > 0
                assert np.isfinite(terrain.GetCoefficientFriction(location))
                # GetProperties is C++ code calling the four virtual methods;
                # this tests SWIG director dispatch, not just Python inheritance.
                native_height, friction = ch.double_ptr(), ch.float_ptr()
                native_point, native_normal = ch.ChVector3d(), ch.ChVector3d()
                terrain.GetProperties(
                    location, native_point, native_height, native_normal, friction
                )
                native_error = max(
                    abs(native_height.value() - height),
                    float(np.max(np.abs(_vector(native_point) - point))),
                    float(np.max(np.abs(_vector(native_normal) - normal))),
                    abs(friction.value() - terrain.GetCoefficientFriction(location)),
                )
                maximum_native_error = max(maximum_native_error, native_error)
                if abs(y) >= 0.65:
                    side = 0 if y > 0 else 1
                    maximum_plateau_error = max(
                        maximum_plateau_error,
                        abs(height - track_height[side]),
                        abs(normal[1]),
                        abs(-normal[0] / normal[2] - track_slope[side]),
                    )
        assert maximum_normal_error < 2e-6, (name, maximum_normal_error)
        assert maximum_native_error < 1e-7, (name, maximum_native_error)
        assert maximum_plateau_error < 1e-10, (name, maximum_plateau_error)
        maximum_wheel_error = 0.0
        for t in np.linspace(0.0, 8.0, 81):
            actual = np.asarray(terrain.sample_wheels(float(t)))
            reference = np.asarray(road.sample(float(t)))
            maximum_wheel_error = max(
                maximum_wheel_error, float(np.max(np.abs(actual - reference)))
            )
        assert maximum_wheel_error < 1e-12, (name, maximum_wheel_error)
        terrain.clear_cache()
        results[name] = {
            "max_normal_finite_difference_error": maximum_normal_error,
            "max_native_virtual_dispatch_error": maximum_native_error,
            "max_unequal_track_plateau_error": maximum_plateau_error,
            "max_wheel_height_velocity_error": maximum_wheel_error,
        }

    class QueryProbe(RoadTerrain):
        def __init__(self, road: FourWheelRoad, config: dict):
            super().__init__(road, config)
            self.height_queries: list[tuple[float, float]] = []

        def GetHeight(self, loc):
            self.height_queries.append((float(loc.x), float(loc.y)))
            return super().GetHeight(loc)

    probe = QueryProbe(_roads(config)["iso_b"], config)
    probe.sample_wheels(0.73)
    assert len(probe.height_queries) == 4, probe.height_queries
    rear = config["vehicle"]["rear_distance_m"]
    expected_x = np.array([front, front, -rear, -rear]) + probe.road.speed * 0.73
    np.testing.assert_allclose(np.asarray(probe.height_queries)[:, 0], expected_x, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(probe.height_queries)[:, 1], [0.65, -0.65, 0.95, -0.95], atol=1e-12
    )
    return {
        "native_type": "chrono::vehicle::ChTerrain",
        "native_dispatch": "ChTerrain::GetProperties via SWIG directors",
        "front_rear_half_tracks_m": [0.65, 0.95],
        "difference_step_m": epsilon,
        "height_queries_per_wheel_sample": len(probe.height_queries),
        "roads": results,
    }


def check_mechanics(config: dict) -> dict:
    import pychrono.vehicle as veh
    from chrono_suspension.physics import SevenDOFModel

    vehicle = config["vehicle"]
    roads = _roads(config)
    roads["flat"] = FourWheelRoad(
        {"kind": "flat"}, vehicle["front_distance_m"], vehicle["rear_distance_m"], 91000
    )
    results = {}
    for name, duration in (("flat", 1.0), ("bump", 1.2), ("iso_c", 1.0), ("composite", 1.0)):
        model = SevenDOFModel(config, roads[name])
        try:
            assert isinstance(model.terrain, veh.ChTerrain)
            assert model.system.GetNumCoordsVelLevel() == 30
            assert model.system.GetNumConstraints() == 23
            assert model.mechanical_dof == 7
            assert len(model.system.GetBodies()) == 6, "Terrain added a dynamic body"
            peak_q = peak_qd = 0.0
            for index in range(round(duration / model.dt)):
                model.step(np.zeros(4))
                if index % 10 == 0:
                    q, qd = model.coordinates()
                    assert q.shape == qd.shape == (7,)
                    _assert_finite((q, qd, model.metrics(), model.export_frame()))
                    peak_q = max(peak_q, float(np.max(np.abs(q))))
                    peak_qd = max(peak_qd, float(np.max(np.abs(qd))))
            assert model.system.GetNumCoordsVelLevel() - model.system.GetNumConstraints() == 7
            if name == "flat":
                assert peak_q < 1e-5 and peak_qd < 1e-4, (peak_q, peak_qd)
            else:
                assert peak_q > 1e-8, f"Moving road produced no response: {name}"
            results[name] = {
                "duration_s": duration,
                "velocity_coordinates": 30,
                "constraints": 23,
                "mechanical_dof": 7,
                "peak_generalized_displacement": peak_q,
                "peak_generalized_velocity": peak_qd,
            }
        finally:
            model.close()
        assert len(model.system.GetBodies()) == 0, "Model bodies remained after close"
    assert not _project_visualization_loaded(), "Project visualization imported in headless model"
    return {
        "project_visualization_loaded": False,
        "vendor_irrlicht_modules": _irrlicht_modules(),
        "models_closed": True,
        "scenarios": results,
    }


def _vector_worker(config: dict, mode: str, rank: int):
    limit_worker_threads()
    guard, blocked = _block_visual_construction()
    import pychrono.vehicle as veh
    from chrono_suspension.env import ChronoSuspensionEnv

    class AuditedEnv(ChronoSuspensionEnv):
        def terrain_audit(self) -> dict:
            return {
                "pid": os.getpid(),
                "rank": rank,
                "start_method": mp.get_start_method(),
                "native_terrain": isinstance(self.model.terrain, veh.ChTerrain),
                "mechanical_dof": self.model.mechanical_dof,
                "body_count_including_fixed_ground": len(self.model.system.GetBodies()),
                "viewer_is_none": self.viewer is None,
                "project_visualization_loaded": _project_visualization_loaded(),
                "blocked_visual_constructors": blocked,
                "irrlicht_modules": _irrlicht_modules(),
            }

        def close(self) -> None:
            try:
                super().close()
            finally:
                guard.close()

    worker_config = deepcopy(config)
    worker_config["physics"]["episode_seconds"] = 20 * config["physics"]["control_dt"]
    worker_config["road"] = seven_road_scenarios()["iso_b" if rank == 0 else "composite"]
    try:
        return AuditedEnv(worker_config, observation_mode=mode, render_mode=None)
    except BaseException:
        guard.close()
        raise


def check_vector(config: dict) -> dict:
    limit_worker_threads()
    from stable_baselines3.common.vec_env import SubprocVecEnv

    results = {}
    for mode in ("privileged", "sensors"):
        vector = SubprocVecEnv(
            [partial(_vector_worker, config, mode, rank) for rank in range(2)], start_method="spawn"
        )
        processes = list(vector.processes)
        try:
            vector.seed(9281)
            observation = vector.reset()
            _assert_finite((observation, vector.reset_infos))
            expected_size = (
                18 + 4 * round(config["actuator"]["delay_s"] / config["physics"]["dt"])
                if mode == "privileged"
                else 13 * config["observation"]["history_steps"]
            )
            assert observation.shape == (2, expected_size)
            finished = np.zeros(2, dtype=int)
            for _ in range(30):
                observation, reward, done, infos = vector.step(np.zeros((2, 4), dtype=np.float32))
                _assert_finite((observation, reward, infos))
                assert observation.shape == (2, expected_size)
                finished += done.astype(int)
            assert np.all(finished >= 1), "Vector automatic reset was not exercised"
            audit = vector.env_method("terrain_audit")
            assert len({item["pid"] for item in audit}) == 2
            for item in audit:
                assert item["start_method"] == "spawn" and item["native_terrain"]
                assert item["mechanical_dof"] == 7
                assert item["body_count_including_fixed_ground"] == 6
                assert item["viewer_is_none"] and not item["project_visualization_loaded"]
            _assert_finite(vector.reset_infos)
            results[mode] = {
                "observation_shape": list(observation.shape),
                "completed_episodes_per_worker": finished.tolist(),
                "workers": audit,
            }
        finally:
            try:
                vector.close()
            finally:
                for process in processes:
                    process.join(timeout=3)
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=3)
        assert all(not process.is_alive() for process in processes), "Worker survived close"
        results[mode]["worker_processes_closed"] = True
    assert not _project_visualization_loaded(), "Project visualization imported in headless vector"
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--with-vector", action="store_true", help="Also check two spawn workers")
    args = parser.parse_args()
    limit_worker_threads()
    config = load_config(args.config)
    if config["observation"]["preview_steps"]:
        parser.error("This two-observation audit requires preview_steps=0.")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = RESULTS_ROOT / "terrain" / f"checks_{timestamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    cases = [
        ("road_metadata_roundtrip", check_road_reconstruction),
        ("spatial_track_sampling", check_spatial_tracks),
        ("native_terrain_queries", check_native_terrain),
        ("strict_seven_dof", check_mechanics),
    ]
    if args.with_vector:
        cases.append(("two_worker_spawn", check_vector))
    report: dict[str, Any] = {
        "created_utc": timestamp,
        "config": config,
        "with_vector": args.with_vector,
        "checks": {},
        "interpretation": "Bounded geometry/API/numerical checks; not road certification or vehicle validation.",
    }
    guard, blocked = _block_visual_construction()
    with guard:
        for name, check in cases:
            started = time.monotonic()
            try:
                detail = check(config)
                report["checks"][name] = {"status": "passed", "detail": detail}
            except Exception as exc:
                report["checks"][name] = {
                    "status": "failed",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            report["checks"][name]["wall_seconds"] = time.monotonic() - started
            print(f"{name}: {report['checks'][name]['status']}", flush=True)
    report["irrlicht_modules"] = _irrlicht_modules()
    report["headless_audit"] = {
        "blocked_visual_constructors": blocked,
        "project_visualization_loaded": _project_visualization_loaded(),
        "vendor_import_note": "Official PyChrono __init__/vehicle import Irrlicht bindings; module presence does not imply a renderer or window exists.",
    }
    report["all_requested_checks_passed"] = not _project_visualization_loaded() and all(
        row["status"] == "passed" for row in report["checks"].values()
    )
    report["skipped"] = [] if args.with_vector else ["two_worker_spawn"]
    # Exclusive creation also protects against a timestamp collision.
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Report: {output}")
    if not report["all_requested_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
