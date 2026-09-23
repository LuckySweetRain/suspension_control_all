"""Audit road mathematics, statistics and optional passive Chrono episodes; no pytest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parent
sys.path.insert(0, str(PROJECT))

from chrono_suspension.config import load_config  # noqa: E402
from chrono_suspension.roads import FourWheelRoad, seven_road_scenarios  # noqa: E402
from chrono_suspension.visualization import result_path  # noqa: E402

# The repository HEAD before the road expansion. Keeping the immutable ref makes
# this comparison useful after the new implementation itself has been committed.
LEGACY_REF = "1bd2e4b329509bf11fbd9a4b310d3f6b64bde1e8"
SEVEN_KINDS = {"iso_b", "iso_c", "sine", "step", "pothole", "bump", "composite"}


def road(cfg: dict, scenario: dict, seed: int = 431) -> FourWheelRoad:
    resolved = {**deepcopy(cfg["road"]), **deepcopy(scenario)}
    resolved.pop("scenarios", None)
    vehicle = cfg["vehicle"]
    return FourWheelRoad(resolved, vehicle["front_distance_m"], vehicle["rear_distance_m"], seed)


def samples(profile: FourWheelRoad, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    heights, velocities = zip(*(profile.sample(float(t)) for t in times), strict=True)
    return np.asarray(heights), np.asarray(velocities)


def check_derivatives(cfg: dict) -> dict:
    presets = seven_road_scenarios()
    assert set(presets) == SEVEN_KINDS
    # Random evaluation points avoid the second-derivative discontinuities at
    # bump boundaries; the dedicated step check covers its endpoints explicitly.
    times = np.random.default_rng(581).uniform(-0.2, 10.2, 321)
    epsilon = 1e-6
    report = {}
    for name, preset in presets.items():
        profile = road(cfg, preset)
        heights, velocity = samples(profile, times)
        plus, _ = samples(profile, times + epsilon)
        minus, _ = samples(profile, times - epsilon)
        numerical = (plus - minus) / (2 * epsilon)
        assert heights.shape == velocity.shape == (len(times), 4)
        assert np.isfinite(heights).all() and np.isfinite(velocity).all()
        np.testing.assert_allclose(velocity, numerical, rtol=2e-5, atol=5e-6)
        assert profile.preview(0, 0, 0.01).shape == (0,)
        preview = profile.preview(0.217, 11, 0.013)
        expected = samples(profile, 0.217 + np.arange(1, 12) * 0.013)[0].reshape(-1)
        np.testing.assert_allclose(preview, expected, rtol=0, atol=1e-12)
        report[name] = {
            "samples": len(times),
            "maximum_derivative_error_mps": float(np.max(np.abs(velocity - numerical))),
            "peak_height_m": float(np.max(np.abs(heights))),
            "peak_velocity_mps": float(np.max(np.abs(velocity))),
        }
    return {"central_difference_epsilon_s": epsilon, "roads": report}


def check_delay_and_tracks(cfg: dict) -> dict:
    times = np.linspace(-0.2, 10.0, 151)
    report = {}
    for name, preset in seven_road_scenarios().items():
        profile = road(cfg, preset)
        delay = profile.wheelbase / profile.speed
        heights, velocities = samples(profile, times)
        delayed_heights, delayed_velocities = samples(profile, times + delay)
        errors = [
            np.max(np.abs(heights[:, :2] - delayed_heights[:, 2:])),
            np.max(np.abs(velocities[:, :2] - delayed_velocities[:, 2:])),
        ]
        assert max(errors) < 1e-9, (name, errors)
        symmetric = road(cfg, {**preset, "left_right_correlation": 1.0, "tracks": "both"})
        base_z, base_v = samples(symmetric, times)
        np.testing.assert_allclose(base_z[:, [0, 2]], base_z[:, [1, 3]], atol=1e-14)
        np.testing.assert_allclose(base_v[:, [0, 2]], base_v[:, [1, 3]], atol=1e-14)
        for track, enabled, disabled in (("left", [0, 2], [1, 3]), ("right", [1, 3], [0, 2])):
            masked = road(cfg, {**preset, "left_right_correlation": 1.0, "tracks": track})
            z, velocity = samples(masked, times)
            assert np.count_nonzero(z[:, disabled]) == 0
            assert np.count_nonzero(velocity[:, disabled]) == 0
            np.testing.assert_allclose(z[:, enabled], base_z[:, enabled], atol=1e-14)
            np.testing.assert_allclose(velocity[:, enabled], base_v[:, enabled], atol=1e-14)
        report[name] = {"axle_delay_s": delay, "height_and_velocity_max_errors": errors}
    return report


def check_step(cfg: dict) -> dict:
    report = {}
    for height in (0.03, -0.03):
        profile = road(
            cfg,
            {"kind": "step", "height_m": height, "start_m": 9.0, "transition_length_m": 0.3},
        )
        start = 9.0 / profile.speed
        duration = 0.3 / profile.speed
        times = np.array([start - duration, start, start + duration / 2, start + duration, 10])
        z, velocity = samples(profile, times)
        np.testing.assert_allclose(z[:, 0], [0, 0, height / 2, height, height], atol=1e-12)
        np.testing.assert_allclose(velocity[[0, 1, 3, 4], 0], 0, atol=1e-12)
        dense_times = np.linspace(start, start + duration, 1001)
        dense_z, dense_v = samples(profile, dense_times)
        assert np.all(np.sign(height) * np.diff(dense_z[:, 0]) >= -1e-12)
        integral = float(np.trapezoid(dense_v[:, 0], dense_times))
        np.testing.assert_allclose(integral, height, rtol=1e-8, atol=1e-12)
        report[str(height)] = {
            "transition_duration_s": duration,
            "integrated_velocity_m": integral,
            "final_height_m": float(z[-1, 0]),
            "endpoint_velocities_mps": velocity[[1, 3], 0].tolist(),
        }
    return report


def check_composite(cfg: dict) -> dict:
    profile = road(
        cfg,
        {
            "kind": "composite",
            "run_in_m": 3.0,
            "components": [
                {"kind": "iso_b"},
                {"kind": "sine", "height_m": 0.02, "phase_rad": 0.6},
                {"kind": "bump", "height_m": 0.012, "start_m": 0.25, "length_m": 2.0},
                {"kind": "pothole", "height_m": 0.008, "start_m": 1.0, "tracks": "right"},
            ],
        },
    )
    # Reconstruct independent children through the public API using resolved
    # metadata seeds, rather than calling the implementation's private sampler.
    children = [road(cfg, {**child.scenario, "seed": child.seed}) for child in profile.components]
    times = np.r_[np.linspace(0, 0.5, 151), np.linspace(0.6, 10, 51)]
    z, velocity = samples(profile, times)
    parts = [samples(child, times) for child in children]
    sum_z = sum(part[0] for part in parts)
    sum_velocity = sum(part[1] for part in parts)
    np.testing.assert_allclose(z, sum_z, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(velocity, sum_velocity, rtol=1e-12, atol=1e-12)
    midpoint = 1.5 / profile.speed
    raw_children = [
        road(cfg, {**child.scenario, "seed": child.seed, "run_in_m": 0.0})
        for child in profile.components
    ]
    raw_z = sum(child.sample(midpoint)[0] for child in raw_children)
    raw_v = sum(child.sample(midpoint)[1] for child in raw_children)
    midpoint_z, midpoint_v = profile.sample(midpoint)
    # Independent smoothstep midpoint identities: E=1/2 and dE/dx=1.5/L.
    assert abs(raw_z[0]) > 1e-5, "Run-in check requires a nonzero raw signal"
    np.testing.assert_allclose(midpoint_z[:2], raw_z[:2] / 2, atol=1e-12)
    np.testing.assert_allclose(
        midpoint_v[:2], raw_v[:2] / 2 + raw_z[:2] * profile.speed * 1.5 / 3.0, atol=1e-12
    )
    return {
        "child_seeds": [child.seed for child in children],
        "sum_height_max_error_m": float(np.max(np.abs(z - sum_z))),
        "sum_velocity_max_error_mps": float(np.max(np.abs(velocity - sum_velocity))),
        "run_in_midpoint_height_ratio": float(midpoint_z[0] / raw_z[0]),
        "run_in_applied_once": True,
    }


def check_seeds(cfg: dict) -> dict:
    times = np.linspace(0.0, 10.0, 97)
    report = {}
    presets = {**seven_road_scenarios(), "random": {"kind": "random"}}
    for name, preset in presets.items():
        first = np.stack(samples(road(cfg, {**preset, "seed": 181}), times))
        repeat = np.stack(samples(road(cfg, {**preset, "seed": 181}), times))
        different = np.stack(samples(road(cfg, {**preset, "seed": 182}), times))
        assert np.array_equal(first, repeat), name
        difference = float(np.max(np.abs(first - different)))
        if name in {"iso_b", "iso_c", "random", "composite"}:
            assert difference > 1e-5, name
        else:
            assert difference == 0, name
        report[name] = {"repeat_max_error": 0.0, "different_seed_max_difference": difference}
    return report


def check_preset_merge_isolation(_cfg: dict) -> dict:
    """Fixed evaluation presets must override unrelated training-road choices."""
    baseline = load_config()
    # Evaluation selects this speed separately; it is intentionally not one of
    # the polluted fields, so this check only audits preset merge isolation.
    baseline["road"]["speed_mps"] = 15.0
    pollution = {
        "kind": "iso_c",
        "tracks": "left",
        "run_in_m": 40.0,
        "left_right_correlation": 0.0,
        "phase_rad": 1.0,
        "level": "C",
        "gdn0_m3": 384e-6,
        "n_min_per_m": 0.07,
        "n_max_per_m": 0.5,
        "spectral_modes": 32,
        "height_m": 0.19,
        "length_m": 5.0,
        "start_m": 1.0,
        "wavelength_m": 1.7,
        "transition_length_m": 2.0,
        "roughness_m": 0.1,
        "components": [{"kind": "flat"}],
    }
    polluted = deepcopy(baseline)
    polluted["road"].update(pollution)
    times = np.r_[[-0.1, 0.0, 0.1, 0.2], np.linspace(0.001, 10.0, 1001)]
    seeds = (431, 890)
    report = {}
    for name, preset in seven_road_scenarios().items():
        maximum_height_error = maximum_velocity_error = 0.0
        for seed in seeds:
            fixed = {**preset, "seed": seed}
            reference = road(baseline, fixed)
            merged = road(polluted, fixed)
            expected_z, expected_velocity = samples(reference, times)
            actual_z, actual_velocity = samples(merged, times)
            assert np.max(np.abs(expected_z)) > 1e-5, "Isolation check needs a nonzero road"
            assert reference.speed == merged.speed == 15.0
            np.testing.assert_allclose(actual_z, expected_z, rtol=0, atol=1e-12)
            np.testing.assert_allclose(actual_velocity, expected_velocity, rtol=0, atol=1e-12)
            maximum_height_error = max(
                maximum_height_error, float(np.max(np.abs(actual_z - expected_z)))
            )
            maximum_velocity_error = max(
                maximum_velocity_error,
                float(np.max(np.abs(actual_velocity - expected_velocity))),
            )
        report[name] = {
            "maximum_height_error_m": maximum_height_error,
            "maximum_velocity_error_mps": maximum_velocity_error,
        }
    return {
        "polluted_training_road_fields": pollution,
        "seeds": list(seeds),
        "samples_per_seed": len(times),
        "evaluation_speed_mps": 15.0,
        "roads": report,
    }


def check_iso_spectrum(cfg: dict) -> dict:
    times = np.linspace(0.0, 30.0, 181)
    b = road(cfg, {"kind": "iso_b", "seed": 9981})
    c = road(cfg, {"kind": "iso_c", "seed": 9981})
    b_signal = np.stack(samples(b, times))
    c_signal = np.stack(samples(c, times))
    np.testing.assert_allclose(c_signal, 2 * b_signal, rtol=1e-12, atol=1e-12)
    targets = {}
    for level, coefficient in (("iso_b", 64e-6), ("iso_c", 256e-6)):
        # Independently integrate the specified continuous one-sided PSD on a
        # dense grid; do not reuse mode amplitudes or the synthesis bin formula.
        frequency = np.geomspace(0.02, 2.0, 20_001)
        psd = coefficient * (frequency / 0.1) ** -2
        integrated = float(np.trapezoid(psd, frequency))
        for modes in (32, 256, 512):
            profile = road(cfg, {"kind": level, "spectral_modes": modes})
            metadata = profile.metadata()["spectral"]
            assert metadata["gdn0_m3"] == coefficient
            np.testing.assert_allclose(metadata["expected_rms_m"] ** 2, integrated, rtol=1e-7)
        targets[level] = {
            "gdn0_m3": coefficient,
            "integrated_psd_variance_m2": integrated,
            "expected_rms_m": math.sqrt(integrated),
        }
    return {"c_to_b_amplitude_ratio": 2.0, "spatial_band_cycles_per_m": [0.02, 2.0], **targets}


def check_ensemble(cfg: dict) -> dict:
    count = 2048
    expected_rms = math.sqrt(64e-6 * 0.1**2 * (1 / 0.02 - 1 / 2.0))
    report = {}
    for correlation in (0.0, 0.6, 1.0):
        measured = []
        for seed in range(27_000, 27_000 + count):
            profile = road(
                cfg,
                {
                    "kind": "iso_b",
                    "seed": seed,
                    "left_right_correlation": correlation,
                    "run_in_m": 0.0,
                },
            )
            # Fixed spatial location, independent phase realizations. This is an
            # ensemble test, not a claim about a short driving-window RMS.
            measured.append(profile.sample(100.0 / profile.speed)[0][:2])
        values = np.asarray(measured)
        rms = np.sqrt(np.mean(values**2, axis=0))
        estimated_correlation = float(np.corrcoef(values.T)[0, 1])
        relative_error = np.abs(rms / expected_rms - 1)
        assert np.all(relative_error < 0.08), (correlation, rms, expected_rms)
        assert abs(estimated_correlation - correlation) < 0.08
        report[str(correlation)] = {
            "left_right_rms_m": rms.tolist(),
            "relative_rms_errors": relative_error.tolist(),
            "left_right_mean_m": values.mean(axis=0).tolist(),
            "estimated_correlation": estimated_correlation,
        }
    return {
        "independent_seeds_per_case": count,
        "seed_range_inclusive": [27_000, 27_000 + count - 1],
        "expected_rms_m": expected_rms,
        "relative_rms_tolerance": 0.08,
        "absolute_correlation_tolerance": 0.08,
        "interpretation": "Fixed ensemble audit; not measured-road certification or per-seed rescaling.",
        "correlations": report,
    }


def check_legacy(cfg: dict, reference: str) -> dict:
    result = subprocess.run(
        ["git", "show", f"{reference}:chrono/chrono_suspension/roads.py"],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    source = result.stdout
    if "seven_road_scenarios" in source:
        raise ValueError("The legacy ref already has the new API; choose the pre-expansion commit")
    module = ModuleType("chrono_legacy_road_audit")
    exec(compile(source, f"git:{reference}:roads.py", "exec"), module.__dict__)
    vehicle = cfg["vehicle"]
    times = np.r_[[-0.1, 0, 0.2, 0.6], np.linspace(0.001, 10, 201)]
    report = {}
    for kind in ("random", "bump", "sine", "pothole", "left_bump"):
        largest = 0.0
        for seed in (42, 183):
            scenario = {
                "kind": kind,
                "speed_mps": 15.0 if seed == 42 else 8.5,
                "height_m": 0.03 if seed == 42 else 0.017,
                "length_m": 0.8,
                "start_m": 9.0,
                "wavelength_m": 5.0,
                "roughness_m": 0.008,
                "left_right_correlation": 0.6,
            }
            old = module.FourWheelRoad(
                scenario, vehicle["front_distance_m"], vehicle["rear_distance_m"], seed
            )
            new = FourWheelRoad(
                scenario, vehicle["front_distance_m"], vehicle["rear_distance_m"], seed
            )
            old_values = np.stack(samples(old, times))
            new_values = np.stack(samples(new, times))
            np.testing.assert_allclose(new_values, old_values, rtol=0, atol=1e-12)
            largest = max(largest, float(np.max(np.abs(new_values - old_values))))
        report[kind] = {"maximum_height_or_velocity_error": largest}
    return {
        "git_ref": reference,
        "legacy_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "roads": report,
    }


def check_invalid_parameters(cfg: dict) -> dict:
    invalid = [
        {"kind": "missing"},
        {"speed_mps": 0},
        {"speed_mps": float("nan")},
        {"height_m": float("inf")},
        {"length_m": 0},
        {"wavelength_m": -1},
        {"run_in_m": -0.1},
        {"transition_length_m": 0},
        {"tracks": "front"},
        {"kind": "left_bump", "tracks": "both"},
        {"seed": -1},
        {"seed": True},
        {"seed": 1.5},
        {"left_right_correlation": -0.1},
        {"left_right_correlation": 1.1},
        {"left_right_correlation": float("nan")},
        {"kind": "random", "roughness_m": -0.1},
        {"kind": "iso_b", "level": "C"},
        {"kind": "iso", "level": "A"},
        {"kind": "iso_b", "gdn0_m3": 1e-6},
        {"kind": "iso_c", "gdn0_m3": 1.0},
        {"kind": "iso_b", "n_min_per_m": 0},
        {"kind": "iso_b", "n_min_per_m": 3, "n_max_per_m": 2},
        {"kind": "iso_b", "spectral_modes": 7},
        {"kind": "iso_b", "spectral_modes": 4097},
        {"kind": "iso_b", "spectral_modes": True},
        {"kind": "iso_b", "spectral_modes": 256.0},
        {"kind": "composite", "components": []},
        {"kind": "composite", "components": {}},
        {"kind": "composite", "components": [{}]},
        {"kind": "composite", "components": [{"kind": "composite"}]},
        {"kind": "composite", "components": [{"kind": "sine", "speed_mps": 99}]},
        {"kind": "composite", "components": [{"kind": "sine", "run_in_m": 99}]},
    ]

    def rejected(function: Callable) -> None:
        try:
            function()
        except ValueError:
            return
        raise AssertionError("Invalid input was accepted")

    for scenario in invalid:
        rejected(lambda scenario=scenario: road(cfg, scenario))
    rejected(lambda: FourWheelRoad({"kind": "flat"}, 0, 0, 1))
    valid = road(cfg, {"kind": "iso_b"})
    rejected(lambda: valid.sample(float("nan")))
    rejected(lambda: valid.sample(float("inf")))
    for steps, dt in ((-1, 0.01), (1.5, 0.01), (1, 0), (1, float("inf"))):
        rejected(lambda steps=steps, dt=dt: valid.preview(0, steps, dt))
    return {"rejected_constructor_cases": len(invalid) + 1, "rejected_sample_or_preview_cases": 6}


def check_environment_selection(_cfg: dict) -> dict:
    from chrono_suspension.env import ChronoSuspensionEnv

    config = load_config(PROJECT / "configs" / "seven_roads_training.yaml")
    report = {}
    for mode, dimension in (("privileged", 58), ("sensors", 130)):
        env = ChronoSuspensionEnv(config, observation_mode=mode)
        try:
            first_obs, first_info = env.reset(seed=681)
            first_road = samples(env.road, np.array([0.25, 1.0, 3.0]))
            repeat_obs, repeat_info = env.reset(seed=681)
            assert np.array_equal(first_obs, repeat_obs)
            assert first_info["road_metadata"] == repeat_info["road_metadata"]
            assert first_info["road_scenario_index"] == repeat_info["road_scenario_index"]
            assert np.array_equal(
                np.stack(first_road), np.stack(samples(env.road, np.array([0.25, 1.0, 3.0])))
            )
            assert env.action_space.shape == (4,) and first_obs.shape == (dimension,)
            counts = {kind: 0 for kind in SEVEN_KINDS}
            if mode == "privileged":
                for seed in range(700, 798):
                    _, info = env.reset(seed=seed)
                    kind = info["road_metadata"]["kind"]
                    counts[kind] += 1
                    assert config["road"]["scenarios"][info["road_scenario_index"]]["kind"] == kind
                assert all(counts.values()), counts
            obs, info = env.reset(seed=681, options={"scenario": {"kind": "flat"}})
            assert info["road_scenario_index"] is None and info["road_metadata"]["kind"] == "flat"
            obs, reward, terminated, _, info = env.step(np.zeros(4, dtype=np.float32))
            assert np.isfinite(obs).all() and math.isfinite(reward) and not terminated
            assert info["scenario"]["kind"] == "flat"
            report[mode] = {
                "observation_dimension": dimension,
                "action_dimension": 4,
                "repeated_reset_matches": True,
                "explicit_scenario_bypasses_mix": True,
                "sampled_kind_counts": counts if mode == "privileged" else None,
            }
        finally:
            env.close()
    return report


def check_passive_episodes(cfg: dict) -> dict:
    from chrono_suspension.env import ChronoSuspensionEnv

    config = load_config({**cfg, "physics": {**cfg["physics"], "episode_seconds": 10.0}})
    report = {}
    env = ChronoSuspensionEnv(config, observation_mode="privileged")
    try:
        for name, preset in seven_road_scenarios().items():
            print(f"  passive {name}: starting 10 s", flush=True)
            observation, reset_info = env.reset(seed=431, options={"scenario": preset})
            assert np.isfinite(observation).all()
            duration = square_integral = 0.0
            lost_contact = np.zeros(4)
            maximum_travel = np.zeros(4)
            minimum_load = np.full(4, np.inf)
            steps_with_travel_exceedance = 0
            terminated = truncated = False
            info = reset_info
            for _ in range(env.max_steps):
                observation, reward, terminated, truncated, info = env.step(
                    np.zeros(4, dtype=np.float32)
                )
                assert np.isfinite(observation).all() and math.isfinite(reward)
                substep = info["substep_metrics"]
                interval = substep["duration_s"]
                duration += interval
                square_integral += substep["body_acc_squared_mean"] * interval
                lost_contact += info["contact_loss_fraction"] * interval
                maximum_travel = np.maximum(maximum_travel, info["max_travel_m"])
                minimum_load = np.minimum(minimum_load, substep["tire_load_min_n"])
                steps_with_travel_exceedance += int(
                    np.any(info["max_travel_m"] > config["vehicle"]["travel_limit_m"])
                )
                if terminated or truncated:
                    break
            complete = not terminated and truncated and abs(duration - 10.0) < 1e-7
            report[name] = {
                "completed_10s": complete,
                "terminated": terminated,
                "truncated": truncated,
                "failure_reason": info.get("failure_reason"),
                "duration_s": duration,
                "body_acc_rms_mps2": math.sqrt(square_integral / duration) if duration else None,
                "maximum_travel_m": maximum_travel.tolist(),
                "minimum_tire_load_n": [
                    float(v) if math.isfinite(v) else None for v in minimum_load
                ],
                "wheel_unloaded_time_fraction": (lost_contact / max(duration, 1e-12)).tolist(),
                "control_steps_with_travel_exceedance": steps_with_travel_exceedance,
                "road_metadata": reset_info["road_metadata"],
            }
            print(
                f"  passive {name}: {'completed' if complete else 'terminated/incomplete'}",
                flush=True,
            )
    finally:
        env.close()
    return {
        "all_completed_10s": all(case["completed_10s"] for case in report.values()),
        "interpretation": "Finite passive episode checks only. Completion does not certify safe contact, travel, comfort or real-vehicle performance.",
        "roads": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument(
        "--with-env",
        action="store_true",
        help="Also check mixture resets and seven 10 s passive Gym episodes",
    )
    parser.add_argument(
        "--legacy-ref",
        default=LEGACY_REF,
        help="Git ref containing the original Chrono road implementation",
    )
    parser.add_argument(
        "--output", help="New results/ JSON path; existing files are never overwritten"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = result_path(
        args.output or REPOSITORY / "results" / "chrono" / "roads" / f"checks_{timestamp}.json"
    )
    if output.exists():
        raise FileExistsError(output)
    checks: list[tuple[str, Callable]] = [
        ("finite_samples_and_time_derivatives", check_derivatives),
        ("axle_delay_symmetry_and_tracks", check_delay_and_tracks),
        ("step_endpoints_and_integral", check_step),
        ("composite_sum_and_single_run_in", check_composite),
        ("seed_reproducibility", check_seeds),
        ("fixed_preset_merge_isolation", check_preset_merge_isolation),
        ("iso_class_scaling_and_psd_integral", check_iso_spectrum),
        ("independent_ensemble_rms_and_correlation", check_ensemble),
        ("legacy_compatibility", lambda cfg: check_legacy(cfg, args.legacy_ref)),
        ("invalid_parameters", check_invalid_parameters),
    ]
    if args.with_env:
        checks += [
            ("gym_mixture_selection", check_environment_selection),
            ("passive_episodes", check_passive_episodes),
        ]
    report: dict[str, Any] = {
        "created_utc": timestamp,
        "config": config,
        "with_environment": args.with_env,
        "road_source_sha256": hashlib.sha256(
            (PROJECT / "chrono_suspension" / "roads.py").read_bytes()
        ).hexdigest(),
        "versions": {"python": sys.version, "numpy": np.__version__},
        "checks": {},
        "limits": "Synthetic-road numerical/statistical checks, not ISO classification of measured roads or vehicle safety validation.",
    }
    for package in ("gymnasium", "stable-baselines3"):
        try:
            report["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["versions"][package] = "not installed"
    for name, check in checks:
        started = time.perf_counter()
        try:
            detail = check(config)
            passed = detail.get("all_completed_10s", True)
            report["checks"][name] = {"status": "passed" if passed else "failed", "detail": detail}
        except Exception as exc:
            report["checks"][name] = {"status": "failed", "error": repr(exc)}
        report["checks"][name]["wall_seconds"] = time.perf_counter() - started
        print(f"{name}: {report['checks'][name]['status']}", flush=True)
    report["all_requested_checks_passed"] = all(
        item["status"] == "passed" for item in report["checks"].values()
    )
    if not args.with_env:
        report["skipped"] = ["gym_mixture_selection", "passive_episodes"]
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Report: {output}")
    if not report["all_requested_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
