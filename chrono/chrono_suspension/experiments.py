"""Budgeted, paired PPO/SAC experiments and fixed physical evaluation cases."""

from __future__ import annotations

import csv
import json
import math
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from chrono_suspension.roads import seven_road_scenarios
from chrono_suspension.training import (
    align_steps,
    benchmark_workers,
    load_policy,
    train_chunk,
    write_json,
)

SCENARIOS = ("flat", "bump", "left_bump", "sine", "random")
TEST_SEEDS = (1001, 1002, 1003, 1004)
VALIDATION_SEEDS = (2001, 2002, 2003, 2004)
TRAINING_SEEDS = (42, 43, 44)


def evaluation_cases(split: str = "test", road_suite: str = "legacy") -> list[dict[str, Any]]:
    """Fixed 20/28-case suites; test seeds are never used for training/tuning."""
    if split not in ("test", "validation"):
        raise ValueError("split must be test or validation")
    if road_suite not in ("legacy", "seven"):
        raise ValueError("road_suite must be legacy or seven")
    seeds = TEST_SEEDS if split == "test" else VALIDATION_SEEDS
    scenarios = (
        {kind: {"kind": kind} for kind in SCENARIOS}
        if road_suite == "legacy"
        else seven_road_scenarios()
    )
    return [
        {"id": f"{name}_{seed}", **deepcopy(scenario), "seed": seed, "speed_mps": 15.0}
        for name, scenario in scenarios.items()
        for seed in seeds
    ]


def experiment_groups() -> list[dict[str, Any]]:
    """The scientific comparison is fixed: 2 algorithms x 2 observations x 3 seeds."""
    return [
        {
            "id": f"{algorithm}_{observation_mode}_seed{seed}",
            "algorithm": algorithm,
            "observation_mode": observation_mode,
            "seed": seed,
            "completed_steps": 0,
            "checkpoints": {},
            "status": "pending",
        }
        for seed in TRAINING_SEEDS
        for observation_mode in ("privileged", "sensors")
        for algorithm in ("ppo", "sac")
    ]


def _case_metrics(
    samples: list[dict[str, Any]], control_dt: float, step_infos: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Integrate physical metrics in FL, FR, RL, RR order.

    New environment records resolve every physics substep. Older records retain
    control-rate fallbacks where the corresponding substep statistic is absent.
    Positive and absorbed mechanical work are integrated separately; neither is
    an electrical-energy estimate. Travel is positive in suspension extension.
    """
    if not samples:
        raise ValueError("Physical metrics require at least one sample.")
    if step_infos is not None and len(step_infos) != len(samples):
        raise ValueError("Physical samples and step infos must have matching lengths.")
    required = (
        "body_acc_mps2",
        "roll_rad",
        "pitch_rad",
        "suspension_travel_m",
        "tire_load_n",
        "actuator_force_n",
        "mechanical_power_w",
    )
    arrays = {
        name: np.asarray([row[name] for row in samples], dtype=np.float64) for name in required
    }
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise FloatingPointError("Nonfinite physical signal in evaluation.")

    def rms(array: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(array))))

    power = arrays["mechanical_power_w"]
    travel = arrays["suspension_travel_m"]
    tire_load = arrays["tire_load_n"]
    positive_work = np.maximum(power, 0.0).sum(axis=0) * control_dt
    negative_work = -np.minimum(power, 0.0).sum(axis=0) * control_dt
    result = {
        "body_acc_rms_mps2": rms(arrays["body_acc_mps2"]),
        "roll_rms_rad": rms(arrays["roll_rad"]),
        "pitch_rms_rad": rms(arrays["pitch_rad"]),
        "body_acc_peak_mps2": float(np.abs(arrays["body_acc_mps2"]).max()),
        "roll_peak_rad": float(np.abs(arrays["roll_rad"]).max()),
        "pitch_peak_rad": float(np.abs(arrays["pitch_rad"]).max()),
        "suspension_travel_peak_m": float(np.abs(travel).max()),
        "suspension_travel_rms_m": rms(travel),
        "suspension_travel_rms_per_wheel_m": np.sqrt(np.mean(travel**2, axis=0)).tolist(),
        "suspension_travel_peak_per_wheel_m": np.abs(travel).max(axis=0).tolist(),
        "suspension_travel_min_per_wheel_m": travel.min(axis=0).tolist(),
        "suspension_travel_max_per_wheel_m": travel.max(axis=0).tolist(),
        "tire_load_min_n": float(tire_load.min()),
        "tire_load_min_per_wheel_n": tire_load.min(axis=0).tolist(),
        "tire_load_std_n_per_wheel": tire_load.std(axis=0).tolist(),
        "wheel_unloaded_fraction": float(np.mean(tire_load <= 0.0)),
        "wheel_unloaded_fraction_per_wheel": np.mean(tire_load <= 0.0, axis=0).tolist(),
        "actuator_force_rms_n": rms(arrays["actuator_force_n"]),
        "positive_mechanical_work_per_wheel_j": positive_work.tolist(),
        "negative_mechanical_work_magnitude_per_wheel_j": negative_work.tolist(),
    }
    if step_infos and all("static_tire_load_n" in info for info in step_infos):
        static_loads = np.asarray([info["static_tire_load_n"] for info in step_infos], dtype=float)
        if not np.isfinite(static_loads).all() or np.any(static_loads <= 0):
            raise FloatingPointError("Static tire loads must be finite and positive.")
        if not np.allclose(static_loads, static_loads[0], rtol=0, atol=1e-10):
            raise ValueError("Static tire loads must remain fixed within an evaluation case.")
        result["tire_dynamic_load_rms_ratio_per_wheel"] = np.sqrt(
            np.mean((tire_load / static_loads - 1.0) ** 2, axis=0)
        ).tolist()
    if step_infos and all("substep_metrics" in info for info in step_infos):
        substeps = [info["substep_metrics"] for info in step_infos]
        durations = np.asarray([row["duration_s"] for row in substeps], dtype=float)
        if durations.sum() <= 0 or not np.isfinite(durations).all() or np.any(durations < 0):
            raise FloatingPointError("Invalid integrated duration in physical metrics.")
        # A first-substep failure has no valid samples and may carry +/-inf
        # initial extrema; it contributes no duration and must not poison a case.
        valid = durations > 0
        substeps = [row for row, keep in zip(substeps, valid, strict=True) if keep]
        valid_infos = [row for row, keep in zip(step_infos, valid, strict=True) if keep]
        durations = durations[valid]

        def weighted_mean(source: str) -> np.ndarray:
            return np.average(
                np.asarray([row[source] for row in substeps], dtype=float),
                axis=0,
                weights=durations,
            )

        for output, source in (
            ("body_acc_rms_mps2", "body_acc_squared_mean"),
            ("roll_rms_rad", "roll_squared_mean"),
            ("pitch_rms_rad", "pitch_squared_mean"),
            ("actuator_force_rms_n", "actuator_force_squared_mean"),
        ):
            squared = np.asarray([np.mean(row[source]) for row in substeps], dtype=float)
            result[output] = float(np.sqrt(np.average(squared, weights=durations)))
        positive_work = np.sum([row["mechanical_work_positive_j"] for row in substeps], axis=0)
        negative_work = -np.sum([row["mechanical_work_negative_j"] for row in substeps], axis=0)
        result["positive_mechanical_work_per_wheel_j"] = positive_work.tolist()
        result["negative_mechanical_work_magnitude_per_wheel_j"] = negative_work.tolist()
        minimum_load = np.min([row["tire_load_min_n"] for row in substeps], axis=0)
        result["tire_load_min_per_wheel_n"] = minimum_load.tolist()
        result["tire_load_min_n"] = float(minimum_load.min())
        peak_travel = np.max([info["max_travel_m"] for info in valid_infos], axis=0)
        result["suspension_travel_peak_per_wheel_m"] = peak_travel.tolist()
        result["suspension_travel_peak_m"] = float(peak_travel.max())
        unloading = np.average(
            [info["contact_loss_fraction"] for info in valid_infos], axis=0, weights=durations
        )
        result["wheel_unloaded_fraction_per_wheel"] = unloading.tolist()
        result["wheel_unloaded_fraction"] = float(unloading.mean())
        for name in ("body_acc_peak_mps2", "roll_peak_rad", "pitch_peak_rad"):
            if all(name in row for row in substeps):
                result[name] = float(max(row[name] for row in substeps))
        if all("suspension_travel_squared_mean" in row for row in substeps):
            travel_squared_mean = weighted_mean("suspension_travel_squared_mean")
            result["suspension_travel_rms_per_wheel_m"] = np.sqrt(travel_squared_mean).tolist()
            result["suspension_travel_rms_m"] = float(np.sqrt(travel_squared_mean.mean()))
        for source, output, reduction in (
            ("suspension_travel_min_m", "suspension_travel_min_per_wheel_m", np.min),
            ("suspension_travel_max_m", "suspension_travel_max_per_wheel_m", np.max),
        ):
            if all(source in row for row in substeps):
                result[output] = reduction([row[source] for row in substeps], axis=0).tolist()
        if all("tire_load_squared_mean" in row and "tire_load_mean_n" in row for row in substeps):
            variance = (
                weighted_mean("tire_load_squared_mean") - weighted_mean("tire_load_mean_n") ** 2
            )
            result["tire_load_std_n_per_wheel"] = np.sqrt(np.maximum(variance, 0)).tolist()
        if all("tire_dynamic_load_ratio_squared_mean" in row for row in substeps):
            result["tire_dynamic_load_rms_ratio_per_wheel"] = np.sqrt(
                weighted_mean("tire_dynamic_load_ratio_squared_mean")
            ).tolist()
    if step_infos and all("static_tire_load_n" in info for info in step_infos):
        result["tire_load_min_ratio_per_wheel"] = (
            np.asarray(result["tire_load_min_per_wheel_n"]) / static_loads[0]
        ).tolist()
    result["positive_mechanical_work_j"] = float(positive_work.sum())
    result["negative_mechanical_work_magnitude_j"] = float(negative_work.sum())
    result["absolute_mechanical_work_j"] = float(positive_work.sum() + negative_work.sum())
    result["net_mechanical_work_j"] = float(positive_work.sum() - negative_work.sum())
    if not all(np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("Nonfinite integrated physical metrics.")
    return result


def evaluate_cases(
    config: dict[str, Any],
    output_dir: Path,
    *,
    checkpoint: Path | None = None,
    baseline: str | None = None,
    observation_mode: str = "privileged",
    split: str = "test",
    road_suite: str = "legacy",
    deadline: float | None = None,
) -> dict[str, Any]:
    """Evaluate the same disturbances, duration and action limits for every policy.

    Evaluation is direct Gym stepping: there is no VecEnv auto-reset that can
    accidentally replace a terminal observation or start an unrequested case.
    """
    from chrono_suspension.env import ChronoSuspensionEnv

    if (checkpoint is None) == (baseline is None):
        raise ValueError("Provide exactly one of checkpoint or baseline.")
    if baseline is not None and baseline not in ("passive", "skyhook", "lqr"):
        raise ValueError("Unknown baseline controller.")
    cases = evaluation_cases(split, road_suite)
    if checkpoint is not None:
        metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
        saved_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        if saved_config != config:
            raise ValueError("Evaluation config differs from the checkpoint's physical config.")
        observation_mode = metadata["observation_mode"]
    output_dir.mkdir(parents=True, exist_ok=True)
    env = ChronoSuspensionEnv(config=config, observation_mode=observation_mode, render_mode=None)
    completed: list[dict[str, Any]] = []
    dt = float(config["physics"]["control_dt"])
    step_limit = math.ceil(float(config["physics"]["episode_seconds"]) / dt) + 1
    policy = None if checkpoint is None else load_policy(checkpoint, env)
    try:
        for case in cases:
            if deadline is not None and time.monotonic() >= deadline:
                break
            observation, reset_info = env.reset(seed=case["seed"], options={"scenario": case})
            samples: list[dict[str, Any]] = []
            step_infos: list[dict[str, Any]] = []
            reward_sum = 0.0
            terminated = truncated = False
            status = "step_limit"
            started = time.monotonic()
            info: dict[str, Any] = {}
            for _ in range(step_limit):
                if deadline is not None and time.monotonic() >= deadline:
                    status = "budget_exhausted"
                    break
                action = env.baseline_action(baseline) if policy is None else policy(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                if not np.isfinite(observation).all() or not np.isfinite(reward):
                    raise FloatingPointError(f"Nonfinite evaluation for {case['id']}")
                samples.append(info["metrics"])
                step_infos.append(info)
                reward_sum += float(reward)
                if terminated or truncated:
                    status = "terminated" if terminated else "completed"
                    break
            metric_error = None
            try:
                metrics = _case_metrics(samples, dt, step_infos) if samples else {}
            except FloatingPointError as exc:
                if not terminated:
                    raise
                metrics = {}
                metric_error = str(exc)
            row = {
                "case": case,
                "road_metadata": reset_info.get("road_metadata"),
                "status": status,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "termination_reason": info.get("failure_reason", info.get("termination_reason")),
                "metric_error": metric_error,
                "steps": len(samples),
                "duration_s": sum(
                    info.get("substep_metrics", {}).get("duration_s", dt) for info in step_infos
                ),
                "elapsed_s": time.monotonic() - started,
                "return": reward_sum,
                "metrics": metrics,
            }
            completed.append(row)
            write_json(output_dir / f"{case['id']}.json", row)
            if status == "budget_exhausted":
                break
    finally:
        env.close()
    finished_ids = {
        row["case"]["id"] for row in completed if row["status"] in ("completed", "terminated")
    }
    successful = [row for row in completed if row["status"] == "completed"]
    scalar_keys = (
        "body_acc_rms_mps2",
        "roll_rms_rad",
        "pitch_rms_rad",
        "suspension_travel_peak_m",
        "suspension_travel_rms_m",
        "tire_load_min_n",
        "actuator_force_rms_n",
        "positive_mechanical_work_j",
        "negative_mechanical_work_magnitude_j",
        "wheel_unloaded_fraction",
    )
    summary = {
        "controller": str(checkpoint) if checkpoint is not None else baseline,
        "split": split,
        "road_suite": road_suite,
        "cases": completed,
        "finished_cases": len(finished_ids),
        "successful_cases": len(successful),
        "terminated_cases": sum(row["terminated"] for row in completed),
        "incomplete_cases": [case["id"] for case in cases if case["id"] not in finished_ids],
        "successful_case_mean": {
            key: float(np.mean([row["metrics"][key] for row in successful])) if successful else None
            for key in scalar_keys
        },
        "aggregation_note": "Means exclude terminated/incomplete cases; always report failure counts alongside them.",
        "energy_note": "Mechanical actuator work only; not electrical consumption or recovered battery energy.",
        "sampling_note": "All current physical RMS, peaks, tire statistics, unloading fractions and actuator work use physics substeps, weighted by valid duration. Per-wheel values are FL, FR, RL, RR. Travel is negative in compression and positive in extension. Older records use control-rate fallbacks for missing substep statistics.",
    }
    write_json(output_dir / "summary.json", summary)
    with (output_dir / "cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case", "status", "return", *scalar_keys])
        writer.writeheader()
        for row in completed:
            writer.writerow(
                {
                    "case": row["case"]["id"],
                    "status": row["status"],
                    "return": row["return"],
                    **{key: row["metrics"].get(key) for key in scalar_keys},
                }
            )
    return summary


def _latest_complete_checkpoint(group: dict[str, Any]) -> Path | None:
    steps = int(group["completed_steps"])
    return Path(group["checkpoints"][str(steps)]) if steps else None


def _plan_target(state: dict[str, Any], remaining_training_s: float) -> int:
    """Cost includes initialization, learning, normalization and checkpoint I/O."""
    cost_per_step = sum(
        group["pilot_elapsed_s"] / group["completed_steps"] for group in state["groups"]
    )
    # Pilot includes SAC warmup and may understate sustained gradient costs.
    # A 50% reserve here complements the separate 20% evaluation time reserve.
    additional_steps = int(max(0.0, remaining_training_s) / (1.5 * cost_per_step))
    requested = min(state["max_timesteps"], state["pilot_steps"] + additional_steps)
    return max(state["pilot_steps"], align_steps(requested, state["n_envs"]))


def run_experiment_suite(
    config: dict[str, Any],
    output_dir: Path,
    *,
    budget_hours: float = 24.0,
    max_timesteps: int = 1_000_000,
    n_envs: int | None = None,
    device: str = "auto",
    resume: bool = False,
    pilot_steps: int = 2048,
    chunk_steps: int = 16_384,
    road_suite: str = "legacy",
) -> dict[str, Any]:
    """Run a fixed twelve-group matrix with equal completed-step comparisons.

    Use scripts/run_experiments.py for a process watchdog enforcing the hard
    wall-time limit.  This library function uses cooperative step deadlines.
    Installation and development are outside this function's time accounting.
    """
    if not 0 < budget_hours <= 24 or max_timesteps <= 0 or max_timesteps > 1_000_000:
        raise ValueError("budget_hours must be in (0,24], max_timesteps in (0,1000000].")
    test_cases = evaluation_cases(road_suite=road_suite)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "experiment.json"
    started = time.monotonic()
    if state_path.exists():
        if not resume:
            raise FileExistsError("Experiment directory already contains state; use --resume.")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["config"] != config:
            raise ValueError("Experiment configuration changed on resume.")
        if state["budget_hours"] != budget_hours or state["max_timesteps"] != max_timesteps:
            raise ValueError("Budget and maximum steps must match the saved experiment.")
        if n_envs is not None and state.get("n_envs") not in (None, n_envs):
            raise ValueError("Worker count cannot change when resuming optimizer/replay state.")
        if state.get("road_suite", "legacy") != road_suite:
            raise ValueError("Road suite cannot change when resuming an experiment.")
        if state.get("test_cases", test_cases) != test_cases:
            raise ValueError("Road scenarios differ from the saved experiment's test cases.")
        state.setdefault("road_suite", road_suite)
        state.setdefault("test_cases", test_cases)
        if state["status"] in ("finished", "budget_exhausted", "watchdog_stopped"):
            return state
        consumed = float(state.get("consumed_s", 0.0))
    else:
        consumed = 0.0
        state = {
            "version": 1,
            "config": config,
            "budget_hours": budget_hours,
            "max_timesteps": max_timesteps,
            "n_envs": n_envs,
            "device": device,
            "groups": experiment_groups(),
            "status": "benchmark",
            "consumed_s": 0.0,
            "common_completed_steps": 0,
            "road_suite": road_suite,
            "test_cases": test_cases,
            "evaluation": {},
            "started_unix_s": time.time(),
        }
        write_json(state_path, state)
    total_seconds = budget_hours * 3600.0
    deadline = started + max(0.0, total_seconds - consumed)
    training_deadline = started + max(0.0, total_seconds * 0.8 - consumed)

    def persist() -> None:
        state["consumed_s"] = consumed + time.monotonic() - started
        write_json(state_path, state)

    def train_group(group: dict[str, Any], target: int, pilot: bool = False) -> bool:
        group["status"] = "running"
        persist()
        print(f"Training {group['id']}: {group['completed_steps']} -> {target}", flush=True)
        result = train_chunk(
            config,
            output_dir / "runs" / group["id"],
            algorithm=group["algorithm"],
            observation_mode=group["observation_mode"],
            seed=group["seed"],
            n_envs=state["n_envs"],
            target_timesteps=target,
            checkpoint=_latest_complete_checkpoint(group),
            device=device,
            deadline=training_deadline,
        )
        group["last_chunk"] = result
        if result["completed"]:
            group["completed_steps"] = result["num_timesteps"]
            group["checkpoints"][str(result["num_timesteps"])] = result["checkpoint"]
            group["status"] = "checkpointed"
            if pilot:
                group["pilot_elapsed_s"] = result["elapsed_s"]
        else:
            group["status"] = "budget_exhausted"
        persist()
        print(
            f"  {group['status']}; steps={group['completed_steps']}; "
            f"elapsed={result['elapsed_s']:.1f}s",
            flush=True,
        )
        return bool(result["completed"])

    try:
        if state["n_envs"] is None:
            bench = benchmark_workers(config, output_dir / "benchmark", deadline=training_deadline)
            state["n_envs"] = bench["recommended_workers"]
        workers = state["n_envs"]
        state.setdefault(
            "pilot_steps", max(align_steps(pilot_steps, workers, upward=True), 256 * workers)
        )
        state.setdefault(
            "chunk_steps", max(align_steps(chunk_steps, workers, upward=True), 256 * workers)
        )
        if state["pilot_steps"] > max_timesteps:
            raise ValueError("Maximum steps must accommodate one aligned pilot rollout.")
        state["status"] = "pilot"
        persist()
        for group in state["groups"]:
            if group["completed_steps"] >= state["pilot_steps"]:
                continue
            if not train_group(group, state["pilot_steps"], pilot=True):
                break
        if all(group["completed_steps"] >= state["pilot_steps"] for group in state["groups"]):
            state["common_completed_steps"] = min(
                group["completed_steps"] for group in state["groups"]
            )
            if "planned_timesteps" not in state:
                state["planned_timesteps"] = _plan_target(
                    state, training_deadline - time.monotonic()
                )
                state["pilot_costs"] = {
                    group["id"]: {
                        "elapsed_s": group["pilot_elapsed_s"],
                        "transitions_per_s": state["pilot_steps"] / group["pilot_elapsed_s"],
                        "estimated_seconds_at_planned_target": group["pilot_elapsed_s"]
                        * state["planned_timesteps"]
                        / state["pilot_steps"],
                    }
                    for group in state["groups"]
                }
            state["status"] = "training"
            persist()
            round_index = int(state.get("round_index", 0))
            while (
                state["common_completed_steps"] < state["planned_timesteps"]
                and time.monotonic() < training_deadline
            ):
                target = min(
                    state["planned_timesteps"],
                    state["common_completed_steps"] + state["chunk_steps"],
                )
                groups = state["groups"]
                offset = round_index % len(groups)
                order = groups[offset:] + groups[:offset]
                completed_round = True
                for group in order:
                    if group["completed_steps"] >= target:
                        continue
                    if not train_group(group, target):
                        completed_round = False
                        break
                if not completed_round:
                    break
                state["common_completed_steps"] = target
                round_index += 1
                state["round_index"] = round_index
                persist()
        state["status"] = "evaluation"
        persist()
        common = state["common_completed_steps"]
        if common:
            # Shared baselines only need one run, not one per algorithm/seed.
            controllers: list[tuple[str, Path | None, str | None]] = [
                (name, None, name) for name in ("passive", "skyhook", "lqr")
            ]
            controllers.extend(
                (group["id"], Path(group["checkpoints"][str(common)]), None)
                for group in state["groups"]
            )
            for label, checkpoint, baseline in controllers:
                if time.monotonic() >= deadline:
                    break
                previous = state["evaluation"].get(label)
                if previous is not None and not previous["incomplete_cases"]:
                    continue
                try:
                    print(f"Evaluating {label} at common step {common}", flush=True)
                    summary = evaluate_cases(
                        config,
                        output_dir / "evaluation" / f"common_{common}" / label,
                        checkpoint=checkpoint,
                        baseline=baseline,
                        road_suite=road_suite,
                        deadline=deadline,
                    )
                    state["evaluation"][label] = {
                        key: value for key, value in summary.items() if key != "cases"
                    }
                    print(
                        f"  finished={summary['finished_cases']}; "
                        f"terminated={summary['terminated_cases']}",
                        flush=True,
                    )
                except Exception as exc:
                    state["evaluation"][label] = {
                        "error": str(exc),
                        "road_suite": road_suite,
                        "incomplete_cases": [case["id"] for case in test_cases],
                    }
                    print(f"  evaluation failed: {exc}", flush=True)
                persist()
        requested = align_steps(max_timesteps, state["n_envs"])
        planned = state.get("planned_timesteps", state["pilot_steps"])
        state["incomplete_training_groups"] = [
            {
                "id": group["id"],
                "completed_steps": group["completed_steps"],
                "planned_steps": planned,
            }
            for group in state["groups"]
            if group["completed_steps"] < planned
        ]
        state["requested_cap_not_reached"] = [
            group["id"] for group in state["groups"] if group["completed_steps"] < requested
        ]
        expected_labels = ["passive", "skyhook", "lqr", *(group["id"] for group in state["groups"])]
        state["incomplete_evaluation_controllers"] = [
            label
            for label in expected_labels
            if label not in state["evaluation"] or state["evaluation"][label]["incomplete_cases"]
        ]
        state["status"] = "budget_exhausted" if time.monotonic() >= deadline else "finished"
        state["scientific_status"] = (
            "exploratory: finite checks and short training do not prove convergence or superiority"
        )
        persist()
        return state
    except BaseException as exc:
        state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["traceback"] = traceback.format_exc()
        persist()
        raise
    finally:
        persist()
