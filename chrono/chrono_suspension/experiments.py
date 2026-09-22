"""Budgeted, paired PPO/SAC experiments and fixed physical evaluation cases."""

from __future__ import annotations

import csv
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

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


def evaluation_cases(split: str = "test") -> list[dict[str, Any]]:
    """Twenty predefined cases; test seeds are never used for training/tuning."""
    if split not in ("test", "validation"):
        raise ValueError("split must be test or validation")
    seeds = TEST_SEEDS if split == "test" else VALIDATION_SEEDS
    return [
        {"id": f"{kind}_{seed}", "kind": kind, "seed": seed, "speed_mps": 15.0}
        for kind in SCENARIOS
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
    result = {
        "body_acc_rms_mps2": rms(arrays["body_acc_mps2"]),
        "roll_rms_rad": rms(arrays["roll_rad"]),
        "pitch_rms_rad": rms(arrays["pitch_rad"]),
        "suspension_travel_peak_m": float(np.abs(arrays["suspension_travel_m"]).max()),
        "suspension_travel_rms_m": rms(arrays["suspension_travel_m"]),
        "tire_load_min_n": float(arrays["tire_load_n"].min()),
        "tire_load_std_n_per_wheel": arrays["tire_load_n"].std(axis=0).tolist(),
        "wheel_unloaded_fraction": float(np.mean(arrays["tire_load_n"] <= 0.0)),
        "actuator_force_rms_n": rms(arrays["actuator_force_n"]),
        "positive_mechanical_work_j": float(np.maximum(power, 0.0).sum() * control_dt),
        "negative_mechanical_work_magnitude_j": float(-np.minimum(power, 0.0).sum() * control_dt),
    }
    if step_infos and all("substep_metrics" in info for info in step_infos):
        substeps = [info["substep_metrics"] for info in step_infos]
        durations = np.asarray([row["duration_s"] for row in substeps], dtype=float)
        if durations.sum() <= 0 or not np.isfinite(durations).all():
            raise FloatingPointError("Invalid integrated duration in physical metrics.")
        for output, source in (
            ("body_acc_rms_mps2", "body_acc_squared_mean"),
            ("roll_rms_rad", "roll_squared_mean"),
            ("pitch_rms_rad", "pitch_squared_mean"),
            ("actuator_force_rms_n", "actuator_force_squared_mean"),
        ):
            squared = np.asarray([np.mean(row[source]) for row in substeps], dtype=float)
            result[output] = float(np.sqrt(np.average(squared, weights=durations)))
        result["positive_mechanical_work_j"] = float(
            sum(np.sum(row["mechanical_work_positive_j"]) for row in substeps)
        )
        result["negative_mechanical_work_magnitude_j"] = float(
            -sum(np.sum(row["mechanical_work_negative_j"]) for row in substeps)
        )
        result["tire_load_min_n"] = float(min(np.min(row["tire_load_min_n"]) for row in substeps))
        result["suspension_travel_peak_m"] = float(
            max(np.max(info["max_travel_m"]) for info in step_infos)
        )
        result["wheel_unloaded_fraction"] = float(
            np.average(
                [np.mean(info["contact_loss_fraction"]) for info in step_infos], weights=durations
            )
        )
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
    if checkpoint is not None:
        metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
        saved_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        if saved_config != config:
            raise ValueError("Evaluation config differs from the checkpoint's physical config.")
        observation_mode = metadata["observation_mode"]
    output_dir.mkdir(parents=True, exist_ok=True)
    env = ChronoSuspensionEnv(config=config, observation_mode=observation_mode, render_mode=None)
    cases = evaluation_cases(split)
    completed: list[dict[str, Any]] = []
    dt = float(config["physics"]["control_dt"])
    step_limit = math.ceil(float(config["physics"]["episode_seconds"]) / dt) + 1
    policy = None if checkpoint is None else load_policy(checkpoint, env)
    try:
        for case in cases:
            if deadline is not None and time.monotonic() >= deadline:
                break
            observation, _ = env.reset(seed=case["seed"], options={"scenario": case})
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
        "sampling_note": "RMS acceleration/attitude/force, work, travel peaks, minimum load and unloading use physics substeps; travel RMS and load std use control-rate samples.",
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
) -> dict[str, Any]:
    """Run a fixed twelve-group matrix with equal completed-step comparisons.

    Use scripts/run_experiments.py for a process watchdog enforcing the hard
    wall-time limit.  This library function uses cooperative step deadlines.
    Installation and development are outside this function's time accounting.
    """
    if not 0 < budget_hours <= 24 or max_timesteps <= 0 or max_timesteps > 1_000_000:
        raise ValueError("budget_hours must be in (0,24], max_timesteps in (0,1000000].")
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
            "test_cases": evaluation_cases(),
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
                        "incomplete_cases": [case["id"] for case in evaluation_cases()],
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
