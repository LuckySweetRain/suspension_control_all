"""Paired reference controllers, validation-only selection and frozen test evaluation."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import merge_config
from .training import limit_worker_threads, write_json


def candidate_controllers() -> list[dict]:
    """A small predeclared search, including the original reference settings."""
    candidates = [{"id": "passive", "baseline": "passive", "parameters": {}}]
    candidates += [
        {
            "id": f"skyhook_c{damping}",
            "baseline": "skyhook",
            "parameters": {"damping_nspm": float(damping)},
        }
        for damping in (750, 1500, 3000, 6000)
    ]
    candidates += [
        {
            "id": f"lqr_r{r_weight:g}_v{velocity_scale:g}",
            "baseline": "lqr",
            "parameters": {
                "r_weight": r_weight,
                "body_velocity_weight_scale": velocity_scale,
            },
        }
        for r_weight in (0.00025, 0.001, 0.004)
        for velocity_scale in (1.0, 4.0)
    ]
    return candidates


def selected_config(config: dict, candidate: dict) -> dict:
    return merge_config(config, {"controllers": {candidate["baseline"]: candidate["parameters"]}})


def digest(payload: dict | list) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def selection_score(rows: list[dict]) -> dict:
    """Fixed dimensionless comfort/contact/effort objective; no test inputs."""
    by_road: dict[str, list[float]] = {}
    failures = sum(row["status"] != "completed" for row in rows)
    for row in rows:
        m = row["metrics"]
        if row["status"] != "completed":
            value = 1e6
        else:
            value = (
                m["body_acc_rms_mps2"] ** 2
                + 0.2 * (m["roll_rms_rad"] ** 2 + m["pitch_rms_rad"] ** 2) / np.deg2rad(1.0) ** 2
                + 0.1 * (m["suspension_travel_rms_m"] / 0.09) ** 2
                + 0.1 * np.mean(np.square(m["tire_dynamic_load_rms_ratio_per_wheel"]))
                + 20.0 * m["wheel_unloaded_fraction"]
                + 20.0 * max(0.0, m["suspension_travel_peak_m"] / 0.09 - 1.0) ** 2
                + 0.01 * m["absolute_mechanical_work_j"] / (100.0 * row["duration_s"])
            )
        by_road.setdefault(row["case"]["kind"], []).append(float(value))
    road_scores = {kind: float(np.mean(values)) for kind, values in by_road.items()}
    return {
        "failed_cases": failures,
        "score": float(np.mean(list(road_scores.values()))),
        "road_scores": road_scores,
        "case_count": len(rows),
    }


def run_case(config: dict, candidate: dict, case: dict, output: str, save_signals: bool) -> dict:
    """A process owns its entire Chrono lifetime; never send C++ objects across IPC."""
    from .env import ChronoSuspensionEnv
    from .experiments import _case_metrics

    limit_worker_threads()
    env = ChronoSuspensionEnv(selected_config(config, candidate), render_mode=None)
    started = time.monotonic()
    samples, infos, commands = [], [], []
    try:
        observation, reset_info = env.reset(seed=case["seed"], options={"scenario": case})
        status, reward_sum = "step_limit", 0.0
        info = {}
        for _ in range(env.max_steps):
            action = env.baseline_action(candidate["baseline"])
            observation, reward, terminated, truncated, info = env.step(action)
            if not np.isfinite(observation).all() or not np.isfinite(reward):
                raise FloatingPointError(f"Nonfinite result: {candidate['id']} / {case['id']}")
            samples.append(info["metrics"])
            infos.append(info)
            commands.append(action)
            reward_sum += float(reward)
            if terminated or truncated:
                status = "terminated" if terminated else "completed"
                break
        metric_error = None
        try:
            metrics = _case_metrics(samples, env.control_dt, infos)
        except FloatingPointError as exc:
            if status != "terminated":
                raise
            metrics, metric_error = {}, str(exc)
        row = {
            "case": case,
            "controller": candidate,
            "road_metadata": reset_info["road_metadata"],
            "static_tire_load_n": reset_info["static_tire_load_n"].tolist(),
            "status": status,
            "termination_reason": info.get("failure_reason"),
            "metric_error": metric_error,
            "steps": len(samples),
            "duration_s": sum(item["substep_metrics"]["duration_s"] for item in infos),
            "elapsed_s": time.monotonic() - started,
            "return": reward_sum,
            "metrics": metrics,
        }
        directory = Path(output)
        write_json(directory / f"{case['id']}.json", row)
        if save_signals:
            np.savez_compressed(
                directory / f"{case['id']}.npz",
                time_s=np.asarray([item["time_s"] for item in infos]),
                command_normalized=np.asarray(commands),
                **{key: np.asarray([item[key] for item in samples]) for key in samples[0]},
            )
        return row
    finally:
        env.close()


def _run_matrix(
    pool: ProcessPoolExecutor,
    config: dict,
    candidates: list[dict],
    cases: list[dict],
    output: Path,
    *,
    save_signals: bool,
) -> dict[str, list[dict]]:
    results: dict[str, list[dict]] = {candidate["id"]: [] for candidate in candidates}
    futures = {
        pool.submit(
            run_case, config, candidate, case, str(output / candidate["id"]), save_signals
        ): candidate["id"]
        for candidate in candidates
        for case in cases
    }
    started = time.monotonic()
    for index, future in enumerate(as_completed(futures), 1):
        candidate_id = futures[future]
        row = future.result()
        results[candidate_id].append(row)
        print(
            f"{output.name}: {index}/{len(futures)} {candidate_id} {row['case']['id']} "
            f"{row['status']} elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )
    order = {case["id"]: index for index, case in enumerate(cases)}
    for rows in results.values():
        rows.sort(key=lambda row: order[row["case"]["id"]])
    return results


def experiment_plan(config: dict) -> dict:
    from .comparison_cases import comparison_cases, design_metadata

    if config["physics"]["episode_seconds"] != 10.0:
        raise ValueError("This frozen road design requires 10-second episodes")
    return {
        "schema": "chrono-baseline-comparison-v1",
        "config": config,
        "design": design_metadata(),
        "validation_cases": comparison_cases("validation"),
        "test_cases": comparison_cases("test"),
        "candidates": candidate_controllers(),
        "selection_protocol": {
            "source_split": "validation",
            "selection": "One global candidate per family; lexicographic (failure count, score, id).",
            "score_formula": "a_rms^2 + 0.2*((roll_rms/1deg)^2+(pitch_rms/1deg)^2) "
            "+ 0.1*(travel_rms/0.09m)^2 + 0.1*mean(dynamic_load_ratio_rms^2) "
            "+ 20*wheel_unloaded_fraction + 20*max(0,travel_peak/0.09m-1)^2 "
            "+ 0.01*absolute_mechanical_work/(100W*duration); mean within road then across 7 roads.",
            "failed_case_score": 1e6,
            "interpretation": "Engineering multi-objective choice; not an ISO comfort rating or guaranteed optimum.",
            "test_usage": "Test outcomes are evaluated once after a persisted controller freeze; no retuning.",
        },
        "sampling": {
            "metrics_dt_s": config["physics"]["dt"],
            "time_series_dt_s": config["physics"]["control_dt"],
            "initial_state_included": False,
            "note": "Summary metrics integrate physics substeps; npz signals use control samples.",
        },
    }


def run_comparison(config: dict, output: Path, workers: int = 4) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {output}")
    if workers < 1:
        raise ValueError("workers must be positive")
    plan = experiment_plan(config)
    # Persist all ranges, objective and candidates before observing any outcomes.
    plan["created_utc"] = datetime.now(timezone.utc).isoformat()
    source_root = Path(__file__).resolve().parent
    plan["source_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source_root.glob("*.py"))
        if path.name != "comparison_report.py"
    }
    plan["plan_sha256"] = digest(plan)
    write_json(output / "plan.json", plan)
    started = time.monotonic()
    limit_worker_threads()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        validation = _run_matrix(
            pool,
            config,
            plan["candidates"],
            plan["validation_cases"],
            output / "validation",
            save_signals=False,
        )
        scores = {key: selection_score(rows) for key, rows in validation.items()}
        selected = {}
        for baseline in ("passive", "skyhook", "lqr"):
            choices = [item for item in plan["candidates"] if item["baseline"] == baseline]
            winner = min(
                choices,
                key=lambda item: (
                    scores[item["id"]]["failed_cases"],
                    scores[item["id"]]["score"],
                    item["id"],
                ),
            )
            selected[baseline] = {**winner, "validation": scores[winner["id"]]}
        frozen = {
            "frozen_utc": datetime.now(timezone.utc).isoformat(),
            "plan_sha256": plan["plan_sha256"],
            "selected_controllers": selected,
            "validation_scores": scores,
        }
        frozen["freeze_sha256"] = digest(frozen)
        write_json(output / "frozen_controllers.json", frozen)
        for baseline, winner in selected.items():
            write_json(output / f"config_{baseline}.json", selected_config(config, winner))
        print("CONTROLLERS FROZEN: " + json.dumps(selected), flush=True)
        test = _run_matrix(
            pool,
            config,
            list(selected.values()),
            plan["test_cases"],
            output / "test",
            save_signals=True,
        )
    result = deepcopy(plan)
    result.update(frozen)
    result["test_results"] = {key: test[value["id"]] for key, value in selected.items()}
    result["elapsed_s"] = time.monotonic() - started
    result["workers"] = workers
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    # Assert identical realized terrain between controllers, including random phases.
    for index in range(len(plan["test_cases"])):
        terrains = [result["test_results"][key][index]["road_metadata"] for key in selected]
        if any(terrain != terrains[0] for terrain in terrains[1:]):
            raise AssertionError("Unpaired realized road in test results")
    write_json(output / "comparison.json", result)
    from .comparison_report import write_comparison_report

    write_comparison_report(result, output)
    return result
