"""Summarize recorded physical transitions without calling incomplete episodes successful."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(run: Path) -> dict:
    episodes = []
    for folder in sorted(run.glob("episode_*")):
        log = folder / "transitions.jsonl"
        rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        if not rows:
            episodes.append(dict(episode=folder.name, completed=False, steps=0))
            continue
        y = np.asarray([row["exports"] for row in rows])
        command = np.asarray([row["executed_action"] for row in rows])
        plans = [row["plan"] for row in rows if row.get("plan") is not None]
        episode = dict(
            episode=folder.name,
            completed=(folder / "summary.json").is_file()
            and not (folder / "failure.json").exists(),
            steps=len(rows),
            sim_seconds=rows[-1]["time"],
            reward_sum=sum(row["reward"] for row in rows),
            # Full episode, including initialization: deliberately no hidden settling exclusion.
            vertical_accel_rms_mps2=float(np.sqrt(np.mean((y[:, 40] * 9.80665) ** 2))),
            vertical_accel_peak_mps2=float(np.max(abs(y[:, 40] * 9.80665))),
            pitch_accel_rms_radps2=float(np.sqrt(np.mean(y[:, 43] ** 2))),
            final_speed_kph=float(y[-1, 76]),
            travelled_m=float(np.linalg.norm(np.diff(y[:, 77:79], axis=0), axis=1).sum()),
            max_commanded_axis_force_N=float(abs(command).max()),
            active_steps=int(np.any(command != 0, axis=1).sum()),
            qwen_plans=len(plans),
            qwen_context_valid_steps=sum(
                len(row["observation"]) == 104
                and row.get("plan") is not None
                and row["observation"][22] > 0
                for row in rows
            ),
        )
        if plans:
            episode["qwen_inference_ms_p50_p95_max"] = np.percentile(
                [plan["inference_ms"] for plan in plans], [50, 95, 100]
            ).tolist()
        asynchronous = [plan for plan in plans if "arrival_age_s" in plan]
        if asynchronous:
            for field in ("arrival_age_s", "request_wall_seconds"):
                episode["qwen_async_" + field + "_p50_p95_max"] = np.percentile(
                    [plan[field] for plan in asynchronous], [50, 95, 100]
                ).tolist()
        remote = [row["inference"] for row in rows if row.get("inference") is not None]
        if remote:
            episode["orin_inference_samples"] = len(remote)
            for field in ("rpc_ms", "compute_ms"):
                values = np.asarray([item[field] for item in remote], dtype=float)
                episode["orin_" + field + "_p50_p95_p99_max"] = np.percentile(
                    values, [50, 95, 99, 100]
                ).tolist()
                episode["orin_" + field + "_over_10ms_fraction"] = float(np.mean(values > 10))
        if all("physical_constraints" in row for row in rows):
            constraints = [row["physical_constraints"] for row in rows]
            episode["max_relative_suspension_travel_m"] = max(
                r["max_relative_travel_m"] for r in constraints
            )
            episode["relative_travel_over_80mm_fraction"] = float(
                np.mean([r["travel_violation_fraction"] for r in constraints])
            )
            episode["low_load_fraction"] = float(
                np.mean([r["low_load_fraction"] for r in constraints])
            )
            episode["constraint_metric_source"] = "native_unmapped_state"
        elif len(rows) > 1 and all(len(row["observation"]) == 104 for row in rows):
            # Observation at step 1 uses exports produced by step 0. Recover
            # the same equilibrium reference used by the recorded policy schema.
            initial_jounce = y[0, 44:48] - 80 * np.asarray(rows[1]["observation"])[4:8]
            travel = (y[:, 44:48] - initial_jounce) / 1000
            episode["max_relative_suspension_travel_m"] = float(abs(travel).max())
            episode["relative_travel_over_80mm_fraction"] = float(np.mean(abs(travel) > 0.08))
            episode["constraint_metric_source"] = "legacy_v1_equilibrium_reconstruction"
        else:
            # Sensor history / atanh coordinates cannot recover the physical origin.
            episode["constraint_metric_source"] = "unavailable_legacy_sensor_schema"
        episode["min_native_tire_load_N"] = float(y[:, 52:56].min())
        episodes.append(episode)
    result = dict(
        run=str(run), scope="recorded 10 ms samples, no substep peak guarantee", episodes=episodes
    )
    (run / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    for run in args.runs:
        print(json.dumps(summarize(run), indent=2))
