"""Run the twelve-group experiment under a hard wall-clock watchdog."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chrono_suspension.config import load_config  # noqa: E402
from chrono_suspension.experiments import (  # noqa: E402
    evaluation_cases,
    experiment_groups,
    run_experiment_suite,
)
from chrono_suspension.training import RESULTS_ROOT, write_json  # noqa: E402


def _run_child(config: dict, output: Path, options: dict) -> None:
    if os.name != "nt":
        os.setsid()
    result = run_experiment_suite(config, output, **options)
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "planned_timesteps",
                    "common_completed_steps",
                    "consumed_s",
                    "incomplete_training_groups",
                    "incomplete_evaluation_controllers",
                )
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


def _terminate_tree(process: mp.Process) -> None:
    if process.pid is None:
        return
    if os.name == "nt":
        # Kill the known child and its simulation descendants; do not leave
        # orphaned physics workers after enforcing the wall-time limit.
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)


def _record_stop(state_path: Path, status: str, consumed_s: float) -> None:
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    state.update(status=status, consumed_s=consumed_s)
    planned = state.get("planned_timesteps", state.get("max_timesteps", 1_000_000))
    groups = state.get("groups", experiment_groups())
    state["incomplete_training_groups"] = [
        {"id": group["id"], "completed_steps": group["completed_steps"], "planned_steps": planned}
        for group in groups
        if group["completed_steps"] < planned
    ]
    evaluation = state.get("evaluation", {})
    state["incomplete_evaluation_controllers"] = [
        label
        for label in ["passive", "skyhook", "lqr", *(group["id"] for group in groups)]
        if label not in evaluation or evaluation[label].get("incomplete_cases")
    ]
    state["watchdog_note"] = (
        "Process stopped; only atomically published checkpoints and case JSON files are valid."
    )
    write_json(state_path, state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--name", default=f"matrix_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--budget-hours", type=float, default=24.0)
    parser.add_argument("--max-steps", type=int, default=1_000_000)
    parser.add_argument(
        "--workers",
        type=int,
        choices=(1, 2, 4, 8),
        help="Omit to choose from the 1/2/4/8-worker throughput pilot.",
    )
    parser.add_argument("--pilot-steps", type=int, default=2048)
    parser.add_argument("--chunk-steps", type=int, default=16_384)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print the matrix and fixed test cases without running or writing files.",
    )
    args = parser.parse_args()
    if args.describe:
        print(
            json.dumps(
                {
                    "groups": experiment_groups(),
                    "test_cases": evaluation_cases(),
                    "training_fraction": 0.8,
                    "evaluation_fraction": 0.2,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("--name must be a simple folder name.")
    if not 0 < args.budget_hours <= 24 or not 0 < args.max_steps <= 1_000_000:
        parser.error("--budget-hours must be in (0,24] and --max-steps in (0,1000000].")
    config = load_config(args.config)
    output = RESULTS_ROOT / "experiments" / args.name
    state_path = output / "experiment.json"
    consumed = 0.0
    if state_path.exists():
        if not args.resume:
            parser.error("Experiment already exists; choose another --name or use --resume.")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        consumed = float(state.get("consumed_s", 0.0))
        if args.config is None:
            config = state["config"]
    remaining = args.budget_hours * 3600.0 - consumed
    if remaining <= 0:
        parser.error("The experiment's saved total time budget has been exhausted.")
    options = {
        "budget_hours": args.budget_hours,
        "max_timesteps": args.max_steps,
        "n_envs": args.workers,
        "device": args.device,
        "resume": args.resume,
        "pilot_steps": args.pilot_steps,
        "chunk_steps": args.chunk_steps,
    }
    process = mp.get_context("spawn").Process(target=_run_child, args=(config, output, options))
    print(f"Experiment: {output}\nRemaining hard budget: {remaining:.1f} s", flush=True)
    deadline = time.monotonic() + remaining
    process.start()
    try:
        while process.is_alive() and time.monotonic() < deadline:
            process.join(timeout=min(1.0, max(0.0, deadline - time.monotonic())))
        if process.is_alive():
            _terminate_tree(process)
            _record_stop(state_path, "watchdog_stopped", args.budget_hours * 3600.0)
            print("Hard budget reached; experiment processes stopped.", flush=True)
        else:
            # Include process startup/shutdown in persisted budget accounting,
            # rather than counting only time spent inside the child function.
            if state_path.exists():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                state["consumed_s"] = max(
                    float(state.get("consumed_s", 0.0)),
                    args.budget_hours * 3600.0 - max(0.0, deadline - time.monotonic()),
                )
                write_json(state_path, state)
            if process.exitcode:
                raise SystemExit(process.exitcode)
    except KeyboardInterrupt:
        _terminate_tree(process)
        _record_stop(
            state_path,
            "interrupted",
            args.budget_hours * 3600.0 - max(0.0, deadline - time.monotonic()),
        )
        raise


if __name__ == "__main__":
    main()
