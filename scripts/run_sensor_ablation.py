"""Run paired sensor ablations serially on the shared CARLA/CarSim instance."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--suite", choices=("sensors", "state-mapping"), default="sensors")
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args()
    if args.episodes < 1 or args.eval_every < 1:
        parser.error("episodes and eval-every must be positive")
    args.output.mkdir(parents=True, exist_ok=False)

    def status(**data):
        (args.output / "queue_status.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    plans = [("measured", "measured"), ("estimated", "estimated")]
    if args.suite == "state-mapping":
        plans = [
            ("mapped_no_state", "mapping_no_state"),
            ("plain_keep", "measured"),
            ("mapped_keep", "mapping_keep"),
            ("plain_no_state", "plain_no_state"),
        ]
    for index, (mode, config_name) in enumerate(plans):
        run = args.output / mode
        status(
            stage="running",
            active=mode,
            episodes=args.episodes,
            queued=[name for name, _ in plans[index + 1 :]],
            suite=args.suite,
        )
        command = [
            sys.executable,
            "scripts/train_cosim_ppo.py",
            "--config",
            f"configs/cosim_ppo_{config_name}.yaml",
            "--carla-python",
            "E:/anaconda3/envs/carla0911/python.exe",
            "--episodes",
            str(args.episodes),
            "--seconds",
            "10",
            "--seed",
            str(args.seed),
            "--road-cycle",
            "wave",
            "bump",
            "pothole",
            "--eval-every",
            str(args.eval_every),
            "--artifact-policy",
            "compact",
            "--keep-every",
            str(args.eval_every),
            "--output",
            str(run),
        ]
        with (args.output / f"{mode}.log").open("w", encoding="utf-8") as log:
            learner = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            watcher = None
            with (args.output / f"{mode}_wandb.log").open("w", encoding="utf-8") as tracking:
                while learner.poll() is None:
                    if watcher is None and (run / "manifest.json").exists():
                        watcher = subprocess.Popen(
                            [
                                sys.executable,
                                "scripts/watch_cosim_wandb.py",
                                "--run-dir",
                                str(run),
                                "--step-offset",
                                "0",
                                "--output",
                                str(args.output / f"wandb_{mode}"),
                                "--name",
                                f"PPO {args.suite} {mode} seed{args.seed} {args.episodes}ep",
                                "--entity",
                                "yu2372432120-harbin-institute-of-technology",
                            ],
                            cwd=ROOT,
                            stdout=tracking,
                            stderr=subprocess.STDOUT,
                        )
                    time.sleep(2)
        if learner.returncode != 0:
            status(
                stage="failed",
                active=mode,
                returncode=learner.returncode,
                message="Stopped queue; no automatic retry",
            )
            raise SystemExit(learner.returncode)
        if args.suite == "state-mapping":
            subprocess.run(
                [sys.executable, "scripts/summarize_state_mapping.py", str(args.output)],
                cwd=ROOT,
                check=True,
            )
    status(stage="completed", episodes_per_group=args.episodes)


if __name__ == "__main__":
    main()
