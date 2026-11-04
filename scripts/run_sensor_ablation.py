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
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    def status(**data):
        (args.output / "queue_status.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    for mode in ("measured", "estimated"):
        run = args.output / mode
        status(
            stage="running",
            active=mode,
            episodes=args.episodes,
            queued=["estimated"] if mode == "measured" else [],
        )
        command = [
            sys.executable,
            "scripts/train_cosim_ppo.py",
            "--config",
            f"configs/cosim_ppo_{mode}.yaml",
            "--carla-python",
            "E:/anaconda3/envs/carla0911/python.exe",
            "--episodes",
            str(args.episodes),
            "--seconds",
            "10",
            "--seed",
            "20261104",
            "--road-cycle",
            "wave",
            "bump",
            "pothole",
            "--eval-every",
            "25",
            "--artifact-policy",
            "compact",
            "--keep-every",
            "25",
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
                                f"PPO sensor {mode} seed20261104 {args.episodes}ep",
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
    status(stage="completed", episodes_per_group=args.episodes)


if __name__ == "__main__":
    main()
