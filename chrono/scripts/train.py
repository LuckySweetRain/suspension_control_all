"""Train or resume a single PPO/SAC policy without visualization workers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chrono_suspension.config import load_config  # noqa: E402
from chrono_suspension.training import RESULTS_ROOT, align_steps, train_chunk  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--algorithm", choices=("ppo", "sac"))
    parser.add_argument("--observation-mode", choices=("privileged", "sensors"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--steps",
        type=int,
        default=65_536,
        help="Absolute aggregate transition target, rounded up to a PPO rollout.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--name", default=None, help="Run folder name below results/chrono/training."
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Checkpoint directory, including config and normalization statistics.",
    )
    parser.add_argument(
        "--budget-seconds",
        type=float,
        help="Cooperative deadline; interrupted rollout is marked incomplete.",
    )
    args = parser.parse_args()
    metadata = {}
    if args.resume is not None:
        metadata = json.loads((args.resume / "metadata.json").read_text(encoding="utf-8"))
    algorithm = args.algorithm or metadata.get("algorithm", "ppo")
    observation = args.observation_mode or metadata.get("observation_mode", "privileged")
    seed = args.seed if args.seed is not None else metadata.get("seed", 42)
    workers = args.workers if args.workers is not None else metadata.get("n_envs", 4)
    if args.config is None and args.resume is not None:
        config = json.loads((args.resume / "config.json").read_text(encoding="utf-8"))
    else:
        config = load_config(args.config)
    name = args.name or f"{algorithm}_{observation}_{time.strftime('%Y%m%d_%H%M%S')}"
    if Path(name).name != name or name in (".", ".."):
        parser.error("--name must be a simple folder name.")
    if not 1 <= workers <= 32 or args.steps <= 0:
        parser.error("--workers must be 1..32 and --steps must be positive.")
    output = RESULTS_ROOT / "training" / name
    if output.exists() and args.resume is None:
        parser.error("Run directory exists; choose a new name or resume a checkpoint.")
    if args.budget_seconds is not None and args.budget_seconds <= 0:
        parser.error("--budget-seconds must be positive.")
    deadline = None if args.budget_seconds is None else time.monotonic() + args.budget_seconds
    target = align_steps(args.steps, workers, upward=True)
    print(
        f"Training {algorithm}/{observation}: aggregate target {target}, workers {workers}",
        flush=True,
    )
    result = train_chunk(
        config,
        output,
        algorithm=algorithm,
        observation_mode=observation,
        seed=seed,
        n_envs=workers,
        target_timesteps=target,
        checkpoint=args.resume,
        device=args.device,
        deadline=deadline,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
