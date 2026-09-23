"""Evaluate one saved policy or reference controller on the held-out road suite."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chrono_suspension.config import load_config  # noqa: E402
from chrono_suspension.experiments import evaluate_cases, evaluation_cases  # noqa: E402
from chrono_suspension.training import RESULTS_ROOT  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    controller = parser.add_mutually_exclusive_group()
    controller.add_argument("--checkpoint", type=Path)
    controller.add_argument("--baseline", choices=("passive", "skyhook", "lqr"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--road-suite", choices=("legacy", "seven"), default="legacy")
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print fixed cases without running or writing files.",
    )
    parser.add_argument("--name", default=f"evaluation_{time.strftime('%Y%m%d_%H%M%S')}")
    args = parser.parse_args()
    if args.describe:
        print(
            json.dumps(
                {
                    "road_suite": args.road_suite,
                    "split": args.split,
                    "cases": evaluation_cases(args.split, args.road_suite),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.checkpoint is None and args.baseline is None:
        parser.error("Provide exactly one of --checkpoint or --baseline.")
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("--name must be a simple folder name")
    output = RESULTS_ROOT / "evaluation" / args.name
    if output.exists():
        parser.error("Evaluation output already exists; choose another --name")
    if args.checkpoint is not None and args.config is None:
        config = json.loads((args.checkpoint / "config.json").read_text(encoding="utf-8"))
    else:
        config = load_config(args.config)
    result = evaluate_cases(
        config,
        output,
        checkpoint=args.checkpoint,
        baseline=args.baseline,
        split=args.split,
        road_suite=args.road_suite,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))
    print(f"Evaluation: {output}")


if __name__ == "__main__":
    main()
