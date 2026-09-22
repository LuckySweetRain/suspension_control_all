"""Benchmark 1/2/4/8 independent headless Chrono workers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chrono_suspension.config import load_config  # noqa: E402
from chrono_suspension.training import RESULTS_ROOT, benchmark_workers  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--vector-steps", type=int, default=200)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--name", default=f"workers_{time.strftime('%Y%m%d_%H%M%S')}")
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("--name must be a simple folder name.")
    result = benchmark_workers(
        load_config(args.config),
        RESULTS_ROOT / "benchmarks" / args.name,
        counts=tuple(args.workers),
        vector_steps=args.vector_steps,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
