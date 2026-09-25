"""Tune on validation, freeze, and compare passive/Skyhook/LQR on seven diverse roads."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chrono_suspension.comparison import experiment_plan, run_comparison  # noqa: E402
from chrono_suspension.config import PROJECT, load_config  # noqa: E402
from chrono_suspension.training import RESULTS_ROOT  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/carsim_dclass.yaml")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default=f"dclass_seven_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--report-only", type=Path, help="Existing comparison.json to render again")
    args = parser.parse_args()
    if args.report_only:
        from chrono_suspension.comparison_report import write_comparison_report

        data = json.loads(args.report_only.read_text(encoding="utf-8"))
        write_comparison_report(data, args.report_only.parent)
        return
    config = load_config(args.config)
    if args.describe:
        print(json.dumps(experiment_plan(config), ensure_ascii=False, indent=2))
        return
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("--name must be a simple directory name")
    output = RESULTS_ROOT / "comparison" / args.name
    result = run_comparison(config, output, args.workers)
    print(f"Comparison complete: {output}; elapsed {result['elapsed_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
