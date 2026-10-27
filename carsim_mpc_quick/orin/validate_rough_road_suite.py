"""Run isolated CARLA/CarSim/Orin road cases and verify mesh plus image preview.

Use conda environment carla0911. Each case owns a fresh child solver process.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time

from carla_stack import ROOT
from rough_road import PROFILES
from verify_camera_geometry import verify as verify_camera
from verify_rough_road import verify as verify_preview


def main(args):
    output = ROOT / "results/carla_rough_suite" / time.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True)
    manifest = dict(passed=False, cases=[])
    try:
        for profile in args.profiles:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "carsim_mpc_quick/orin/run_environment_loop.py"),
                    "--road-profile",
                    profile,
                    "--seconds",
                    str(args.seconds),
                    "--orin-host",
                    args.orin_host,
                ],
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            (output / (profile + ".log")).write_text(result.stdout, encoding="utf-8")
            matches = re.findall(r"^Artifacts: (.+)$", result.stdout, flags=re.MULTILINE)
            if result.returncode or not matches:
                raise RuntimeError(profile + " failed; inspect " + str(output / (profile + ".log")))
            from pathlib import Path

            folder = Path(matches[-1].strip())
            case = dict(profile=profile, artifacts=str(folder), passed=False)
            manifest["cases"].append(case)
            verify_preview(folder)
            captures = sorted(folder.glob("approach_*"))
            if not captures:
                raise AssertionError("Run did not reach a road feature")
            for capture in captures:
                verify_camera(capture)
            case["passed"] = True
            print(profile + ": PASS", flush=True)
        manifest["passed"] = True
    finally:
        (output / "report.json").write_text(json.dumps(manifest, indent=2))
        print("Suite report:", output / "report.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--orin-host", default="192.168.50.2")
    main(parser.parse_args())
