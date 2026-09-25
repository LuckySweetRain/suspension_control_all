"""Check comparison road coverage, split isolation and waveform coherence; no pytest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parent
sys.path.insert(0, str(PROJECT))

from chrono_suspension.comparison_cases import (  # noqa: E402
    DETERMINISTIC_NAMES,
    ROAD_NAMES,
    comparison_cases,
    design_metadata,
)
from chrono_suspension.config import load_config  # noqa: E402
from chrono_suspension.roads import FourWheelRoad  # noqa: E402


def physical_signature(case: dict) -> str:
    """Ignore bookkeeping and seed; retain every physically meaningful field."""
    return json.dumps(
        {key: value for key, value in case.items() if key not in {"id", "name", "seed"}},
        sort_keys=True,
    )


def validate_design(cfg: dict) -> dict:
    designs = {split: comparison_cases(split) for split in ("validation", "test")}
    metadata = design_metadata()
    length_fields = metadata["length_parameter_by_road"]
    counts = {}
    for split, cases in designs.items():
        expected_per_road = 4 if split == "validation" else 8
        assert Counter(case["kind"] for case in cases) == dict.fromkeys(
            ROAD_NAMES, expected_per_road
        )
        assert len({case["id"] for case in cases}) == len(cases)
        counts[split] = len(cases)
        for kind in (*DETERMINISTIC_NAMES, "composite"):
            selected = [case for case in cases if case["kind"] == kind]
            fields = ("height_m", length_fields[kind], "speed_mps")
            levels = [sorted({case[field] for case in selected}) for field in fields]
            assert all(len(values) == 2 for values in levels), (split, kind, levels)
            if split == "test":
                actual = {tuple(case[field] for field in fields) for case in selected}
                assert actual == set(product(*levels)), kind
            sides = Counter(case.get("local_tracks", case["tracks"]) for case in selected)
            assert sides["both"] == expected_per_road // 2
            assert sides["left"] == sides["right"] == expected_per_road // 4

    # Do not count a different RNG seed as a different deterministic test condition.
    validation_physics = {physical_signature(case) for case in designs["validation"]}
    test_physics = {physical_signature(case) for case in designs["test"]}
    assert validation_physics.isdisjoint(test_physics)
    for kind in ROAD_NAMES:
        valid_seeds = {c["seed"] for c in designs["validation"] if c["kind"] == kind}
        test_seeds = {c["seed"] for c in designs["test"] if c["kind"] == kind}
        assert valid_seeds.isdisjoint(test_seeds)
    # Callers may alter their local dictionaries without contaminating later runs.
    fresh = comparison_cases("test")
    fresh[-1]["components"][1]["height_m"] = 99.0
    assert comparison_cases("test")[-1]["components"][1]["height_m"] != 99.0

    vehicle = cfg["vehicle"]
    a, b = vehicle["front_distance_m"], vehicle["rear_distance_m"]
    times = np.linspace(0, 9.98, 67)
    maximum_delay_error = 0.0
    latest_local_event_passage = 0.0
    seed_invariant_cases = 0
    waveform_hashes = {}
    for cases in designs.values():
        for case in cases:
            road = FourWheelRoad(case, a, b, case["seed"])
            restored = FourWheelRoad.from_metadata(road.metadata(), a, b)
            heights = np.asarray([road.sample(float(t))[0] for t in times])
            velocities = np.asarray([road.sample(float(t))[1] for t in times])
            assert np.isfinite(heights).all() and np.isfinite(velocities).all()
            delayed = np.asarray(
                [road.sample(float(t) + road.wheelbase / road.speed)[0] for t in times]
            )
            delay_error = float(np.max(np.abs(heights[:, :2] - delayed[:, 2:])))
            maximum_delay_error = max(maximum_delay_error, delay_error)
            assert delay_error < 1e-10
            for t in (0.05, 0.8, 2.41, 5.67, 9.98):
                np.testing.assert_allclose(road.sample(t), restored.sample(t), atol=1e-12, rtol=0)
            stations = np.linspace(3.1, 60.0, 19)
            spatial, slopes = road.sample_tracks(stations)
            epsilon = 1e-5
            plus, _ = road.sample_tracks(stations + epsilon)
            minus, _ = road.sample_tracks(stations - epsilon)
            np.testing.assert_allclose(slopes, (plus - minus) / (2 * epsilon), atol=1e-7)
            assert np.isfinite(spatial).all()
            if case["kind"] in DETERMINISTIC_NAMES:
                altered_case = {**deepcopy(case), "seed": case["seed"] + 987654}
                altered = FourWheelRoad(altered_case, a, b, altered_case["seed"])
                altered_heights = np.asarray([altered.sample(float(t))[0] for t in times])
                np.testing.assert_array_equal(heights, altered_heights)
                seed_invariant_cases += 1
            local_events = case["components"] if case["kind"] == "composite" else [case]
            for event in local_events:
                if event["kind"] not in {"step", "bump", "pothole"}:
                    continue
                length = event.get("transition_length_m", event.get("length_m", 0.0))
                passed_at = (event["start_m"] + length + a + b) / case["speed_mps"]
                assert passed_at < min(8.0, cfg["physics"]["episode_seconds"]), case["id"]
                latest_local_event_passage = max(latest_local_event_passage, passed_at)
                local = FourWheelRoad({**event, "speed_mps": case["speed_mps"]}, a, b, 0)
                station = event["start_m"] + length * (1 if event["kind"] == "step" else 0.5)
                local_z, _ = local.sample(station / case["speed_mps"])
                np.testing.assert_allclose(np.max(np.abs(local_z[:2])), abs(event["height_m"]))
            waveform_hashes[case["id"]] = hashlib.sha256(heights.tobytes()).hexdigest()
    manifests = {
        split: hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()
        for split, cases in designs.items()
    }
    return {
        "passed": True,
        "counts": counts,
        "checks": [
            "all seven road classes and unique case IDs",
            "full amplitude/length/speed test factorial",
            "preassigned balanced single/both-side local excitations",
            "independent validation/test conditions and random seeds",
            "fresh nested case dictionaries on each API call",
            "finite height/velocity and exact metadata reconstruction",
            "rear wheels follow the same spatial tracks",
            "analytic spatial slopes agree with central differences",
            "deterministic waveforms are invariant to seed changes",
            "all local events pass both axles before eight seconds",
        ],
        "seed_invariant_deterministic_cases": seed_invariant_cases,
        "maximum_axle_delay_error_m": maximum_delay_error,
        "latest_local_event_rear_passage_s": latest_local_event_passage,
        "manifest_sha256": manifests,
        "waveform_sha256": waveform_hashes,
        "design": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT / "configs/carsim_dclass.yaml"))
    parser.add_argument("--output")
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (
        Path(args.output)
        if args.output
        else REPOSITORY / "results/chrono/comparison" / f"checks_{timestamp}.json"
    ).resolve()
    results_root = (REPOSITORY / "results").resolve()
    if output != results_root and results_root not in output.parents:
        raise ValueError("Generated comparison checks must be written under repository results/")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    report = validate_design(load_config(args.config))
    report["created_at_utc"] = timestamp
    report["config"] = str(Path(args.config).resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("passed", "counts", "checks")}, indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
