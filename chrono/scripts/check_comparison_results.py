"""Independently audit a completed frozen seven-road baseline comparison."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROAD_NAMES = ("iso_b", "iso_c", "sine", "step", "pothole", "bump", "composite")
FAMILIES = ("passive", "skyhook", "lqr")
WHEEL_ORDER = ("FL", "FR", "RL", "RR")
VECTOR_SIGNALS = (
    "command_normalized",
    "suspension_travel_m",
    "suspension_velocity_mps",
    "tire_load_n",
    "actuator_force_n",
    "mechanical_power_w",
)
SCALAR_SIGNALS = ("time_s", "body_acc_mps2", "roll_rad", "pitch_rad")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_hash(payload: dict, excluded: str) -> str:
    contents = {key: value for key, value in payload.items() if key != excluded}
    return hashlib.sha256(json.dumps(contents, sort_keys=True).encode("utf-8")).hexdigest()


def independent_scores(rows: list[dict]) -> dict:
    """Reproduce the published equation without importing comparison code."""
    scores_by_road: dict[str, list[float]] = defaultdict(list)
    failures = 0
    for row in rows:
        if row["status"] != "completed":
            failures += 1
            score = 1_000_000.0
        else:
            m = row["metrics"]
            load_ratios = m["tire_dynamic_load_rms_ratio_per_wheel"]
            score = (
                m["body_acc_rms_mps2"] ** 2
                + (0.2 / math.radians(1) ** 2) * (m["roll_rms_rad"] ** 2 + m["pitch_rms_rad"] ** 2)
                + 0.1 * (m["suspension_travel_rms_m"] / 0.09) ** 2
                + (0.1 / len(load_ratios)) * sum(value * value for value in load_ratios)
                + 20 * m["wheel_unloaded_fraction"]
                + 20 * max(0, m["suspension_travel_peak_m"] / 0.09 - 1) ** 2
                + (0.01 / 100) * m["absolute_mechanical_work_j"] / row["duration_s"]
            )
        scores_by_road[row["case"]["kind"]].append(score)
    road_scores = {name: sum(values) / len(values) for name, values in scores_by_road.items()}
    return {
        "failed_cases": failures,
        "score": sum(road_scores.values()) / len(road_scores),
        "road_scores": road_scores,
        "case_count": len(rows),
    }


def close(actual: Any, expected: Any, name: str, atol: float = 1e-9) -> None:
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), f"{name}: keys differ"
        for key, value in expected.items():
            close(actual[key], value, f"{name}.{key}", atol)
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=atol, err_msg=name)


def inspect_metrics(row: dict) -> None:
    metrics = row["metrics"]
    if not metrics:
        assert row["status"] != "completed" and row["metric_error"], "Missing valid metrics"
        return
    for key, value in metrics.items():
        array = np.asarray(value, dtype=float)
        assert np.isfinite(array).all(), f"Nonfinite {key}"
        if "per_wheel" in key:
            assert array.shape == (4,), f"Wheel dimension for {key}: {array.shape}"
    static = np.asarray(row["static_tire_load_n"], dtype=float)
    assert static.shape == (4,) and np.isfinite(static).all() and np.all(static > 0)
    positive = np.asarray(metrics["positive_mechanical_work_per_wheel_j"])
    absorbed = np.asarray(metrics["negative_mechanical_work_magnitude_per_wheel_j"])
    assert np.all(positive >= 0) and np.all(absorbed >= 0)
    close(metrics["positive_mechanical_work_j"], positive.sum(), "positive work")
    close(metrics["negative_mechanical_work_magnitude_j"], absorbed.sum(), "absorbed work")
    close(metrics["absolute_mechanical_work_j"], positive.sum() + absorbed.sum(), "absolute work")
    close(metrics["net_mechanical_work_j"], positive.sum() - absorbed.sum(), "net work")
    close(
        metrics["tire_load_min_ratio_per_wheel"],
        np.asarray(metrics["tire_load_min_per_wheel_n"]) / static,
        "minimum load ratio",
    )
    close(metrics["tire_load_min_n"], min(metrics["tire_load_min_per_wheel_n"]), "minimum load")
    unloading = np.asarray(metrics["wheel_unloaded_fraction_per_wheel"])
    assert np.all((unloading >= 0) & (unloading <= 1))
    close(metrics["wheel_unloaded_fraction"], unloading.mean(), "unloading fraction")
    travel_rms = np.asarray(metrics["suspension_travel_rms_per_wheel_m"])
    travel_peak = np.asarray(metrics["suspension_travel_peak_per_wheel_m"])
    travel_min = np.asarray(metrics["suspension_travel_min_per_wheel_m"])
    travel_max = np.asarray(metrics["suspension_travel_max_per_wheel_m"])
    close(metrics["suspension_travel_rms_m"], np.sqrt(np.mean(travel_rms**2)), "travel RMS")
    close(metrics["suspension_travel_peak_m"], travel_peak.max(), "travel peak")
    close(travel_peak, np.maximum(abs(travel_min), abs(travel_max)), "signed travel extrema")
    assert np.all(travel_rms >= 0) and np.all(travel_peak + 1e-12 >= travel_rms)
    assert np.all(travel_max >= travel_min)
    assert np.all(np.asarray(metrics["tire_load_std_n_per_wheel"]) >= 0)
    assert np.all(np.asarray(metrics["tire_dynamic_load_rms_ratio_per_wheel"]) >= 0)
    for quantity, unit in (("body_acc", "mps2"), ("roll", "rad"), ("pitch", "rad")):
        assert metrics[f"{quantity}_peak_{unit}"] + 1e-12 >= metrics[f"{quantity}_rms_{unit}"] >= 0
    if row["controller"]["baseline"] == "passive":
        close(metrics["actuator_force_rms_n"], 0, "passive actuator force", atol=1e-12)
        close(positive, np.zeros(4), "passive positive work", atol=1e-12)
        close(absorbed, np.zeros(4), "passive absorbed work", atol=1e-12)


def inspect_row(row: dict, case: dict, controller: dict) -> None:
    assert row["case"] == case, "Row differs from predeclared case"
    assert row["controller"] == controller, "Controller differs from declared/frozen parameters"
    assert row["status"] in {"completed", "terminated", "step_limit"}, "Unknown status"
    assert isinstance(row["steps"], int) and 0 < row["steps"] <= 1000
    assert math.isfinite(row["return"])
    assert math.isfinite(row["duration_s"]) and 0 < row["duration_s"] <= 10 + 1e-8
    if row["status"] == "completed":
        assert row["steps"] == 1000 and row["termination_reason"] is None
        assert row["metric_error"] is None
        close(row["duration_s"], 10, "completed duration")
    elif row["status"] == "terminated":
        assert row["termination_reason"], "Termination must state the reason"
    inspect_metrics(row)


def inspect_signals(path: Path, row: dict) -> None:
    with np.load(path, allow_pickle=False) as signals:
        assert set(signals.files) == set((*VECTOR_SIGNALS, *SCALAR_SIGNALS)), "Signal fields differ"
        count = row["steps"]
        for key in signals.files:
            expected = (count, 4) if key in VECTOR_SIGNALS else (count,)
            assert signals[key].shape == expected, f"{key} has shape {signals[key].shape}"
            assert np.isfinite(signals[key]).all(), f"Nonfinite NPZ {key}"
        times = signals["time_s"]
        intervals = np.diff(np.r_[0.0, times])
        close(times[-1], row["duration_s"], "signal/metric duration", atol=1e-7)
        if row["status"] == "completed":
            close(intervals, np.full(count, 0.01), "control sample spacing", atol=1e-8)
        else:
            close(intervals[:-1], np.full(count - 1, 0.01), "prefix sample spacing", atol=1e-8)
            assert 0 < intervals[-1] <= 0.01 + 1e-8
        assert np.all(abs(signals["command_normalized"]) <= 1 + 1e-7)
        close(
            signals["mechanical_power_w"],
            signals["actuator_force_n"] * signals["suspension_velocity_mps"],
            "per-wheel force-velocity mechanical power",
        )
        m = row["metrics"]
        if m:
            assert np.all(
                np.asarray(m["suspension_travel_peak_per_wheel_m"]) + 1e-10
                >= abs(signals["suspension_travel_m"]).max(axis=0)
            ), "Physics travel peak missed a control sample"
            assert np.all(
                np.asarray(m["tire_load_min_per_wheel_n"])
                <= signals["tire_load_n"].min(axis=0) + 1e-8
            ), "Physics minimum load missed a control sample"
            for quantity, unit in (("body_acc", "mps2"), ("roll", "rad"), ("pitch", "rad")):
                assert (
                    m[f"{quantity}_peak_{unit}"] + 1e-10 >= abs(signals[f"{quantity}_{unit}"]).max()
                ), f"Physics {quantity} peak missed a control sample"
        if row["controller"]["baseline"] == "passive":
            for key in ("command_normalized", "actuator_force_n", "mechanical_power_w"):
                close(signals[key], np.zeros((count, 4)), f"passive {key}", atol=1e-12)


def audit(directory: Path) -> dict:
    # comparison.json is the atomically persisted completion marker. Never open
    # partial test results or examine scores during the controller search.
    comparison_path = directory / "comparison.json"
    if not comparison_path.is_file():
        raise FileNotFoundError("comparison.json is not ready; wait for the run to finish")
    comparison = read_json(comparison_path)
    plan = read_json(directory / "plan.json")
    frozen = read_json(directory / "frozen_controllers.json")
    failures: list[dict] = []
    checks = 0

    def check(name: str, operation) -> None:
        nonlocal checks
        checks += 1
        try:
            operation()
        except (AssertionError, KeyError, TypeError, ValueError, OSError) as exc:
            failures.append({"check": name, "error": str(exc)})

    def manifests() -> None:
        assert canonical_hash(plan, "plan_sha256") == plan["plan_sha256"], "Plan hash differs"
        assert canonical_hash(frozen, "freeze_sha256") == frozen["freeze_sha256"], (
            "Freeze hash differs"
        )
        assert frozen["plan_sha256"] == plan["plan_sha256"]
        for key, value in plan.items():
            assert comparison[key] == value, f"Completion changed plan field {key}"
        for key, value in frozen.items():
            assert comparison[key] == value, f"Completion changed frozen field {key}"
        assert plan["selection_protocol"]["source_split"] == "validation"
        assert plan["sampling"]["metrics_dt_s"] == 0.0005
        assert plan["sampling"]["time_series_dt_s"] == 0.01
        assert (
            datetime.fromisoformat(plan["created_utc"])
            <= datetime.fromisoformat(frozen["frozen_utc"])
            <= datetime.fromisoformat(comparison["finished_utc"])
        )

    check("manifest_hashes_and_freeze_links", manifests)
    validation_cases = {case["id"]: case for case in plan["validation_cases"]}
    test_cases = {case["id"]: case for case in plan["test_cases"]}
    candidates = {item["id"]: item for item in plan["candidates"]}
    selected = frozen["selected_controllers"]

    def design() -> None:
        assert len(validation_cases) == len(plan["validation_cases"]) == 28
        assert len(test_cases) == len(plan["test_cases"]) == 56
        assert len(candidates) == len(plan["candidates"]) == 11
        assert set(validation_cases).isdisjoint(test_cases)
        assert set(selected) == set(FAMILIES)
        assert set(comparison["test_results"]) == set(FAMILIES)
        assert Counter(case["kind"] for case in validation_cases.values()) == dict.fromkeys(
            ROAD_NAMES, 4
        )
        assert Counter(case["kind"] for case in test_cases.values()) == dict.fromkeys(ROAD_NAMES, 8)
        assert {case["seed"] for case in validation_cases.values()}.isdisjoint(
            case["seed"] for case in test_cases.values()
        )
        assert Counter(item["baseline"] for item in candidates.values()) == {
            "passive": 1,
            "skyhook": 4,
            "lqr": 6,
        }

    check("predeclared_counts_split_isolation", design)
    source_root = Path(__file__).resolve().parents[1] / "chrono_suspension"

    def sources() -> None:
        for name, expected_sha256 in plan["source_sha256"].items():
            assert Path(name).name == name, "Source manifest must contain plain filenames"
            actual_sha256 = hashlib.sha256((source_root / name).read_bytes()).hexdigest()
            assert actual_sha256 == expected_sha256, f"Source changed after plan freeze: {name}"

    check("current_source_sha256", sources)
    rows_by_candidate: dict[str, list[dict]] = {}
    metadata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    status_counts: dict[str, Counter] = {"validation": Counter(), "test": Counter()}
    expected_validation_files = set()
    expected_test_files = set()
    for candidate_id, candidate in candidates.items():
        rows = []
        for case_id, case in validation_cases.items():
            path = directory / "validation" / candidate_id / f"{case_id}.json"
            expected_validation_files.add(path)
            try:
                row = read_json(path)
            except (OSError, ValueError) as exc:
                failures.append(
                    {"check": f"validation/{candidate_id}/{case_id}", "error": str(exc)}
                )
                continue
            rows.append(row)
            status_counts["validation"][row.get("status", "missing")] += 1
            metadata[("validation", case_id)].append(row.get("road_metadata"))
            check(
                f"validation/{candidate_id}/{case_id}",
                lambda row=row, case=case, candidate=candidate: inspect_row(row, case, candidate),
            )
        rows_by_candidate[candidate_id] = rows
    for family in FAMILIES:
        controller = selected[family]
        records = comparison["test_results"][family]
        by_id = {row["case"]["id"]: row for row in records}

        def test_ids(by_id=by_id, records=records) -> None:
            assert set(by_id) == set(test_cases) and len(records) == len(test_cases)

        check(f"test/{family}/case_ids", test_ids)
        for case_id, case in test_cases.items():
            path = directory / "test" / controller["id"] / f"{case_id}.json"
            expected_test_files.add(path)
            try:
                row = read_json(path)
            except (OSError, ValueError) as exc:
                failures.append({"check": f"test/{family}/{case_id}", "error": str(exc)})
                continue
            status_counts["test"][row.get("status", "missing")] += 1
            metadata[("test", case_id)].append(row.get("road_metadata"))

            def test_row(row=row, case=case, controller=controller, by_id=by_id) -> None:
                assert row == by_id[case["id"]], "Case file and final results differ"
                inspect_row(row, case, controller)

            check(f"test/{family}/{case_id}", test_row)
            check(
                f"signals/{family}/{case_id}",
                lambda path=path, row=row: inspect_signals(path.with_suffix(".npz"), row),
            )

    def file_counts() -> None:
        assert set((directory / "validation").glob("*/*.json")) == expected_validation_files
        assert set((directory / "test").glob("*/*.json")) == expected_test_files
        assert len(expected_validation_files) == 308 and len(expected_test_files) == 168
        assert set((directory / "test").glob("*/*.npz")) == {
            path.with_suffix(".npz") for path in expected_test_files
        }

    check("exact_case_and_signal_file_counts", file_counts)

    def paired_metadata() -> None:
        for (split, case_id), items in metadata.items():
            assert len(items) == (11 if split == "validation" else 3)
            assert all(item is not None and item == items[0] for item in items), (
                f"Unpaired road realization: {split}/{case_id}"
            )

    check("realized_road_metadata_paired_for_all_controllers", paired_metadata)
    recomputed = {}

    def scores_and_selection() -> None:
        for candidate_id, rows in rows_by_candidate.items():
            assert len(rows) == 28, f"Missing validation rows for {candidate_id}"
            recomputed[candidate_id] = independent_scores(rows)
        close(frozen["validation_scores"], recomputed, "validation scores")
        for family in FAMILIES:
            choices = [item for item in candidates.values() if item["baseline"] == family]
            winner = min(
                choices,
                key=lambda item: (
                    recomputed[item["id"]]["failed_cases"],
                    recomputed[item["id"]]["score"],
                    item["id"],
                ),
            )
            assert selected[family]["id"] == winner["id"], f"Nonminimum {family} candidate selected"
            assert selected[family]["parameters"] == winner["parameters"]
            close(selected[family]["validation"], recomputed[winner["id"]], f"selected {family}")

    check("independent_validation_scores_and_family_minima", scores_and_selection)
    package_versions = {}
    for name in ("numpy", "scipy", "gymnasium", "pychrono", "stable-baselines3", "torch"):
        try:
            package_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[name] = "not exposed through installed Python package metadata"
    return {
        "schema": "chrono-baseline-comparison-audit-v1",
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "audit_runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "package_versions": package_versions,
        },
        "all_checks_passed": not failures,
        "checks_executed": checks,
        "failures": failures,
        "status_counts": {split: dict(counts) for split, counts in status_counts.items()},
        "expected_case_counts": {"validation": 308, "test": 168},
        "expected_test_signal_files": 168,
        "per_wheel_order": WHEEL_ORDER,
        "independently_recomputed_validation_scores": recomputed,
        "selected_candidate_ids": {family: selected[family]["id"] for family in FAMILIES},
        "plan_sha256": plan["plan_sha256"],
        "freeze_sha256": frozen["freeze_sha256"],
        "note": "Failed simulation cases are counted, not hidden or automatically treated as audit failures. This audit verifies artifacts and selection arithmetic; it does not establish physical agreement with CarSim or electrical energy use.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path("results/chrono/comparison/dclass_seven_20260925"),
    )
    args = parser.parse_args()
    try:
        report = audit(args.directory)
    except FileNotFoundError as exc:
        parser.exit(2, f"{exc}\n")
    output = args.directory / "audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "all_checks_passed": report["all_checks_passed"],
                "checks_executed": report["checks_executed"],
                "status_counts": report["status_counts"],
                "failures": report["failures"],
            },
            ensure_ascii=False,
        )
    )
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
