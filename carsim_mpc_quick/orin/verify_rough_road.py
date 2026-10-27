"""Compare recorded depth-derived preview against the independently authored road."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rough_road import elevation


def verify(folder):
    folder = Path(folder)
    report = json.loads((folder / "report.json").read_text())
    spec = json.loads((folder / "road_specification.json").read_text())
    packet = json.loads((folder / "first_vision_packet.json").read_text())
    trace = json.loads((folder / "trace.json").read_text())
    assert report["passed"], report
    inverse = np.linalg.inv(np.asarray(packet["world_to_carsim"]))
    errors = []
    active_steps = 0
    visited = []
    axle_visited = {"front": [], "rear": []}
    for index in range(1, len(trace)):
        row = trace[index]
        if row["reply"].get("preview_warmup", True):
            continue
        active_steps += 1
        previous = trace[index - 1]["values"]
        preview = np.asarray(row["reply"]["image_preview_w"]).reshape(-1, 4)
        plane = np.asarray(row["reply"]["image_reference_plane"])
        speed = abs(previous[76]) / 3.6
        for cols, x_indices, y_indices in [(0, [68, 70], [72, 74]), (2, [69, 71], [73, 75])]:
            x = np.mean([previous[i] for i in x_indices])
            y = np.mean([previous[i] for i in y_indices])
            for step in range(len(preview)):
                query_x = x + speed * 0.01 * step
                world = inverse @ [query_x, y, 0, 1]
                predicted = preview[step, cols] + plane[0] + plane[1] * query_x + plane[2] * y
                expected = elevation(float(world[0]), spec["profile"])
                errors.append(predicted - expected)
                if step == 0:
                    visited.append(float(world[0]))
                    axle_visited["front" if cols == 0 else "rear"].append(float(world[0]))
    if not errors:
        raise AssertionError("No active control steps with observed road preview")
    maximum = float(np.max(np.abs(errors)))
    enabled = spec.get(
        "enabled_features",
        [spec["profile"]] if spec["profile"] != "combined" else ["wave", "bump", "pothole"],
    )
    intervals = {"wave": (25, 50), "bump": (60, 64), "pothole": (85, 89)}
    start = min(intervals[name][0] for name in enabled)
    end = max(intervals[name][1] for name in enabled)
    covered = all(
        min(stations) <= start and max(stations) >= end for stations in axle_visited.values()
    )
    result = dict(
        passed=maximum < 0.003 and covered,
        all_features_traversed_by_both_axles=covered,
        active_steps=active_steps,
        preview_height_samples=len(errors),
        max_height_error_m=maximum,
        mean_absolute_height_error_m=float(np.mean(np.abs(errors))),
        visited_station_min_m=min(visited),
        visited_station_max_m=max(visited),
        analytic_profile_used_for_verification_only=True,
    )
    (folder / "road_preview_check.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    assert result["passed"], result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    verify(parser.parse_args().folder)
