"""Check frame invariance and rear-road spatial retention using a real camera packet."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from controller import Controller
from depth_preview import DepthPreview


def verify(folder, data):
    folder = Path(folder)
    packet = json.loads((folder / "first_vision_packet.json").read_text())
    trace = json.loads((folder / "trace.json").read_text())
    controller = Controller(data)
    cfg = controller.c
    vision = DepthPreview()
    vision.observe(packet)
    initial = np.asarray(trace[0]["values"][38:77], dtype=float)
    random = np.random.default_rng(42)
    worst = 0.0
    old_worst = 0.0
    slope = vision.reference_plane[1]
    speed = initial[38] / 3.6
    cy, fy = controller.d["sys"]["Cy"], controller.d["sys"]["Fy"]
    for k in range(10):
        now = k * 0.01
        measured = initial.copy()
        measured[30:34] += speed * now
        preview = vision.preview(measured, cfg, now)
        w, offset = preview[0][:4], preview[3]
        xf, xr = np.mean(measured[[30, 32]]), np.mean(measured[[31, 33]])
        yf, yr = np.mean(measured[[34, 36]]), np.mean(measured[[35, 37]])
        plane = (
            vision.reference_plane[0] + np.array([[xf, yf], [xr, yr]]) @ vision.reference_plane[1:]
        )
        # Independent reference: road inputs measured relative to the two
        # initialized contact elevations, before subtracting the camera plane.
        raw_w = w + np.array(
            [
                plane[0] - vision.initial_road[0],
                slope * speed,
                plane[1] - vision.initial_road[1],
                slope * speed,
            ]
        )
        state = random.normal(0, 0.01, 8)
        expected = cy @ state + fy @ raw_w
        transformed = cy @ (state - offset) + fy @ w
        worst = max(worst, float(np.max(np.abs(expected - transformed))))
        np.testing.assert_allclose(expected, transformed, atol=1e-10, rtol=0)
        # The former distance-only subtraction must fail this same check.
        dz = slope * speed * now
        old_offset = np.array([dz, slope * speed, 0, 0, dz, slope * speed, dz, slope * speed])
        old_worst = max(
            old_worst, float(np.max(np.abs(expected - (cy @ (state - old_offset) + fy @ w))))
        )
    assert old_worst > 1e-3, "Fixture does not expose the former reference-frame bug"

    # Move the camera 20 m while feeding its depth observations. Old road cells
    # must survive repeated new pixels until the rear axle passes them.
    retained = DepthPreview()
    camera_x = packet["camera_to_carsim"][0][3]
    for index in range(21):
        moving = copy.deepcopy(packet)
        moving["camera_to_carsim"][0][3] += index
        moving["frame"] += index
        moving["sim_time"] = index * 0.1
        retained.observe(moving)
    rear_x = camera_x + 20 - 3.5
    minimum, maximum = np.min(retained.points[:, 0]), np.max(retained.points[:, 0])
    assert minimum <= rear_x <= maximum, "Rear axle lost spatial road history"
    report = dict(
        passed=True,
        output_frame_invariance_max_error=worst,
        former_frame_invariance_max_error=old_worst,
        rear_history_retained=True,
        retained_points=len(retained.points),
        initial_calibration_required=True,
    )
    missing = copy.deepcopy(packet)
    missing.pop("initial_road_heights_m")
    try:
        DepthPreview().observe(missing)
    except ValueError:
        pass
    else:
        raise AssertionError("Missing frame calibration was accepted")
    (folder / "reference_frame_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--data", default="results/orin/quick_mpc_data.mat")
    args = parser.parse_args()
    verify(args.folder, args.data)
