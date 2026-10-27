"""Replay real camera evidence and verify preview dependence and stale-frame rejection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from controller import Controller
from depth_preview import DepthPreview


def verify(folder, data):
    folder = Path(folder)
    packet = json.loads((folder / "first_vision_packet.json").read_text())
    trace = json.loads((folder / "trace.json").read_text())
    y = trace[0]["values"][:77]
    vision = DepthPreview()
    vision.observe(packet)
    controller = Controller(str(data))

    def forbidden_static_preview(*args):
        raise AssertionError("Static CarSim road preview was used")

    controller.preview = forbidden_static_preview
    preview = vision.preview(y, controller.c, 0.01)
    image_forces, _, _ = controller.step(y, preview)
    zero_preview = (np.zeros_like(preview[0]), preview[1], preview[2], preview[3])
    controller.reset()
    zero_forces, _, _ = controller.step(y, zero_preview)
    delta = float(np.max(np.abs(np.array(image_forces) - zero_forces)))
    assert delta > 1e-5, "Camera preview did not affect control output"
    try:
        vision.preview(y, controller.c, packet["sim_time"] + 0.2)
    except ValueError:
        pass
    else:
        raise AssertionError("Stale image was accepted")
    report = dict(
        image_vs_zero_preview_max_force_delta_N=delta,
        image_point_count=vision.valid_points,
        image_reference_plane=vision.reference_plane.tolist(),
        stale_frame_rejected=True,
        static_road_lookup_used=False,
    )
    (folder / "input_dependency_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--data", default="results/orin/quick_mpc_data.mat")
    args = parser.parse_args()
    verify(args.folder, args.data)
