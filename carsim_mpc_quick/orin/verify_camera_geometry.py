"""Compare reconstructed camera points with independent CARLA scene ray hits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from carla_stack import CARLA_ROOT
from depth_preview import DepthPreview


def verify(folder):
    sys.path.insert(0, str(CARLA_ROOT / "PythonAPI/carla/dist/carla-0.9.11-py3.7-win-amd64.egg"))
    import carla

    folder = Path(folder)
    packet = json.loads((folder / "first_vision_packet.json").read_text())
    vision = DepthPreview()
    vision.observe(packet)
    camera = np.asarray(packet["camera_to_carsim"])[:3, 3]
    points = vision.points[np.abs(vision.points[:, 1] - camera[1]) < 1.2]
    points = points[(points[:, 0] > camera[0] + 0.2) & (points[:, 0] < camera[0] + 7)]
    if len(points) < 20:
        raise RuntimeError("Insufficient road points for geometry verification")
    selected = points[np.linspace(0, len(points) - 1, 20).astype(int)]
    inverse = np.linalg.inv(np.array(packet["world_to_carsim"]))
    world_points = np.column_stack((selected, np.ones(len(selected)))) @ inverse.T
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(20)
    world = client.get_world()
    errors = []
    signed_errors = []
    for x, y, z, _ in world_points:
        hits = world.cast_ray(carla.Location(x=x, y=y, z=z + 1), carla.Location(x=x, y=y, z=z - 2))
        procedural = packet.get("road_mask_mode") == "opendrive_test_road"
        heights = [
            h.location.z
            for h in hits
            if str(h.label) in ("Roads", "RoadLines")
            or (procedural and str(h.label).upper() == "NONE")
        ]
        if not heights:
            raise RuntimeError("No corresponding CARLA road geometry")
        signed = z - min(heights, key=lambda height: abs(z - height))
        signed_errors.append(signed)
        errors.append(abs(signed))
    report = dict(
        samples=len(errors),
        max_height_error_m=max(errors),
        mean_height_error_m=float(np.mean(errors)),
        mean_signed_height_error_m=float(np.mean(signed_errors)),
    )
    # A centimetre tolerance previously hid the missing half-pixel centre offset.
    assert report["max_height_error_m"] < 0.002, report
    (folder / "camera_geometry_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    args = parser.parse_args()
    verify(args.folder)
