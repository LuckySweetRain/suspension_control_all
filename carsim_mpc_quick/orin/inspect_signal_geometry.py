"""Read-only map stop-line estimates for recorded traffic-light evaluation.

Uses the trigger-volume lane sampling / last non-junction waypoint approach in
CARLA ScenarioRunner RunningRedLightTest. These are map geometry estimates,
not a detector of the painted road marking or the official infraction metric.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from runtime.paths import load_carla


def inspect():
    carla = load_carla()
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(30)
    world = client.get_world()
    # A new CARLA client can have an empty actor snapshot until its first
    # streamed frame arrives. Observe an existing tick; never advance the world.
    world.wait_for_tick(20)
    road_map = world.get_map()
    result = dict(
        map=road_map.name, scope="read_only_trigger_volume_stop_line_estimates", lights=[]
    )
    for actor in world.get_actors().filter("traffic.traffic_light"):
        transform = actor.get_transform()
        box = actor.trigger_volume
        center = transform.transform(carla.Location(box.location.x, box.location.y, box.location.z))
        angle = math.radians(transform.rotation.yaw)
        seeds, keys = [], set()
        for offset in np.arange(-0.9 * box.extent.x, 0.9 * box.extent.x, 1.0):
            point = carla.Location(
                x=center.x + offset * math.cos(angle),
                y=center.y + offset * math.sin(angle),
                z=center.z,
            )
            wp = road_map.get_waypoint(point)
            if wp is not None and (wp.road_id, wp.lane_id) not in keys:
                seeds.append(wp)
                keys.add((wp.road_id, wp.lane_id))
        lines = []
        for waypoint in seeds:
            started_inside = waypoint.is_junction
            for _ in range(2000):
                if waypoint.is_junction:
                    break
                future = waypoint.next(0.5)
                if not future or future[0].is_junction:
                    break
                waypoint = future[0]
            pose = waypoint.transform
            forward = pose.get_forward_vector()
            lines.append(
                dict(
                    road_id=waypoint.road_id,
                    lane_id=waypoint.lane_id,
                    center=[pose.location.x, pose.location.y, pose.location.z],
                    forward=[forward.x, forward.y],
                    lane_width_m=waypoint.lane_width,
                    seed_inside_junction=started_inside,
                )
            )
        result["lights"].append(
            dict(
                id=actor.id,
                trigger_center=[center.x, center.y, center.z],
                group_ids=[x.id for x in actor.get_group_traffic_lights()],
                green_s=actor.get_green_time(),
                yellow_s=actor.get_yellow_time(),
                red_s=actor.get_red_time(),
                stop_lines=lines,
            )
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print("Saved", len(result["lights"]), "signal geometries:", args.output)
