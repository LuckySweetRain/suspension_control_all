"""CARLA scene geometry -> longitudinal compliant contact -> CarSim external force.

This is a bounded frontal penalty-contact prototype, not PhysX impulses or a
general crash/multibody solver. The CarSim vehicle remains the sole dynamics owner.
"""

from __future__ import annotations

import math


class FrontalContact:
    def __init__(self, world, carla, terrain, vehicle):
        self.world, self.carla, self.terrain = world, carla, terrain
        self.front = vehicle.bounding_box.location.x + vehicle.bounding_box.extent.x + 0.05
        self.half_width = vehicle.bounding_box.extent.y * 0.8
        self.stop_x = None
        self.events = []
        self.max_force = 0

    def force(self, values, now):
        x, y = values[77], values[78]
        v = values[76] / 3.6
        pitch = math.radians(values[41])
        front = self.front * math.cos(pitch)
        probe_height = values[38] + 0.25 - self.front * math.sin(pitch)
        travel = max(v, 0) * 0.01
        if self.stop_x is None:
            for lateral in [-self.half_width, 0, self.half_width]:
                start = self.terrain.to_world(x + front, y + lateral, probe_height)
                end = self.terrain.to_world(
                    x + front + max(travel, 0.05), y + lateral, probe_height
                )
                hits = self.world.cast_ray(start, end)
                hits = [
                    h
                    for h in hits
                    if str(h.label)
                    # This runner refuses other vehicles at startup. Any vehicle
                    # ray hit here is therefore the ego mesh, not an obstacle.
                    not in (
                        "Roads",
                        "RoadLines",
                        "Ground",
                        "Terrain",
                        "Sidewalks",
                        "Vehicles",
                        "None",
                    )
                ]
                if hits:
                    if str(hits[0].label).upper() == "NONE":
                        raise RuntimeError(
                            "Unclassified CARLA contact geometry; native collision coupling is required"
                        )
                    distance = min(start.distance(h.location) for h in hits)
                    self.stop_x = x + front + distance
                    self.events.append(
                        dict(time=now, x=self.stop_x, speed_mps=v, label=str(hits[0].label))
                    )
                    break
        penetration = 0 if self.stop_x is None else x + front + travel - self.stop_x
        active = penetration > 0
        # Spring/damper contact at the CarSim aero-force reference; additive FX.
        # Zero restitution is approximated by strong damping and bounded force.
        # Signed damping dissipates rebound energy; cap closing impulse.
        force = (
            -min(max(1e6 * penetration + 8e4 * v, 0), 100000, 1680.2 * max(v, 0) / 0.01)
            if active
            else 0.0
        )
        if not math.isfinite(force):
            raise RuntimeError("Nonfinite contact feedback")
        self.max_force = max(self.max_force, abs(force))
        return force, dict(
            active=self.stop_x is not None,
            contacting=active,
            force_N=force,
            penetration_m=max(penetration, 0),
            model="frontal_penalty",
        )
