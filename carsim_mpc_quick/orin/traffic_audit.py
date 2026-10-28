"""Observe signal trigger associations and junction traversal without changing lights."""

from __future__ import annotations

import math


class TrafficAudit:
    def __init__(self, world):
        self.lights = [
            (a, a.get_location()) for a in world.get_actors().filter("traffic.traffic_light")
        ]
        self.samples, self.events = [], []
        self.previous_junction = None
        self.last_second = -1
        self.crossed = 0
        self.started = False
        self.entered_from_outside = False

    def observe(self, vehicle, lane, now):
        junction = lane.junction_id if lane.is_junction else None
        if not self.started:
            self.started = True
            self.previous_junction = junction
            if junction is not None:
                self.events.append(dict(time_s=now, event="spawn_inside", junction_id=junction))
        if junction != self.previous_junction:
            if self.previous_junction is not None:
                self.events.append(
                    dict(
                        time_s=now,
                        event="exit",
                        junction_id=self.previous_junction,
                        complete=self.entered_from_outside,
                    )
                )
                self.crossed += int(self.entered_from_outside)
                self.entered_from_outside = False
            if junction is not None:
                self.events.append(dict(time_s=now, event="enter", junction_id=junction))
                self.entered_from_outside = True
            self.previous_junction = junction
        light = vehicle.get_traffic_light() if vehicle.is_at_traffic_light() else None
        snapshot = dict(
            time_s=now,
            junction_id=junction,
            completed_junctions=self.crossed,
            associated_light_id=light.id if light is not None else None,
            associated_light_state=str(light.get_state()) if light is not None else None,
        )
        if int(now) > self.last_second:
            self.last_second = int(now)
            position = vehicle.get_location()
            nearby = []
            for actor, location in self.lights:
                distance = math.hypot(position.x - location.x, position.y - location.y)
                if distance < 60:
                    nearby.append(
                        dict(
                            id=actor.id,
                            distance_m=distance,
                            state=str(actor.get_state()),
                            frozen=actor.is_frozen(),
                        )
                    )
            self.samples.append(dict(snapshot, nearby_signals=nearby))
        return snapshot
