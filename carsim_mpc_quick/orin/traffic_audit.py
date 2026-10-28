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
        self.signal_changes = []
        self.signal_states = {}

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
        # Evaluation only: sample every camera frame so a red/green transition
        # can be aligned with the 20 Hz native TCP actions without a 1 s gap.
        if now > self.last_second:
            self.last_second = now
            position = vehicle.get_location()
            transform = vehicle.get_transform()
            nearby = []
            for actor, location in self.lights:
                distance = math.hypot(position.x - location.x, position.y - location.y)
                if distance < 60:
                    state = str(actor.get_state())
                    if self.signal_states.get(actor.id) != state:
                        self.signal_changes.append(dict(time_s=now, id=actor.id, state=state))
                        self.signal_states[actor.id] = state
                    nearby.append(
                        dict(
                            id=actor.id,
                            distance_m=distance,
                            state=state,
                            frozen=actor.is_frozen(),
                            elapsed_s=actor.get_elapsed_time(),
                        )
                    )
            self.samples.append(
                dict(
                    snapshot,
                    nearby_signals=nearby,
                    world_pose=[position.x, position.y, position.z, transform.rotation.yaw],
                    road_id=lane.road_id,
                    lane_id=lane.lane_id,
                )
            )
        return snapshot
