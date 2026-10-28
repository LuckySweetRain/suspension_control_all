"""TCP's released agent fusion, without the project's extra trajectory tracker.

Reference: OpenDriveLab/TCP leaderboard/team_code/tcp_agent.py, lines 191-234.
The model's own control_pid is intentionally retained as part of that agent.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class TCPOriginalPolicy:
    def __init__(self):
        self.status = 0
        self.last_steers = deque(maxlen=20)

    def step(self, model, prediction, raw, velocity, target, warmup=False):
        if warmup:
            return dict(
                steer_right_positive=0.0,
                throttle=0.0,
                brake=0.0,
                desired_speed_mps=0.0,
                policy="tcp_official_agent",
                branch="warmup",
            )
        # control_pid flips waypoint/target coordinates in place on CPU tensors.
        # Clone both so exported learned trajectories keep their original frame.
        steer, throttle, brake, metadata = model.control_pid(
            prediction["pred_wp"].clone(), velocity, target.clone()
        )
        if brake < 0.05:
            brake = 0.0
        if throttle > brake:
            brake = 0.0
        trajectory = (float(steer), float(throttle), float(brake))
        learned = tuple(raw[k] for k in ("steer_right_positive", "throttle", "brake"))
        weight = 0.3 if self.status == 0 else 0.7
        action = [weight * a + (1 - weight) * b for a, b in zip(learned, trajectory)]
        steer = float(np.clip(action[0], -1, 1))
        throttle = float(np.clip(action[1], 0, 0.75))
        brake = float(np.clip(action[2], 0, 1))
        if brake > 0.5:
            throttle = 0.0
        branch = "trajectory" if self.status == 0 else "control"
        self.last_steers.append(abs(steer))
        self.status = int(sum(s > 0.10 for s in self.last_steers) > 10)
        return dict(
            steer_right_positive=steer,
            throttle=throttle,
            brake=brake,
            desired_speed_mps=float(metadata["desired_speed"]),
            policy="tcp_official_agent",
            branch=branch,
            trajectory_branch=dict(zip(("steer_right_positive", "throttle", "brake"), trajectory)),
        )
