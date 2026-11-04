"""Causal sensor-only ablation; estimates never read simulator velocities/loads."""

from __future__ import annotations

from collections import deque

import numpy as np


class SensorObservation:
    def __init__(self, config):
        self.config = config
        self.history = deque(maxlen=8)
        self.last_time = None
        self.origin = None
        self.travel = None
        self.relative_velocity = np.zeros(4)
        self.body_velocity = 0.0
        self.rng = np.random.default_rng(config["noise_seed"])

    def __call__(self, request, previous):
        now = float(request["time"])
        if self.last_time == now:
            return self.cached.copy()
        y = np.asarray(request["exports"], dtype=float)
        if self.origin is None:
            self.origin = y[44:48].copy()
        dt = 0.01 if self.last_time is None else now - self.last_time
        if not 0 < dt < 0.02:
            raise ValueError("Sensor observation requires monotonic 100 Hz samples")
        # Synthetic sensor noise assumptions, not hardware specifications.
        az = y[40] * 9.80665 + self.rng.normal(0, 0.02)
        pitch_rate = np.deg2rad(y[42]) + self.rng.normal(0, 0.001)
        travel = (y[44:48] - self.origin) / 1000 + self.rng.normal(0, 0.0001, 4)
        if self.travel is not None:
            alpha = dt / (0.05 + dt)
            self.relative_velocity += alpha * ((travel - self.travel) / dt - self.relative_velocity)
        # Leaky integration is a band-limited estimate, not absolute velocity.
        self.body_velocity = np.exp(-dt / 0.5) * self.body_velocity + dt * az
        estimates = np.zeros(3)
        if self.config["mode"] == "estimated":
            estimates = (
                np.array(
                    [
                        self.body_velocity,
                        self.relative_velocity[[0, 2]].mean(),
                        self.relative_velocity[[1, 3]].mean(),
                    ]
                )
                / 0.5
            )
        frame = np.concatenate(
            (
                [az / 5, pitch_rate],
                travel / 0.08,
                np.asarray(previous) / 2000,
                [y[76] / 72],
                estimates,
            )
        )
        if not self.history:
            self.history.extend([frame.copy() for _ in range(8)])
        else:
            self.history.append(frame)
        # Constant placeholder token preserves identical network size between A/B.
        self.cached = np.clip(np.r_[np.concatenate(self.history), np.zeros(4)], -20, 20).astype(
            np.float32
        )
        if not np.isfinite(self.cached).all():
            raise ValueError("Nonfinite sensor observation")
        self.travel, self.last_time = travel, now
        return self.cached.copy()
