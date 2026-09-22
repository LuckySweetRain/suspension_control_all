"""Spatial road inputs with coherent rear axle delay and analytic slopes.

Random road is a finite-band n^-2 spatial PSD realization, not an ISO road
class certification. Its explicit RMS and bandwidth are included in metadata.
"""

from __future__ import annotations

import numpy as np


class FourWheelRoad:
    def __init__(self, scenario: dict, front_distance: float, rear_distance: float, seed: int):
        self.scenario = dict(scenario)
        self.kind = scenario.get("kind", "random")
        if self.kind not in {"flat", "random", "bump", "left_bump", "sine", "pothole"}:
            raise ValueError(f"Unknown road kind {self.kind!r}")
        self.speed = float(scenario.get("speed_mps", 15.0))
        if not np.isfinite(self.speed) or self.speed <= 0:
            raise ValueError("Road speed must be finite and positive")
        wheelbase = front_distance + rear_distance
        self.offsets = np.array([0.0, 0.0, -wheelbase, -wheelbase])
        self.height = float(scenario.get("height_m", 0.03))
        self.length = float(scenario.get("length_m", 0.8))
        self.start = float(scenario.get("start_m", 9.0))
        self.wavelength = float(scenario.get("wavelength_m", 5.0))
        if not np.all(np.isfinite([self.height, self.length, self.start, self.wavelength])):
            raise ValueError("Road dimensions must be finite")
        if self.length <= 0 or self.wavelength <= 0:
            raise ValueError("Road lengths must be positive")
        rng = np.random.default_rng(scenario.get("seed", seed))
        self.frequencies = np.geomspace(0.02, 2.0, 48)
        widths = np.gradient(self.frequencies)
        amplitude = np.sqrt(widths) / self.frequencies
        roughness = float(scenario.get("roughness_m", 0.008))
        if not np.isfinite(roughness) or roughness < 0:
            raise ValueError("Road roughness must be finite and nonnegative")
        amplitude *= roughness / np.sqrt(np.sum(amplitude**2) / 2)
        rho = float(scenario.get("left_right_correlation", 0.6))
        if not 0 <= rho <= 1:
            raise ValueError("Left/right correlation must lie in [0,1]")
        self.amplitude = amplitude
        self.phase = rng.uniform(0, 2 * np.pi, (3, len(amplitude)))
        self.mix = np.array(
            [[np.sqrt(rho), np.sqrt(1 - rho), 0], [np.sqrt(rho), 0, np.sqrt(1 - rho)]]
        )

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        x = self.speed * float(t) + self.offsets
        z, slope = np.zeros(4), np.zeros(4)
        if self.kind == "flat":
            return z, slope
        if self.kind in {"bump", "left_bump", "pothole"}:
            phase = (x - self.start) / self.length
            mask = (phase >= 0) & (phase <= 1)
            height = -abs(self.height) if self.kind == "pothole" else self.height
            z[mask] = height * 0.5 * (1 - np.cos(2 * np.pi * phase[mask]))
            slope[mask] = height * np.pi / self.length * np.sin(2 * np.pi * phase[mask])
            if self.kind == "left_bump":
                z[[1, 3]], slope[[1, 3]] = 0, 0
        elif self.kind == "sine":
            k = 2 * np.pi / self.wavelength
            z = self.height * np.sin(k * x)
            slope = self.height * k * np.cos(k * x)
        else:
            phase = 2 * np.pi * x[:, None, None] * self.frequencies + self.phase
            signals = np.sum(self.amplitude * np.cos(phase), axis=2)
            gradients = np.sum(
                -self.amplitude * (2 * np.pi * self.frequencies) * np.sin(phase), axis=2
            )
            mixing = self.mix[[0, 1, 0, 1]]
            z = np.sum(signals * mixing, axis=1)
            slope = np.sum(gradients * mixing, axis=1)
        # Smooth spatial run-in prevents an artificial reset impact and keeps
        # rear inputs exact delayed copies of the corresponding front track.
        phase_in = np.clip(x / 3.0, 0, 1)
        envelope = phase_in**2 * (3 - 2 * phase_in)
        envelope_slope = np.where((x > 0) & (x < 3), 2 * phase_in * (1 - phase_in), 0)
        return z * envelope, self.speed * (slope * envelope + z * envelope_slope)

    def preview(self, t: float, steps: int, dt: float) -> np.ndarray:
        return np.array([self.sample(t + (i + 1) * dt)[0] for i in range(steps)]).reshape(-1)
