"""Spatial road profiles, coherent rear axle delays, and analytic velocities.

ISO presets target a finite band of the ISO 8608 spatial displacement PSD.
They are synthetic realizations, not classifications of measured roads.
The original RMS-based ``random`` realization remains available unchanged.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

ROAD_KINDS = (
    "flat",
    "random",
    "bump",
    "left_bump",
    "sine",
    "pothole",
    "iso",
    "iso_b",
    "iso_c",
    "step",
    "composite",
)
ISO_CLASSES = {"B": (32e-6, 64e-6, 128e-6), "C": (128e-6, 256e-6, 512e-6)}


def seven_road_scenarios() -> dict[str, dict]:
    """Explicit engineering presets; dimensions are SI, not old-project replicas."""
    presets = {
        "iso_b": {
            "kind": "iso_b",
            "level": "B",
            "gdn0_m3": 64e-6,
            "n_min_per_m": 0.02,
            "n_max_per_m": 2.0,
            "spectral_modes": 256,
        },
        "iso_c": {
            "kind": "iso_c",
            "level": "C",
            "gdn0_m3": 256e-6,
            "n_min_per_m": 0.02,
            "n_max_per_m": 2.0,
            "spectral_modes": 256,
        },
        "sine": {"kind": "sine", "height_m": 0.03, "wavelength_m": 5.0, "phase_rad": 0.0},
        "step": {
            "kind": "step",
            "height_m": 0.03,
            "start_m": 9.0,
            "transition_length_m": 0.3,
        },
        "pothole": {"kind": "pothole", "height_m": 0.03, "length_m": 0.8, "start_m": 9.0},
        "bump": {"kind": "bump", "height_m": 0.03, "length_m": 0.8, "start_m": 9.0},
        "composite": {
            "kind": "composite",
            "components": [
                {
                    "kind": "iso_b",
                    "level": "B",
                    "gdn0_m3": 64e-6,
                    "n_min_per_m": 0.02,
                    "n_max_per_m": 2.0,
                    "spectral_modes": 256,
                },
                {"kind": "bump", "height_m": 0.03, "length_m": 0.8, "start_m": 30.0},
                {"kind": "pothole", "height_m": 0.03, "length_m": 0.8, "start_m": 75.0},
            ],
        },
    }
    return {
        name: {"tracks": "both", "run_in_m": 3.0, "left_right_correlation": 0.6, **preset}
        for name, preset in presets.items()
    }


class FourWheelRoad:
    def __init__(self, scenario: dict, front_distance: float, rear_distance: float, seed: int):
        self.scenario = deepcopy(scenario)
        self.kind = scenario.get("kind", "random")
        if self.kind not in ROAD_KINDS:
            raise ValueError(f"Unknown road kind {self.kind!r}")
        self.speed = float(scenario.get("speed_mps", 15.0))
        self.wheelbase = float(front_distance + rear_distance)
        if not np.isfinite(self.speed) or self.speed <= 0:
            raise ValueError("Road speed must be finite and positive")
        if not np.isfinite(self.wheelbase) or self.wheelbase <= 0:
            raise ValueError("Wheelbase must be finite and positive")
        self.offsets = np.array([0.0, 0.0, -self.wheelbase, -self.wheelbase])
        self.height = float(scenario.get("height_m", 0.03))
        self.length = float(scenario.get("length_m", 0.8))
        self.start = float(scenario.get("start_m", 9.0))
        self.wavelength = float(scenario.get("wavelength_m", 5.0))
        self.sine_phase = float(scenario.get("phase_rad", 0.0))
        self.run_in = float(scenario.get("run_in_m", 3.0))
        self.transition_length = float(scenario.get("transition_length_m", 0.3))
        dimensions = [
            self.height,
            self.length,
            self.start,
            self.wavelength,
            self.sine_phase,
            self.run_in,
            self.transition_length,
        ]
        if not np.all(np.isfinite(dimensions)):
            raise ValueError("Road dimensions must be finite")
        if self.length <= 0 or self.wavelength <= 0 or self.run_in < 0:
            raise ValueError("Road lengths must be positive; run_in_m can be zero")
        if self.transition_length <= 0:
            raise ValueError("Step transition_length_m must be positive (no ideal discontinuity)")
        self.tracks = scenario.get("tracks", "left" if self.kind == "left_bump" else "both")
        if self.tracks not in {"both", "left", "right"}:
            raise ValueError("tracks must be both, left, or right")
        if self.kind == "left_bump" and self.tracks != "left":
            raise ValueError("left_bump requires tracks=left")
        self.track_mask = np.array([self.tracks != "right", self.tracks != "left"] * 2, dtype=float)
        self.seed = scenario.get("seed", seed)
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, (int, np.integer))
            or self.seed < 0
        ):
            raise ValueError("Road seed must be a nonnegative integer")
        self.seed = int(self.seed)
        self.rho = float(scenario.get("left_right_correlation", 0.6))
        if not 0 <= self.rho <= 1:
            raise ValueError("Left/right correlation must lie in [0,1]")
        self.spectral = None
        self.components: list[FourWheelRoad] = []
        if self.kind == "composite":
            components = scenario.get(
                "components", seven_road_scenarios()["composite"]["components"]
            )
            if not isinstance(components, list) or not components:
                raise ValueError("composite requires a nonempty list of components")
            seeds = np.random.SeedSequence(self.seed).spawn(len(components))
            for part, child_seed in zip(components, seeds, strict=True):
                if not isinstance(part, dict) or part.get("kind") not in ROAD_KINDS:
                    raise ValueError("Each road component must specify a supported kind")
                if part["kind"] == "composite":
                    raise ValueError("Nested composite roads are not supported")
                if "speed_mps" in part and float(part["speed_mps"]) != self.speed:
                    raise ValueError("Composite components must share the vehicle speed")
                if "run_in_m" in part and float(part["run_in_m"]) != self.run_in:
                    raise ValueError("Composite applies one shared run-in envelope")
                child = {
                    "speed_mps": self.speed,
                    "run_in_m": self.run_in,
                    "left_right_correlation": self.rho,
                    **part,
                }
                self.components.append(
                    FourWheelRoad(
                        child,
                        front_distance,
                        rear_distance,
                        int(child_seed.generate_state(1, dtype=np.uint32)[0]),
                    )
                )
            return
        if self.kind not in {"random", "iso", "iso_b", "iso_c"}:
            return
        rng = np.random.default_rng(self.seed)
        if self.kind == "random":
            # Preserve the original frequency grid, amplitude and random draws.
            self.frequencies = np.geomspace(0.02, 2.0, 48)
            amplitude = np.sqrt(np.gradient(self.frequencies)) / self.frequencies
            roughness = float(scenario.get("roughness_m", 0.008))
            if not np.isfinite(roughness) or roughness < 0:
                raise ValueError("Road roughness must be finite and nonnegative")
            amplitude *= roughness / np.sqrt(np.sum(amplitude**2) / 2)
            self.spectral = {
                "convention": "legacy finite-band RMS-scaled n^-2 realization",
                "n_min_per_m": 0.02,
                "n_max_per_m": 2.0,
                "modes": 48,
                "expected_rms_m": roughness,
            }
        else:
            level = str(scenario.get("level", self.kind[-1] if self.kind != "iso" else "B")).upper()
            if level not in ISO_CLASSES:
                raise ValueError("ISO road level must be B or C")
            if self.kind != "iso" and level != self.kind[-1].upper():
                raise ValueError("ISO kind and level disagree")
            lower, default_gdn0, upper = ISO_CLASSES[level]
            gdn0 = float(scenario.get("gdn0_m3", default_gdn0))
            if not np.isfinite(gdn0) or not lower <= gdn0 <= upper:
                raise ValueError(f"gdn0_m3 must lie within ISO class {level}: {lower}..{upper}")
            n_min = float(scenario.get("n_min_per_m", 0.02))
            n_max = float(scenario.get("n_max_per_m", 2.0))
            modes = scenario.get("spectral_modes", 256)
            if not np.isfinite(n_min + n_max) or not 0 < n_min < n_max:
                raise ValueError("ISO spatial frequency bounds must satisfy 0 < min < max")
            if isinstance(modes, bool) or not isinstance(modes, int) or not 8 <= modes <= 4096:
                raise ValueError("spectral_modes must be an integer in [8,4096]")
            edges = np.geomspace(n_min, n_max, modes + 1)
            self.frequencies = np.sqrt(edges[:-1] * edges[1:])
            # For w=2, integrate each one-sided PSD bin exactly. Each cosine
            # contributes A^2/2 to ensemble variance, with no per-seed rescaling.
            bin_variance = gdn0 * 0.1**2 * (1 / edges[:-1] - 1 / edges[1:])
            amplitude = np.sqrt(2 * bin_variance)
            self.spectral = {
                "convention": "one-sided spatial displacement PSD",
                "reference_frequency_per_m": 0.1,
                "gdn0_m3": gdn0,
                "level": level,
                "class_bounds_m3": [lower, upper],
                "exponent": 2.0,
                "n_min_per_m": n_min,
                "n_max_per_m": n_max,
                "modes": modes,
                "expected_rms_m": float(np.sqrt(np.sum(bin_variance))),
                "frequency_unit": "cycles/m",
                "psd_unit": "m^3",
                "amplitude_method": "sqrt(2 * integral_of_PSD_over_bin)",
            }
        self.amplitude = amplitude
        self.phase = rng.uniform(0, 2 * np.pi, (3, len(amplitude)))
        self.mix = np.array(
            [
                [np.sqrt(self.rho), np.sqrt(1 - self.rho), 0],
                [np.sqrt(self.rho), 0, np.sqrt(1 - self.rho)],
            ]
        )
        self.spectral["left_right_target_correlation"] = self.rho

    def _spatial_sample(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return height and dz/dx, before the common run-in envelope."""
        z, slope = np.zeros(4), np.zeros(4)
        if self.kind == "flat":
            return z, slope
        if self.kind == "composite":
            for component in self.components:
                part_z, part_slope = component._spatial_sample(x)
                z += part_z
                slope += part_slope
        elif self.kind in {"bump", "left_bump", "pothole"}:
            phase = (x - self.start) / self.length
            mask = (phase >= 0) & (phase <= 1)
            height = -abs(self.height) if self.kind == "pothole" else self.height
            z[mask] = height * 0.5 * (1 - np.cos(2 * np.pi * phase[mask]))
            slope[mask] = height * np.pi / self.length * np.sin(2 * np.pi * phase[mask])
        elif self.kind == "sine":
            k = 2 * np.pi / self.wavelength
            z = self.height * np.sin(k * x + self.sine_phase)
            slope = self.height * k * np.cos(k * x + self.sine_phase)
        elif self.kind == "step":
            u = np.clip((x - self.start) / self.transition_length, 0, 1)
            z = self.height * u**3 * (10 + u * (-15 + 6 * u))
            slope = self.height * 30 * u**2 * (1 - u) ** 2 / self.transition_length
        else:
            phase = 2 * np.pi * x[:, None, None] * self.frequencies + self.phase
            signals = np.sum(self.amplitude * np.cos(phase), axis=2)
            gradients = np.sum(
                -self.amplitude * (2 * np.pi * self.frequencies) * np.sin(phase), axis=2
            )
            mixing = self.mix[[0, 1, 0, 1]]
            z = np.sum(signals * mixing, axis=1)
            slope = np.sum(gradients * mixing, axis=1)
        return z * self.track_mask, slope * self.track_mask

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Return four heights (m) and time derivatives (m/s), FL/FR/RL/RR."""
        if not np.isfinite(t):
            raise ValueError("Road sample time must be finite")
        x = self.speed * float(t) + self.offsets
        z, slope = self._spatial_sample(x)
        if self.run_in == 0:
            return z, self.speed * slope
        phase_in = np.clip(x / self.run_in, 0, 1)
        envelope = phase_in**2 * (3 - 2 * phase_in)
        envelope_slope = np.where(
            (x > 0) & (x < self.run_in),
            (6 / self.run_in) * phase_in * (1 - phase_in),
            0,
        )
        return z * envelope, self.speed * (slope * envelope + z * envelope_slope)

    def preview(self, t: float, steps: int, dt: float) -> np.ndarray:
        if not isinstance(steps, int) or steps < 0 or not np.isfinite(dt) or dt <= 0:
            raise ValueError("Preview requires nonnegative integer steps and positive dt")
        return np.array([self.sample(t + (i + 1) * dt)[0] for i in range(steps)]).reshape(-1)

    def sample_tracks(self, station: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sample left/right tracks at spatial stations, independent of axle delay.

        Returns height (m) and slope dz/ds (dimensionless), each shaped
        ``station.shape + (2,)``. A scalar station returns two-element arrays.
        This is the common source for Chrono terrain queries and mesh vertices.
        """
        positions = np.asarray(station, dtype=float)
        if not np.all(np.isfinite(positions)):
            raise ValueError("Road stations must be finite")
        shape = (*positions.shape, 2)
        heights = np.empty((positions.size, 2))
        slopes = np.empty_like(heights)
        for index, position in enumerate(positions.reshape(-1)):
            z, velocity = self.sample(float(position) / self.speed)
            heights[index] = z[:2]
            slopes[index] = velocity[:2] / self.speed
        return heights.reshape(shape), slopes.reshape(shape)

    @classmethod
    def from_metadata(
        cls, metadata: dict, front_distance: float, rear_distance: float
    ) -> FourWheelRoad:
        """Reconstruct the recorded realization, including composite child seeds.

        Episode RNG seeds are deliberately not used as substitutes for the
        resolved road seed. Callers must additionally compare recorded wheel
        heights when replaying trajectories produced by another source version.
        """

        def scenario_from_record(record: dict) -> dict:
            if not isinstance(record, dict) or record.get("schema") != "chrono-spatial-road-v2":
                raise ValueError("Unsupported or missing resolved road metadata")
            if not np.isclose(
                float(record["wheelbase_m"]),
                front_distance + rear_distance,
                rtol=0,
                atol=1e-10,
            ):
                raise ValueError("Recorded road wheelbase differs from the vehicle geometry")
            scenario = {
                **deepcopy(record["parameters"]),
                "kind": record["kind"],
                "seed": record["seed"],
                "speed_mps": record["speed_mps"],
                "run_in_m": record["run_in_m"],
                "tracks": record["tracks"],
            }
            spectrum = record.get("spectral")
            if scenario["kind"] in {"random", "iso", "iso_b", "iso_c"}:
                if not isinstance(spectrum, dict):
                    raise ValueError("Random road metadata must contain its spectrum")
                scenario["left_right_correlation"] = spectrum["left_right_target_correlation"]
                if scenario["kind"] == "random":
                    scenario["roughness_m"] = spectrum["expected_rms_m"]
                else:
                    scenario.update(
                        level=spectrum["level"],
                        gdn0_m3=spectrum["gdn0_m3"],
                        n_min_per_m=spectrum["n_min_per_m"],
                        n_max_per_m=spectrum["n_max_per_m"],
                        spectral_modes=spectrum["modes"],
                    )
            if scenario["kind"] == "composite":
                scenario["components"] = [
                    scenario_from_record(child) for child in record["components"]
                ]
            return scenario

        return cls(scenario_from_record(metadata), front_distance, rear_distance, metadata["seed"])

    def metadata(self) -> dict:
        """Resolved road parameters; random seeds allow reconstruction."""
        return {
            "schema": "chrono-spatial-road-v2",
            "kind": self.kind,
            "seed": self.seed,
            "speed_mps": self.speed,
            "wheelbase_m": self.wheelbase,
            "run_in_m": self.run_in,
            "tracks": self.tracks,
            "parameters": {
                "height_m": self.height,
                "length_m": self.length,
                "start_m": self.start,
                "wavelength_m": self.wavelength,
                "phase_rad": self.sine_phase,
                "transition_length_m": self.transition_length,
            },
            "spectral": deepcopy(self.spectral),
            "components": [component.metadata() for component in self.components],
            "notes": [
                "Rear wheel follows the same track after wheelbase/speed seconds.",
                "Correlation and RMS targets are ensemble values before the run-in envelope.",
                "Finite-band synthetic ISO profiles are not measured-road certification.",
                "Step uses a finite-length quintic transition, not a discontinuous jump.",
                "Composite sums heights and slopes, then applies one run-in envelope.",
            ],
        }
