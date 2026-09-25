"""Frozen road designs for validation-only tuning and held-out comparisons.

The deterministic roads vary physical amplitude, spatial length and speed.
Road seeds only add diversity to the stochastic B/C and composite backgrounds.
All dimensions are SI. These engineering scenarios are not measured road data.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import product

from .roads import seven_road_scenarios

DESIGN_VERSION = "carsim-seven-road-comparison-v1"
ROAD_NAMES = ("iso_b", "iso_c", "sine", "step", "pothole", "bump", "composite")
DETERMINISTIC_NAMES = ("sine", "step", "pothole", "bump")

_LEVELS = {
    "validation": {
        "height_m": (0.015, 0.025),
        "speed_mps": (12.0, 18.0),
        "wavelength_m": (5.5, 8.0),
        "transition_length_m": (0.45, 0.75),
        "length_m": (0.9, 1.3),
    },
    "test": {
        "height_m": (0.020, 0.030),
        "speed_mps": (10.0, 20.0),
        "wavelength_m": (5.0, 9.0),
        "transition_length_m": (0.5, 0.9),
        "length_m": (1.0, 1.5),
    },
}
_LENGTH_PARAMETER = {
    "sine": "wavelength_m",
    "step": "transition_length_m",
    "pothole": "length_m",
    "bump": "length_m",
    "composite": "length_m",
}
_VALIDATION_FACTORS = ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0))
_TEST_FACTORS = tuple(product(range(2), repeat=3))
_TRACKS = {
    "validation": ("both", "left", "right", "both"),
    "test": ("both", "left", "right", "both", "right", "both", "both", "left"),
}
_STOCHASTIC_SEED_BASE = {
    "validation": {"iso_b": 21001, "iso_c": 22001, "composite": 23001},
    "test": {"iso_b": 11001, "iso_c": 12001, "composite": 13001},
}


def comparison_cases(split: str = "test") -> list[dict]:
    """Return fresh, independent dictionaries accepted by ``env.reset``.

    Validation has 28 cases: four per road. Test has 56: eight per road.
    For deterministic and composite roads, test covers the full 2 x 2 x 2
    amplitude/length/speed design. Validation uses a balanced half-factorial
    at entirely different factor values. Side selection is fixed in advance:
    half the cases excite both tracks; the remainder split equally left/right.
    Composite background always excites both tracks; only its local events
    use the assigned side, with pothole opposite to bump for single-side cases.
    """
    if split not in _LEVELS:
        raise ValueError("split must be validation or test")
    levels = _LEVELS[split]
    factors = _VALIDATION_FACTORS if split == "validation" else _TEST_FACTORS
    presets = seven_road_scenarios()
    cases = []
    for road_index, name in enumerate(ROAD_NAMES):
        if name in {"iso_b", "iso_c"}:
            seed_count = 2 if split == "validation" else 4
            for seed_index, speed in product(range(seed_count), levels["speed_mps"]):
                seed = _STOCHASTIC_SEED_BASE[split][name] + seed_index
                cases.append(
                    {
                        **deepcopy(presets[name]),
                        "id": f"{split}_{name}_seed{seed}_v{speed:g}",
                        "name": name,
                        "seed": seed,
                        "speed_mps": speed,
                    }
                )
            continue
        for index, (amplitude_index, length_index, speed_index) in enumerate(factors):
            length_parameter = _LENGTH_PARAMETER[name]
            height = levels["height_m"][amplitude_index]
            length = levels[length_parameter][length_index]
            speed = levels["speed_mps"][speed_index]
            tracks = _TRACKS[split][index]
            seed = (51000 if split == "validation" else 41000) + road_index * 100 + index
            case = {
                **deepcopy(presets[name]),
                "id": f"{split}_{name}_a{amplitude_index}_l{length_index}_v{speed_index}",
                "name": name,
                "seed": seed,
                "height_m": height,
                length_parameter: length,
                "speed_mps": speed,
                "start_m": 12.0,
                "tracks": tracks,
            }
            if name == "composite":
                background = deepcopy(presets["iso_b"])
                # Children inherit the common run-in envelope from their parent.
                background.pop("run_in_m")
                case.update(
                    seed=_STOCHASTIC_SEED_BASE[split][name] + index // 2,
                    tracks="both",
                    local_tracks=tracks,
                    components=[
                        background,
                        {
                            "kind": "bump",
                            "height_m": height,
                            "length_m": length,
                            "start_m": 24.0,
                            "tracks": tracks,
                        },
                        {
                            "kind": "pothole",
                            "height_m": height,
                            "length_m": length,
                            "start_m": 55.0,
                            "tracks": {"both": "both", "left": "right", "right": "left"}[tracks],
                        },
                    ],
                )
            cases.append(case)
    return cases


def design_metadata() -> dict:
    """Describe the predeclared matrix without inspecting simulation outcomes."""
    return {
        "version": DESIGN_VERSION,
        "units": "SI: height/length in m, speed in m/s",
        "road_names": list(ROAD_NAMES),
        "episode_seconds": 10.0,
        "expected_counts": {"validation": 28, "test": 56},
        "factor_levels": deepcopy(_LEVELS),
        "length_parameter_by_road": deepcopy(_LENGTH_PARAMETER),
        "factor_index_order": ["height", "length", "speed"],
        "validation_factor_indices": deepcopy(_VALIDATION_FACTORS),
        "test_factor_indices": deepcopy(_TEST_FACTORS),
        "tracks_by_factor_row": deepcopy(_TRACKS),
        "stochastic_seed_bases": deepcopy(_STOCHASTIC_SEED_BASE),
        "local_event_stations_m": {"step_bump_pothole": 12.0, "composite": [24.0, 55.0]},
        "notes": [
            "Freeze controller gains using validation only, before opening test outcomes.",
            "Test and validation amplitude, length, speed levels and random seeds are disjoint.",
            "B/C use 2 validation or 4 test realizations, each repeated at the two split speeds.",
            "Deterministic profiles do not gain diversity by changing their RNG seed.",
            "The test matrix fully crosses amplitude, length and speed, not track side.",
            "Track side is assigned in advance, with equal single-side and both-side counts.",
            "Composite local events vary in amplitude/length/speed over a B-class background.",
            "Composite single-side cases place bump and pothole on opposite sides.",
            "A step is a finite quintic transition to a persistent new height.",
            "The slowest case passes all local events with its rear axle before 6 seconds.",
        ],
    }
