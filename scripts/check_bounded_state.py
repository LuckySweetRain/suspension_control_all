"""Numerical and physical counterexample checks for inverse-tanh observations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import load_config  # noqa: E402
from rl.bounded_state import inverse_tanh_state  # noqa: E402
from rl.cosim_observation import SensorObservation  # noqa: E402
from rl.cosim_reward import reward_terms, validate_reward  # noqa: E402
from scripts.summarize_cosim_ppo import summarize  # noqa: E402


def main():
    values = np.linspace(-0.079, 0.079, 101)
    original = values.copy()
    mapped, diag = inverse_tanh_state(values, -0.08, 0.08)
    np.testing.assert_allclose(0.08 * np.tanh(mapped), values, atol=1e-15)
    np.testing.assert_array_equal(values, original)
    assert np.all(np.diff(mapped) > 0) and not any(diag.values())
    edge, diag = inverse_tanh_state([-0.1, -0.08, 0, 0.08, 0.1], -0.08, 0.08)
    assert np.isfinite(edge).all()
    assert diag == dict(outside_bounds=2, at_or_outside_bounds=4, numerical_saturations=4)
    for args in (([np.nan], -1, 1), ([0], 1, 1), ([0], -1, 1, 0)):
        try:
            inverse_tanh_state(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid mapping input accepted")
    # x_next=x+u leaves the safe set despite perfectly finite current coordinates.
    current, _ = inverse_tanh_state([0.9], -1, 1)
    _, violation = inverse_tanh_state([np.tanh(current[0]) + 0.2], -1, 1)
    assert violation["outside_bounds"] == 1
    configs = [
        load_config(ROOT / f"configs/{name}.yaml")
        for name in (
            "cosim_ppo_measured",
            "cosim_ppo_mapping_keep",
            "cosim_ppo_plain_no_state",
            "cosim_ppo_mapping_no_state",
        )
    ]
    observers = [SensorObservation(c["cosim_observation"]) for c in configs]
    for i in range(20):
        y = np.zeros(77)
        y[44:48] = i * 5  # Deliberately drive beyond 80 mm in a synthetic observation.
        obs = [o(dict(time=i * 0.01, exports=y), [0, 0]) for o in observers]
        for o in obs:
            assert o.shape == (100,) and np.isfinite(o).all()
        np.testing.assert_array_equal(obs[0], obs[2])
        np.testing.assert_array_equal(obs[1], obs[3])
        mask = np.ones(100, dtype=bool)
        for j in range(8):
            mask[j * 12 + 2 : j * 12 + 6] = False
        np.testing.assert_array_equal(obs[0][mask], obs[1][mask])
    assert observers[1].mapping_diagnostics["outside_bounds"] == 4
    origin = np.zeros(77)
    origin[52:56] = 3000
    y[40], y[43], y[52:56] = 0.2, 0.1, 100
    penalties = []
    for c in configs:
        validate_reward(c["cosim_reward"])
        penalties.append(
            reward_terms(
                y, origin, np.ones(2) * 200, np.ones(2) * 100, np.zeros(2), False, c["cosim_reward"]
            )
        )
    assert penalties[0] == penalties[1] and penalties[2] == penalties[3]
    for key in ("travel", "tire", "violation", "unload"):
        assert penalties[0][key] > 0 and penalties[2][key] == 0
    for key in ("accel", "pitch", "effort", "slew", "projection"):
        assert penalties[0][key] == penalties[2][key]
    fixture = ROOT / "results/state_mapping_checks/summary_case"
    episode = fixture / "episode_0000"
    episode.mkdir(parents=True, exist_ok=True)
    row = dict(
        exports=np.zeros(82).tolist(),
        executed_action=[0, 0],
        reward=0,
        time=0.01,
        observation=np.ones(100).tolist(),
        plan=None,
        physical_constraints=dict(
            max_relative_travel_m=0.09, travel_violation_fraction=0.25, low_load_fraction=0.5
        ),
    )
    (episode / "summary.json").write_text("{}", encoding="utf-8")
    (episode / "transitions.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    result = summarize(fixture)["episodes"][0]
    assert result["max_relative_suspension_travel_m"] == 0.09
    assert result["relative_travel_over_80mm_fraction"] == 0.25
    assert result["qwen_context_valid_steps"] == 0
    assert result["constraint_metric_source"] == "native_unmapped_state"
    print(
        "PASS: roundtrip, boundaries, no mutation, physical counterexample, paired observations/rewards"
    )


if __name__ == "__main__":
    main()
