"""Check substep integration analytically, without pytest or a training run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chrono_suspension.config import load_config  # noqa: E402
from chrono_suspension.env import ChronoSuspensionEnv  # noqa: E402
from chrono_suspension.experiments import _case_metrics  # noqa: E402


class SignalModel:
    """Prescribed finite physics signals, independent of the metric reducer."""

    def __init__(self, signals: dict[str, np.ndarray], dt: float, static_load: np.ndarray):
        self.signals = signals
        self.dt = dt
        self.static_load = static_load
        self.time = 0.0
        self.index = -1
        self.wheel_z0 = np.full(4, 0.3)
        self.suspensions = []
        self.delay_steps = 0
        self.actuator_force = np.zeros(4)

    def step(self, _force: np.ndarray) -> None:
        self.index += 1
        self.time += self.dt
        self.actuator_force = self.signals["actuator_force_n"][self.index]

    def metrics(self) -> dict:
        return {key: values[self.index] for key, values in self.signals.items()}

    def coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(7), np.zeros(7)

    def close(self) -> None:
        pass


def check_analytic_signals() -> dict:
    dt = 0.001
    wheel_scale = np.array([1.0, 2.0, 3.0, 4.0])
    static_load = np.array([100.0, 200.0, 300.0, 400.0])
    ratios = np.array(
        [
            [1, 1, 1, 1],
            [0, 0.5, 1.5, 2],
            [0.5, 0, 2, 1.5],
            [1.5, 1.5, 0, 0.5],
            [2, 2, 0.5, 0],
            [1, 1, 1, 1],
            [0.25, 0.75, 1.25, 1.75],
            [1.75, 1.25, 0.75, 0.25],
        ]
    )
    # Zero net work but nonzero positive and absorbed work catches cancellation.
    power = np.array([10, -10, 20, -20, 30, -30, 40, -40])[:, None] * wheel_scale
    travel = np.array([-1, 2, -3, 4, -5, 6, -7, 8])[:, None] * wheel_scale * 0.001
    signals = {
        "body_acc_mps2": np.array([0, 2, -4, 1, 3, -2, 5, -1], dtype=float),
        "roll_rad": np.arange(-4, 4, dtype=float) * 0.01,
        "pitch_rad": np.arange(8, 0, -1, dtype=float) * -0.02,
        "suspension_travel_m": travel,
        "suspension_velocity_mps": power / 100.0,
        "tire_load_n": ratios * static_load,
        "actuator_force_n": np.full((8, 4), 100.0),
        "mechanical_power_w": power,
    }
    config = load_config({"actuator": {"delay_s": 0.0}})
    env = ChronoSuspensionEnv(config)
    env.model = SignalModel(signals, dt, static_load)
    env.scenario = {"kind": "analytic_fixture"}
    env.dt = dt
    samples, infos = [], []
    try:
        # Unequal integration windows also exercise early-termination weighting.
        for substep_count in (3, 5):
            env.substeps = substep_count
            _, _, terminated, _, info = env.step(np.zeros(4))
            assert not terminated
            samples.append(info["metrics"])
            infos.append(info)
        result = _case_metrics(samples, 0.01, infos)
    finally:
        env.close()
    expectations = {
        "body_acc_rms_mps2": np.sqrt(60 / 8),
        "body_acc_peak_mps2": 5.0,
        "roll_rms_rad": np.sqrt(np.mean(signals["roll_rad"] ** 2)),
        "pitch_rms_rad": np.sqrt(np.mean(signals["pitch_rad"] ** 2)),
        "roll_peak_rad": 0.04,
        "pitch_peak_rad": 0.16,
        "suspension_travel_rms_m": np.sqrt(np.mean(travel**2)),
        "suspension_travel_rms_per_wheel_m": np.sqrt(25.5) * 0.001 * wheel_scale,
        "suspension_travel_peak_per_wheel_m": 0.008 * wheel_scale,
        "suspension_travel_min_per_wheel_m": -0.007 * wheel_scale,
        "suspension_travel_max_per_wheel_m": 0.008 * wheel_scale,
        "tire_load_min_n": 0.0,
        "tire_load_min_per_wheel_n": np.zeros(4),
        "tire_load_min_ratio_per_wheel": np.zeros(4),
        "tire_load_std_n_per_wheel": np.std(ratios * static_load, axis=0),
        "tire_dynamic_load_rms_ratio_per_wheel": np.sqrt(np.mean((ratios - 1) ** 2, axis=0)),
        "wheel_unloaded_fraction": 1 / 8,
        "wheel_unloaded_fraction_per_wheel": np.full(4, 1 / 8),
        "actuator_force_rms_n": 100.0,
        "positive_mechanical_work_per_wheel_j": 0.1 * wheel_scale,
        "negative_mechanical_work_magnitude_per_wheel_j": 0.1 * wheel_scale,
        "positive_mechanical_work_j": 1.0,
        "negative_mechanical_work_magnitude_j": 1.0,
        "absolute_mechanical_work_j": 2.0,
        "net_mechanical_work_j": 0.0,
    }
    for name, expected in expectations.items():
        np.testing.assert_allclose(result[name], expected, rtol=1e-12, atol=1e-12, err_msg=name)

    # Old records still accept their original statistics without requiring the
    # new moment/extremum keys. Fully control-rate records remain readable too.
    old_keys = {
        "body_acc_squared_mean",
        "roll_squared_mean",
        "pitch_squared_mean",
        "actuator_force_squared_mean",
        "mechanical_work_positive_j",
        "mechanical_work_negative_j",
        "tire_load_min_n",
        "duration_s",
    }
    old_infos = [
        {**info, "substep_metrics": {key: info["substep_metrics"][key] for key in old_keys}}
        for info in infos
    ]
    old_result = _case_metrics(samples, 0.01, old_infos)
    np.testing.assert_allclose(old_result["body_acc_rms_mps2"], np.sqrt(60 / 8))
    np.testing.assert_allclose(old_result["absolute_mechanical_work_j"], 2.0)
    control_result = _case_metrics(samples, 0.01)
    assert "tire_dynamic_load_rms_ratio_per_wheel" not in control_result
    np.testing.assert_allclose(
        control_result["body_acc_rms_mps2"], np.sqrt(((-4) ** 2 + (-1) ** 2) / 2)
    )
    return {
        "status": "passed",
        "physics_samples": 8,
        "window_durations_s": [0.003, 0.005],
        "analytic_metric_checks": len(expectations),
        "legacy_fallback_checked": True,
        "wheel_order": ["FL", "FR", "RL", "RR"],
    }


def check_chrono(config_path: str | None) -> dict:
    config = load_config(config_path)
    env = ChronoSuspensionEnv(config)
    samples, infos = [], []
    try:
        env.reset(seed=7823, options={"scenario": {"kind": "iso_b", "speed_mps": 15.0}})
        for _ in range(100):
            _, reward, terminated, truncated, info = env.step(env.baseline_action("skyhook"))
            assert np.isfinite(reward) and not terminated
            samples.append(info["metrics"])
            infos.append(info)
            if truncated:
                break
        result = _case_metrics(samples, config["physics"]["control_dt"], infos)
    finally:
        env.close()
    assert result["body_acc_peak_mps2"] >= result["body_acc_rms_mps2"]
    assert result["positive_mechanical_work_j"] > 0
    assert result["negative_mechanical_work_magnitude_j"] > 0
    assert result["absolute_mechanical_work_j"] >= abs(result["net_mechanical_work_j"])
    np.testing.assert_allclose(
        result["positive_mechanical_work_j"], sum(result["positive_mechanical_work_per_wheel_j"])
    )
    assert np.all(
        np.asarray(result["suspension_travel_peak_per_wheel_m"])
        >= np.asarray(result["suspension_travel_rms_per_wheel_m"])
    )
    return {"status": "passed", "control_steps": len(samples), "metrics": result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--with-chrono", action="store_true")
    parser.add_argument("--output", default="results/chrono/check_metrics.json")
    args = parser.parse_args()
    report = {"analytic": check_analytic_signals()}
    if args.with_chrono:
        report["chrono"] = check_chrono(args.config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "checks": list(report)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
