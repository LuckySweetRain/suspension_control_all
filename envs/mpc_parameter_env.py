from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None
    spaces = None

from config import load_config


GAIN_KEYS = ["k_z", "k_dz", "k_pitch", "k_dpitch", "k_roll", "k_droll", "k_preview"]


class _FallbackBox:
    def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype):
        self.low = np.full(shape, low, dtype=dtype)
        self.high = np.full(shape, high, dtype=dtype)
        self.shape = shape
        self.dtype = dtype

    def sample(self):
        return np.random.uniform(self.low, self.high).astype(self.dtype)


class MPCParameterPolicyEnv(gym.Env if gym is not None else object):
    """One-step environment for learning MPC-lite gain schedules.

    The action is a normalized vector in [-1, 1]. It is mapped to multiplicative
    gain scales around the base `reduced_full_car_mpc` gains. The step reward is
    the negative rollout score, so maximizing return minimizes the MPC-lite
    evaluation objective.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        config_path: str | Path | None = None,
        scenario_limit: int | None = None,
        scale_low: float = 0.5,
        scale_high: float = 1.5,
        controller_name: str = "FULL_CAR_MPC_LITE",
    ):
        if config is None:
            if config_path is None:
                config_path = Path(__file__).resolve().parents[1] / "configs" / "mujoco_full_car_corner.yaml"
            config = load_config(config_path)
        self.config = copy.deepcopy(config)
        self.scenario_limit = scenario_limit
        self.scale_low = float(scale_low)
        self.scale_high = float(scale_high)
        # 可配置调度对象（A2': 半车 MPC 状态权重；默认整车 MPC-lite 增益，向后兼容）
        sched_cfg = self.config.get("mpc_parameter_scheduling", {})
        self.gain_section = sched_cfg.get("gain_section", "reduced_full_car_mpc")
        self.gain_keys = list(sched_cfg.get("gain_keys", GAIN_KEYS))
        if sched_cfg.get("controller_name"):
            controller_name = sched_cfg["controller_name"]
        self.controller_name = controller_name
        self.base_gains = dict(self.config.get(self.gain_section, {}))
        self.gain_dim = len(self.gain_keys)
        self.scenarios = list(self.config.get("scenarios", []))
        if self.scenario_limit is not None:
            self.scenarios = self.scenarios[: int(self.scenario_limit)]
            self.config["scenarios"] = list(self.scenarios)
        self.rng = np.random.default_rng(int(self.config.get("seed", 42)))
        self.active_scenario = self.scenarios[0] if self.scenarios else None
        self.episode_count = 0
        self.last_score: float | None = None
        self.last_gains: dict[str, float] | None = None
        self.last_metrics: dict | None = None
        box_cls = spaces.Box if spaces is not None else _FallbackBox
        self.action_space = box_cls(-1.0, 1.0, shape=(self.gain_dim,), dtype=np.float32)
        self.observation_space = box_cls(-np.inf, np.inf, shape=(self.gain_dim + 10,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            np.random.seed(seed)
            self.rng = np.random.default_rng(seed)
        if options and "scenario" in options:
            self.active_scenario = options["scenario"]
        elif options and "scenario_index" in options:
            self.active_scenario = self.scenarios[int(options["scenario_index"])]
        elif self.scenarios:
            self.active_scenario = self._select_scenario(self.episode_count)
        self.episode_count += 1
        self.last_score = None
        self.last_gains = None
        self.last_metrics = None
        return self._obs(), self._info()

    def step(self, action):
        from scripts.tune_mpc_parameters import _evaluate_gain_config

        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        gains = self.action_to_gains(action)
        trial_config = copy.deepcopy(self.config)
        trial_config[self.gain_section] = {**self.base_gains, **gains}
        if self.active_scenario is not None:
            trial_config["scenarios"] = [self.active_scenario]
        score, scenario_metrics = _evaluate_gain_config(
            trial_config,
            controller_name=self.controller_name,
            scenario_limit=1 if self.active_scenario is not None else self.scenario_limit,
            score_weights=self.config.get("mpc_parameter_tuning", {}).get("score_weights"),
        )
        self.last_score = float(score)
        self.last_gains = gains
        self.last_metrics = scenario_metrics
        reward = -float(score)
        terminated = True
        truncated = False
        return self._obs(), reward, terminated, truncated, self._info()

    def action_to_gains(self, action: np.ndarray) -> dict[str, float]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        scale_mid = 0.5 * (self.scale_low + self.scale_high)
        scale_half = 0.5 * (self.scale_high - self.scale_low)
        scales = scale_mid + scale_half * action
        return {key: float(self.base_gains.get(key, 0.0)) * float(scale) for key, scale in zip(self.gain_keys, scales)}

    def _obs(self) -> np.ndarray:
        base = np.asarray([float(self.base_gains.get(key, 0.0)) for key in self.gain_keys], dtype=np.float64)
        norm = base / np.maximum(np.abs(base), 1.0)
        road = self._scenario_features(self.active_scenario)
        meta = np.asarray(
            [
                float(len(self.config.get("scenarios", []))),
                float(self.scenario_limit or len(self.config.get("scenarios", []))),
                float(self.last_score if self.last_score is not None else 0.0),
            ],
            dtype=np.float64,
        )
        return np.concatenate([norm, road, meta]).astype(np.float32)

    def _scenario_features(self, scenario: dict | None) -> np.ndarray:
        if not scenario:
            return np.zeros(7, dtype=np.float64)
        road_type = str(scenario.get("type", "iso")).lower()
        type_code = {
            "iso": 1.0,
            "bump": 2.0,
            "sine": 3.0,
            "step": 4.0,
            "pothole": 5.0,
            "composite": 6.0,
            "csv": 7.0,
            "file": 7.0,
        }.get(road_type, 0.0)
        level = str(scenario.get("level", "B")).upper()
        level_code = max(0.0, float(ord(level[:1]) - ord("A") + 1)) if level else 0.0
        speed = float(scenario.get("speed", self.config.get("speed", 0.0)))
        amplitude = float(
            scenario.get(
                "amplitude",
                scenario.get("height", scenario.get("bump_height", abs(float(scenario.get("depth", 0.0))))),
            )
        )
        wavelength = float(scenario.get("wavelength", scenario.get("length", scenario.get("bump_length", 0.0))))
        start_time = float(scenario.get("start_time", 0.0))
        component_count = float(len(scenario.get("components", []))) if road_type == "composite" else 0.0
        return np.asarray(
            [
                type_code / 7.0,
                level_code / 6.0,
                speed / 30.0,
                amplitude / 0.1,
                wavelength / 5.0,
                start_time / 5.0,
                component_count / 5.0,
            ],
            dtype=np.float64,
        )

    def _select_scenario(self, episode: int) -> dict:
        sampling = dict(self.config.get("scenario_sampling", {}))
        mode = str(sampling.get("mode", "cycle")).lower()
        if mode == "uniform":
            return self.scenarios[int(self.rng.integers(0, len(self.scenarios)))]
        if mode == "weighted":
            weights_cfg = sampling.get("weights", {})
            weights = np.asarray([float(weights_cfg.get(str(s.get("name", "")), 1.0)) for s in self.scenarios], dtype=np.float64)
            if np.any(weights < 0.0) or not np.any(weights > 0.0):
                weights = np.ones(len(self.scenarios), dtype=np.float64)
            weights = weights / np.sum(weights)
            return self.scenarios[int(self.rng.choice(len(self.scenarios), p=weights))]
        return self.scenarios[episode % len(self.scenarios)]

    def _info(self) -> dict:
        return {
            "last_score": self.last_score,
            "last_gains": self.last_gains,
            "last_metrics": self.last_metrics,
            "active_scenario": self.active_scenario.get("name", "") if self.active_scenario else "",
            "scenario_features": self._scenario_features(self.active_scenario).tolist(),
            "gain_keys": list(self.gain_keys),
            "scale_low": self.scale_low,
            "scale_high": self.scale_high,
        }

    def close(self):
        pass
