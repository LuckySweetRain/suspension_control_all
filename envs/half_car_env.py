from __future__ import annotations

from typing import Any

import numpy as np

from config import half_car_dimensions

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None
    spaces = None

from models.half_car import HalfCarModel, HalfCarParams
from roads.road_profiles import RoadProfileFactory


class HalfCarEnv(gym.Env if gym is not None else object):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: dict[str, Any],
        scenario: dict[str, Any] | None = None,
        use_preview: bool = True,
    ):
        self.config = config
        self.dt = float(config["dt"])
        self.control_dt = float(config["control_dt"])
        self.episode_seconds = float(config["episode_seconds"])
        for name, value in (
            ("dt", self.dt),
            ("control_dt", self.control_dt),
            ("episode_seconds", self.episode_seconds),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        ratio = self.control_dt / self.dt
        if ratio < 1.0 or not np.isclose(ratio, round(ratio), rtol=0.0, atol=1e-6):
            raise ValueError("control_dt must be an integer multiple of dt")
        self.substeps = int(round(ratio))
        self.max_steps = int(round(self.episode_seconds / self.control_dt))
        if self.max_steps < 1:
            raise ValueError("episode_seconds must span at least one control step")
        self.settle_steps = int(round(float(config.get("settle_seconds", 0.0)) / self.control_dt))
        self.wheelbase = float(config["wheelbase"])
        self.force_limit = float(config["force_limit"])
        self.preview_steps = int(config["preview"]["steps"])
        self.preview_dt = float(config["preview"].get("dt", self.control_dt))
        self.road_scale = float(config.get("road_scale", 0.1))
        for name, value in (
            ("wheelbase", self.wheelbase),
            ("force_limit", self.force_limit),
            ("preview.dt", self.preview_dt),
            ("road_scale", self.road_scale),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.preview_steps < 1 or self.preview_steps != config["preview"]["steps"]:
            raise ValueError("preview.steps must be a positive integer")
        if str(config.get("environment", {}).get("engine", "python")).lower() != "python":
            raise ValueError("HalfCarEnv requires environment.engine=python")
        if config.get("preview_error", {}).get("enabled", False):
            raise ValueError(
                "HalfCarEnv does not implement preview_error; corrupted-preview experiments are unsupported"
            )
        self.params = HalfCarParams.from_config(config)
        self.model = HalfCarModel(self.params)
        self.front_static_load, self.rear_static_load = self.params.static_tire_loads()
        safety_cfg = dict(config.get("safety_limits", {}))
        suspension_limit = float(safety_cfg.get("max_suspension_travel", 0.08))
        self.safety_limits = {
            "max_suspension_travel": suspension_limit,
            "max_front_tire_dynamic_load": float(
                safety_cfg.get("max_front_tire_dynamic_load", self.front_static_load)
            ),
            "max_rear_tire_dynamic_load": float(
                safety_cfg.get("max_rear_tire_dynamic_load", self.rear_static_load)
            ),
        }
        norm_cfg = dict(config.get("normalization", {}))
        self.reward_scales = {
            "body_acc": max(float(norm_cfg.get("body_acc", 0.491)), 1e-9),
            "pitch_acc": max(float(norm_cfg.get("pitch_acc", 1.821)), 1e-9),
            "suspension_travel": max(
                float(norm_cfg.get("suspension_travel", suspension_limit)), 1e-9
            ),
            "front_tire_load": max(
                float(norm_cfg.get("front_tire_load", self.front_static_load)), 1e-9
            ),
            "rear_tire_load": max(
                float(norm_cfg.get("rear_tire_load", self.rear_static_load)), 1e-9
            ),
            "control_force": max(float(norm_cfg.get("control_force", self.force_limit)), 1e-9),
        }
        self.use_preview = use_preview
        self.scenario = dict(scenario if scenario is not None else config["scenarios"][0])
        self.speed = float(self.scenario.get("speed", config.get("speed", 20.0)))
        self.scenario.setdefault("speed", self.speed)
        self.road = RoadProfileFactory.create(self.scenario, self.episode_seconds, self.dt)
        self._preview_cache: np.ndarray | None = None
        self.state = np.zeros(8, dtype=np.float64)
        self.t = 0.0
        self.step_count = 0
        self.last_action = np.zeros(2, dtype=np.float64)
        self.last_derived = self.model.derived(self.state, np.zeros(4), self.last_action)

        _, _, self.base_obs_dim = half_car_dimensions(config)
        self.action_in_obs = bool(config.get("action_in_observation", False))
        self.obs_dim = self.base_obs_dim + (self.preview_steps * 2 if use_preview else 0)
        if spaces is not None:
            self.action_space = spaces.Box(
                -self.force_limit, self.force_limit, shape=(2,), dtype=np.float32
            )
            self.observation_space = spaces.Box(
                -np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32
            )
        else:
            # Fallback when gymnasium is not available
            class _Box:
                def __init__(self, low, high, shape, dtype):
                    self.low = np.full(shape, low, dtype=dtype)
                    self.high = np.full(shape, high, dtype=dtype)
                    self.shape = shape
                    self.dtype = dtype

            self.action_space = _Box(-self.force_limit, self.force_limit, (2,), np.float32)
            self.observation_space = _Box(-np.inf, np.inf, (self.obs_dim,), np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if gym is not None:
            super().reset(seed=seed)
        if options and "scenario" in options:
            self.scenario = dict(options["scenario"])
            self.speed = float(self.scenario.get("speed", self.config.get("speed", 20.0)))
            self.scenario.setdefault("speed", self.speed)
            self.road = RoadProfileFactory.create(self.scenario, self.episode_seconds, self.dt)
        self._preview_cache = None
        self.state = np.zeros(8, dtype=np.float64)
        zf, dzf, zr, dzr = self.road.pair(0.0, self.speed, self.wheelbase)
        self.state[4] = zf
        self.state[6] = zr
        self.t = 0.0
        self.step_count = 0
        self.last_action = np.zeros(2, dtype=np.float64)
        self.last_derived = self.model.derived(
            self.state, np.asarray([zf, dzf, zr, dzr]), self.last_action
        )
        self.last_safety = self._safety()
        return self._obs(), self._info(0.0, unsafe=self.last_safety["unsafe"])

    def step(self, action):
        # Residual policies may also supply the prior action for diagnostics.
        if isinstance(action, dict):
            action = action["action"]
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size != 2 or not np.all(np.isfinite(action)):
            raise ValueError("HalfCarEnv action must contain two finite actuator forces")
        action = self.model.clip_action(action)
        total_reward = 0.0
        for _ in range(self.substeps):
            road = np.asarray(self.road.pair(self.t, self.speed, self.wheelbase), dtype=np.float64)
            self.state = self.model.rk4_step(self.state, action, road, self.dt)
            self.t += self.dt
        road = np.asarray(self.road.pair(self.t, self.speed, self.wheelbase), dtype=np.float64)
        self.last_action = action
        self.last_derived = self.model.derived(self.state, road, action)
        reward = self._reward(action)
        total_reward += reward
        self.step_count += 1
        finite = bool(np.all(np.isfinite(self.state)) and np.all(np.isfinite(action)))
        self.last_safety = self._safety(finite)
        self._preview_cache = None
        unsafe = bool(self.last_safety["unsafe"])
        terminated = not finite
        truncated = self.step_count >= self.max_steps
        if unsafe and self.step_count > self.settle_steps:
            total_reward -= 10.0
        return (
            self._obs(),
            float(total_reward),
            terminated,
            truncated,
            self._info(total_reward, unsafe=unsafe),
        )

    def _base_obs(self) -> np.ndarray:
        d = self.last_derived
        base = np.asarray(
            [
                self.state[0],
                self.state[1],
                self.state[2],
                self.state[3],
                self.state[4],
                self.state[5],
                self.state[6],
                self.state[7],
                d["delta_yf"],
                d["delta_yr"],
                d["delta_dyf"],
                d["delta_dyr"],
                d["ddzb"],
                d["ddtheta"],
            ],
            dtype=np.float64,
        )
        if self.action_in_obs:
            return np.concatenate([base, self.last_action.astype(np.float64)])
        return base

    def _obs(self) -> np.ndarray:
        base = self._base_obs()
        if not self.use_preview:
            return base.astype(np.float32)
        preview = self._road_preview()
        return np.concatenate([base, preview.reshape(-1) / self.road_scale]).astype(np.float32)

    def _road_preview(self) -> np.ndarray:
        if self._preview_cache is None:
            self._preview_cache = self.road.preview(
                self.t, self.speed, self.wheelbase, self.preview_steps, self.preview_dt
            )
        return self._preview_cache

    def _reward(self, action: np.ndarray) -> float:
        w = self.config["reward"]
        d = self.last_derived
        s = self.reward_scales
        body = (d["ddzb"] / s["body_acc"]) ** 2
        pitch = (d["ddtheta"] / s["pitch_acc"]) ** 2
        susp = (d["delta_yf"] / s["suspension_travel"]) ** 2 + (
            d["delta_yr"] / s["suspension_travel"]
        ) ** 2
        tire = (d["Fpf"] / s["front_tire_load"]) ** 2 + (d["Fpr"] / s["rear_tire_load"]) ** 2
        control = float(np.sum(np.square(action / s["control_force"])))
        raw = -float(
            w["body_acc"] * body
            + w["pitch_acc"] * pitch
            + w["suspension"] * susp
            + w["tire_load"] * tire
            + w["control"] * control
        )
        clip_cfg = w.get("clip", None)
        if clip_cfg is not None:
            return float(
                np.clip(raw, float(clip_cfg.get("min", -200.0)), float(clip_cfg.get("max", 0.0)))
            )
        return raw

    def _safety(self, finite: bool = True) -> dict[str, Any]:
        d = self.last_derived
        s = self.state
        max_susp = float(max(abs(d["delta_yf"]), abs(d["delta_yr"])))
        abs_pitch = float(abs(s[2]))
        max_wheel = float(max(abs(s[4]), abs(s[6])))
        front_tire = float(abs(d["Fpf"]))
        rear_tire = float(abs(d["Fpr"]))
        lim = self.safety_limits
        checks = {
            "finite": bool(finite),
            "suspension_travel": bool(max_susp <= lim["max_suspension_travel"]),
            "front_tire_dynamic_load": bool(front_tire <= lim["max_front_tire_dynamic_load"]),
            "rear_tire_dynamic_load": bool(rear_tire <= lim["max_rear_tire_dynamic_load"]),
        }
        violations = [name for name, ok in checks.items() if not ok]
        return {
            "unsafe": bool(violations),
            "checks": checks,
            "violations": violations,
            "max_abs_suspension_travel": max_susp,
            "max_abs_wheel_displacement": max_wheel,
            "max_abs_front_tire_dynamic_load": front_tire,
            "max_abs_rear_tire_dynamic_load": rear_tire,
            "abs_pitch": abs_pitch,
            "limits": dict(lim),
        }

    def close(self):
        pass

    def _info(self, reward: float, unsafe: bool = False) -> dict[str, Any]:
        road = np.asarray(self.road.pair(self.t, self.speed, self.wheelbase), dtype=np.float64)
        preview = self._road_preview()
        return {
            "engine": "python",
            "time": self.t,
            "state": self.state.copy(),
            "base_obs": self._base_obs(),
            "road": road,
            "road_preview": preview.copy(),
            "action": self.last_action.copy(),
            "derived": dict(self.last_derived),
            "reward": float(reward),
            "reward_scales": dict(self.reward_scales),
            "unsafe": bool(unsafe),
            "safety": dict(getattr(self, "last_safety", None) or self._safety()),
            "scenario": self.scenario.get("name", ""),
        }
