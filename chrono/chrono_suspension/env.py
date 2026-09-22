"""Gymnasium interface with privileged and delayed noisy sensor observations."""

from __future__ import annotations

from collections import deque
from copy import deepcopy

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import load_config, merge_config
from .controllers import linear_matrices, lqr_gain
from .physics import SevenDOFModel
from .roads import FourWheelRoad


class ChronoSuspensionEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        config: dict | None = None,
        observation_mode: str = "privileged",
        render_mode: str | None = None,
    ):
        super().__init__()
        self.config = load_config(config)
        if observation_mode not in {"privileged", "sensors"}:
            raise ValueError("observation_mode must be privileged or sensors")
        if render_mode not in {None, "human"}:
            raise ValueError("Only human native rendering is supported")
        self.observation_mode, self.render_mode = observation_mode, render_mode
        p, o, a = self.config["physics"], self.config["observation"], self.config["actuator"]
        self.dt, self.control_dt = p["dt"], p["control_dt"]
        self.substeps = int(round(self.control_dt / self.dt))
        self.max_steps = int(round(p["episode_seconds"] / self.control_dt))
        self.force_limit = float(a["force_limit_n"])
        self.history_steps = o["history_steps"]
        self.preview_steps = o["preview_steps"]
        if observation_mode == "sensors" and self.preview_steps:
            raise ValueError("Road preview is privileged: sensor mode must use preview_steps=0")
        privileged_size = 14 + 4 + 4 * int(round(a["delay_s"] / self.dt))
        size = privileged_size if observation_mode == "privileged" else 13 * self.history_steps
        size += 4 * self.preview_steps
        self.observation_space = spaces.Box(-np.inf, np.inf, (size,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
        self.model = None
        self.viewer = None
        self._gain = None
        self._jac = linear_matrices(self.config)[2]
        self.previous_action = np.zeros(4)
        self.step_count = 0
        self._last_observation = np.zeros(size, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.model is not None:
            self.model.close()
        scenario = merge_config(self.config["road"], (options or {}).get("scenario", {}))
        road_seed = int(self.np_random.integers(0, 2**31 - 1))
        v = self.config["vehicle"]
        self.road = FourWheelRoad(scenario, v["front_distance_m"], v["rear_distance_m"], road_seed)
        self.model = SevenDOFModel(self.config, self.road)
        self.scenario = deepcopy(scenario)
        self.step_count = 0
        self.previous_action[:] = 0
        sensor = self._measure_sensors(self.model.metrics())
        self.sensor_delay = deque(
            [sensor.copy() for _ in range(self.config["observation"]["sensor_delay_steps"])]
        )
        self.sensor_history = deque(
            [sensor.copy() for _ in range(self.history_steps)], maxlen=self.history_steps
        )
        obs = self._observation()
        self._last_observation = obs.copy()
        return obs, self._info(self.model.metrics())

    def _measure_sensors(self, metrics: dict) -> np.ndarray:
        cfg = self.config["observation"]
        q, qd = self.model.coordinates()
        # z acceleration is gravity-compensated inertial acceleration, NOT raw
        # accelerometer specific force. Attitude estimates are explicitly noisy.
        measured = np.concatenate(
            [
                [metrics["body_acc_mps2"]],
                q[1:3],
                qd[1:3],
                metrics["suspension_travel_m"],
                self.previous_action,
            ]
        )
        sigma = np.array(
            [
                cfg["acceleration_noise_mps2"],
                *([cfg["attitude_noise_rad"]] * 2),
                *([cfg["gyro_noise_radps"]] * 2),
                *([cfg["travel_noise_m"]] * 4),
                *([0.0] * 4),
            ]
        )
        return measured + self.np_random.normal(0.0, sigma)

    def _observation(self) -> np.ndarray:
        if self.observation_mode == "sensors":
            observation = np.concatenate(list(self.sensor_history))
        else:
            q, qd = self.model.coordinates()
            parts = [q, qd, self.model.actuator_force / self.force_limit]
            if self.model.delay_steps:
                parts.append(np.asarray(self.model.delay_buffer).reshape(-1) / self.force_limit)
            observation = np.concatenate(parts)
        if self.preview_steps:
            observation = np.concatenate(
                [
                    observation,
                    self.road.preview(
                        self.model.time,
                        self.preview_steps,
                        self.config["observation"]["preview_dt"],
                    ),
                ]
            )
        return observation.astype(np.float32)

    def _cost(self, metrics: dict, action: np.ndarray) -> float:
        w, v = self.config["reward"], self.config["vehicle"]
        tire_ratio = metrics["tire_load_n"] / self.model.static_load
        components = {
            "acceleration": metrics["body_acc_mps2"] ** 2,
            "attitude": (metrics["roll_rad"] / 0.1) ** 2 + (metrics["pitch_rad"] / 0.1) ** 2,
            "travel": np.mean((metrics["suspension_travel_m"] / v["travel_limit_m"]) ** 2),
            "tire_load": np.mean((tire_ratio - 1) ** 2),
            "force": np.mean((metrics["actuator_force_n"] / self.force_limit) ** 2),
            "power": np.mean(
                (metrics["mechanical_power_w"] / self.config["actuator"]["power_limit_w"]) ** 2
            ),
            "action_rate": np.mean((action - self.previous_action) ** 2),
            "unsafe": np.mean(tire_ratio <= 0)
            + np.mean(np.abs(metrics["suspension_travel_m"]) > v["travel_limit_m"]),
        }
        return float(sum(w[k] * value for k, value in components.items()))

    def _info(self, metrics: dict) -> dict:
        return {
            "time_s": self.model.time,
            "metrics": metrics,
            "mechanical_dof": 7,
            "observation_mode": self.observation_mode,
            "scenario": self.scenario,
            "static_tire_load_n": self.model.static_load.copy(),
        }

    def step(self, action):
        if self.model is None:
            raise RuntimeError("Call reset before step")
        action = np.asarray(action, dtype=float)
        if action.shape != (4,) or not np.all(np.isfinite(action)):
            raise ValueError(
                "Action must contain four finite normalized commands in FL FR RL RR order"
            )
        action = np.clip(action, -1.0, 1.0)
        costs = []
        contact_loss = np.zeros(4)
        max_travel = np.zeros(4)
        square_sums = np.zeros(3)
        force_square_sum = np.zeros(4)
        work_positive = np.zeros(4)
        work_negative = np.zeros(4)
        tire_min = np.full(4, np.inf)
        failed = False
        failure_reason = None
        for _ in range(self.substeps):
            self.model.step(action * self.force_limit)
            metrics = self.model.metrics()
            q, qd = self.model.coordinates()
            if not np.all(np.isfinite(np.r_[q, qd])) or not all(
                np.all(np.isfinite(value)) for value in metrics.values()
            ):
                failed = True
                failure_reason = "nonfinite_state_or_metric"
                break
            contact_loss += metrics["tire_load_n"] <= 0
            max_travel = np.maximum(max_travel, np.abs(metrics["suspension_travel_m"]))
            square_sums += np.square(
                [metrics["body_acc_mps2"], metrics["roll_rad"], metrics["pitch_rad"]]
            )
            force_square_sum += metrics["actuator_force_n"] ** 2
            work_positive += np.maximum(metrics["mechanical_power_w"], 0) * self.dt
            work_negative += np.minimum(metrics["mechanical_power_w"], 0) * self.dt
            tire_min = np.minimum(tire_min, metrics["tire_load_n"])
            costs.append(self._cost(metrics, action))
            if (
                np.max(np.abs(q[1:3])) > 0.5
                or np.max(max_travel) > 2 * self.config["vehicle"]["travel_limit_m"]
            ):
                failed = True
                failure_reason = "physical_envelope_exceeded"
                break
            if np.min(q[3:] + self.model.wheel_z0) <= 0:
                failed = True
                failure_reason = "wheel_below_tire_anchor"
                break
            if any(
                suspension.GetPoint1Abs().z <= suspension.GetPoint2Abs().z
                for suspension in self.model.suspensions
            ):
                failed = True
                failure_reason = "suspension_mount_crossed_wheel"
                break
        self.step_count += 1
        self.previous_action = action.copy()
        info = self._info(metrics)
        info["contact_loss_fraction"] = contact_loss / max(1, len(costs))
        info["max_travel_m"] = max_travel
        info["physical_failure"] = failed
        counted_steps = max(1, len(costs))
        info["substep_metrics"] = {
            "body_acc_squared_mean": float(square_sums[0] / counted_steps),
            "roll_squared_mean": float(square_sums[1] / counted_steps),
            "pitch_squared_mean": float(square_sums[2] / counted_steps),
            "actuator_force_squared_mean": force_square_sum / counted_steps,
            "mechanical_work_positive_j": work_positive,
            "mechanical_work_negative_j": work_negative,
            "tire_load_min_n": tire_min,
            "duration_s": len(costs) * self.dt,
        }
        if failed:
            info["failure_reason"] = failure_reason
            return self._last_observation.copy(), -10000.0, True, False, info
        if self.observation_mode == "sensors":
            self.sensor_delay.append(self._measure_sensors(metrics))
            self.sensor_history.append(self.sensor_delay.popleft())
        obs = self._observation()
        self._last_observation = obs.copy()
        if self.render_mode == "human":
            self.render()
        return obs, -float(np.mean(costs)), False, self.step_count >= self.max_steps, info

    def baseline_action(self, name: str) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call reset before requesting baseline actions")
        q, qd = self.model.coordinates()
        if name == "passive":
            force = np.zeros(4)
        elif name == "skyhook":
            sprung_corner_velocity = self._jac[:, :3] @ qd[:3]
            force = -3000.0 * sprung_corner_velocity
        elif name == "lqr":
            if self._gain is None:
                self._gain = lqr_gain(self.config)
            force = -self._gain @ np.r_[q, qd]
        else:
            raise ValueError(f"Unknown baseline {name}")
        return np.clip(force / self.force_limit, -1, 1).astype(np.float32)

    def export_frame(self) -> dict:
        if self.model is None:
            raise RuntimeError("Call reset before exporting frames")
        return self.model.export_frame()

    def render(self):
        if self.model is None:
            return None
        if self.viewer is None:
            from .visualization import NativeViewer

            self.viewer = NativeViewer(self.model, self.config)
        return self.viewer.render(self.export_frame())

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.model is not None:
            self.model.close()
        self.model = None
