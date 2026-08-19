from __future__ import annotations

import numpy as np

from .base import Controller
from models.half_car import HalfCarModel

try:
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover
    minimize = None


class MPCController(Controller):
    name = "MPC"

    def __init__(self, model: HalfCarModel, config: dict):
        self.model = model
        self.config = config
        self.horizon = int(config["mpc"]["horizon"])
        self.dt = float(config["control_dt"])
        self.force_limit = float(config["force_limit"])
        self.max_iter = int(config["mpc"].get("max_iter", 40))
        self.replan_interval = int(config["mpc"].get("replan_interval", 1))
        mpc_cfg = dict(config["mpc"])
        reward_cfg = dict(config.get("reward", {}))
        norm_cfg = dict(config.get("normalization", {}))
        front_static_load, rear_static_load = self.model.p.static_tire_loads()
        self.norm_body_acc = max(float(norm_cfg.get("body_acc", 0.491)), 1e-9)
        self.norm_pitch_acc = max(float(norm_cfg.get("pitch_acc", 1.821)), 1e-9)
        self.norm_suspension = max(float(norm_cfg.get("suspension_travel", 0.08)), 1e-9)
        self.norm_front_tire = max(float(norm_cfg.get("front_tire_load", front_static_load)), 1e-9)
        self.norm_rear_tire = max(float(norm_cfg.get("rear_tire_load", rear_static_load)), 1e-9)
        self.norm_control = max(float(norm_cfg.get("control_force", self.force_limit)), 1e-9)
        self.w_body = float(mpc_cfg.get("w_body", reward_cfg.get("body_acc", 0.45)))
        self.w_pitch = float(mpc_cfg.get("w_pitch", reward_cfg.get("pitch_acc", 0.20)))
        self.w_susp = float(mpc_cfg.get("w_susp", reward_cfg.get("suspension", 0.25)))
        self.w_tire = float(mpc_cfg.get("w_tire", reward_cfg.get("tire_load", 0.10)))
        self.w_control = float(
            mpc_cfg.get("w_control", mpc_cfg.get("control_weight", reward_cfg.get("control", 0.05)))
        )
        self.delta_weight = float(mpc_cfg.get("delta_weight", 0.0))
        safety_cfg = dict(config.get("safety_limits", {}))
        self.constraint_weight = float(mpc_cfg.get("constraint_weight", 0.0))
        self.constraint_margin = float(np.clip(mpc_cfg.get("constraint_margin", 0.85), 0.0, 1.0))
        self.norm_step = float(mpc_cfg.get("finite_diff_eps", 1e-4))
        self.max_suspension = max(float(safety_cfg.get("max_suspension_travel", self.norm_suspension)), 1e-9)
        self.preview_stride = max(1, int(round(self.dt / float(config.get("preview", {}).get("dt", self.dt)))))
        self.prev_solution = np.zeros((self.horizon, 2), dtype=np.float64)
        self.step_count = 0

    def reset(self):
        self.prev_solution[:] = 0.0
        self.step_count = 0

    def compute_action(self, obs: np.ndarray, info: dict) -> np.ndarray:
        if minimize is None:
            raise RuntimeError("scipy is required for MPCController")
        if self.step_count % self.replan_interval != 0:
            action = self.prev_solution[0].copy()
            self.prev_solution = np.roll(self.prev_solution, -1, axis=0)
            self.prev_solution[-1] = self.prev_solution[-2]
            self.step_count += 1
            return np.clip(action, -self.force_limit, self.force_limit)

        preview = np.asarray(info["road_preview"], dtype=np.float64)
        if self.preview_stride > 1:
            preview = preview[::self.preview_stride]
        x0 = np.asarray(info["state"], dtype=np.float64)
        current_road = np.asarray(info["road"], dtype=np.float64)
        z_current = current_road[[0, 2]]
        z_seq = np.vstack([z_current, preview[: self.horizon]])

        def cost(flat_v):
            u_seq = flat_v.reshape(self.horizon, 2) * self.norm_control
            x = x0.copy()
            total = 0.0
            last_u = np.zeros(2)
            for k in range(self.horizon):
                dzf = (z_seq[k + 1, 0] - z_seq[k, 0]) / self.dt
                dzr = (z_seq[k + 1, 1] - z_seq[k, 1]) / self.dt
                road = np.asarray([z_seq[k, 0], dzf, z_seq[k, 1], dzr])
                u = np.clip(u_seq[k], -self.force_limit, self.force_limit)
                d = self.model.derived(x, road, u)
                total += (
                    self.w_body * (d["ddzb"] / self.norm_body_acc) ** 2
                    + self.w_pitch * (d["ddtheta"] / self.norm_pitch_acc) ** 2
                    + self.w_susp
                    * (
                        (d["delta_yf"] / self.norm_suspension) ** 2
                        + (d["delta_yr"] / self.norm_suspension) ** 2
                    )
                    + self.w_tire
                    * (
                        (d["Fpf"] / self.norm_front_tire) ** 2
                        + (d["Fpr"] / self.norm_rear_tire) ** 2
                    )
                    + self.w_control * np.sum((u / self.norm_control) ** 2)
                    + self.delta_weight * np.sum(((u - last_u) / self.norm_control) ** 2)
                )
                if self.constraint_weight > 0.0:
                    total += self.constraint_weight * self._constraint_penalty(d)
                x = self.model.rk4_step(x, u, road, self.dt)
                last_u = u
            return float(total)

        x_init = np.roll(self.prev_solution, -1, axis=0) / self.norm_control
        x_init[-1] = x_init[-2]
        control_bound = self.force_limit / self.norm_control
        bounds = [(-control_bound, control_bound)] * (self.horizon * 2)
        result = minimize(
            cost,
            x_init.reshape(-1),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iter, "ftol": 1e-5, "eps": self.norm_step},
        )
        if result.success or np.isfinite(result.fun):
            self.prev_solution = result.x.reshape(self.horizon, 2) * self.norm_control
        action = self.prev_solution[0].copy()
        self.prev_solution = np.roll(self.prev_solution, -1, axis=0)
        self.prev_solution[-1] = self.prev_solution[-2]
        self.step_count += 1
        return np.clip(action, -self.force_limit, self.force_limit)

    def _constraint_penalty(self, d: dict) -> float:
        margin = self.constraint_margin
        terms = (
            abs(d["delta_yf"]) / self.max_suspension,
            abs(d["delta_yr"]) / self.max_suspension,
            abs(d["Fpf"]) / self.norm_front_tire,
            abs(d["Fpr"]) / self.norm_rear_tire,
        )
        return float(sum(max(0.0, value - margin) ** 2 for value in terms))
