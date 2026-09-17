from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from models.half_car import HalfCarParams

from .base import Controller, split_generalized_force


class SPDFController(Controller):
    name = "SPDF"

    def __init__(
        self,
        params: HalfCarParams,
        dt: float,
        period: float = 1.0,
        config: dict[str, Any] | None = None,
    ):
        self.p = params
        self.dt = dt
        self.period = period
        self.h = period / 4.0
        self.delay_steps = max(1, int(round(self.h / dt)))
        self.a1 = 40.0
        self.a2 = 40.0
        self.tau1 = 0.75
        self.tau2 = 0.75
        self.eps = 1e-3
        full_config = dict(config or {})
        spdf_cfg = dict(full_config.get("spdf", full_config))
        ref_cfg = dict(spdf_cfg.get("preview_reference", {}))
        safety_cfg = dict(full_config.get("safety_limits", {}))
        max_suspension = max(
            float(
                ref_cfg.get("max_suspension_travel", safety_cfg.get("max_suspension_travel", 0.08))
            ),
            1e-9,
        )
        stroke_fraction = float(ref_cfg.get("stroke_fraction", 0.35))
        stroke_default = max_suspension * stroke_fraction
        self.uses_preview = bool(ref_cfg.get("enabled", False))
        self.ref_stroke_limit = float(
            np.clip(ref_cfg.get("stroke_limit", stroke_default), 1e-9, max_suspension)
        )
        self.ref_lookahead_steps = max(1, int(ref_cfg.get("lookahead_steps", 10)))
        self.ref_peak_blend = float(np.clip(ref_cfg.get("peak_blend", 0.75), 0.0, 1.0))
        self.ref_body_kp = max(float(ref_cfg.get("body_kp", 25.0)), 0.0)
        self.ref_body_kd = max(float(ref_cfg.get("body_kd", 10.0)), 0.0)
        self.ref_pitch_kp = max(float(ref_cfg.get("pitch_kp", 25.0)), 0.0)
        self.ref_pitch_kd = max(float(ref_cfg.get("pitch_kd", 10.0)), 0.0)
        self.ref_body_acc_limit = max(float(ref_cfg.get("body_acc_limit", 0.8)), 1e-9)
        self.ref_pitch_acc_limit = max(float(ref_cfg.get("pitch_acc_limit", 3.0)), 1e-9)
        self.ref_body_vel_limit = max(float(ref_cfg.get("body_vel_limit", 1.0)), 1e-9)
        self.ref_pitch_vel_limit = max(float(ref_cfg.get("pitch_vel_limit", 2.0)), 1e-9)
        self.ref_body_abs_limit = max(float(ref_cfg.get("body_abs_limit", max_suspension)), 1e-9)
        self.ref_pitch_abs_limit = max(float(ref_cfg.get("pitch_abs_limit", 0.35)), 1e-9)
        self.reset()

    def reset(self):
        self.time = 0.0
        self.history = deque(maxlen=2 * self.delay_steps + 2)
        for _ in range(2 * self.delay_steps + 2):
            # Simulink delays the mux [s1, s2, x1, x2, x3, x4] by T/4.
            self.history.append(np.zeros(6, dtype=np.float64))
        self.prev_k1 = 0.0
        self.prev_k2 = 0.0
        self.ref_z = 0.0
        self.ref_dz = 0.0
        self.ref_ddz = 0.0
        self.ref_theta = 0.0
        self.ref_dtheta = 0.0
        self.ref_ddtheta = 0.0
        self.last_reference = self._reference_payload(0.0, 0.0)

    def compute_action(self, obs: np.ndarray, info: dict) -> np.ndarray:
        p = self.p
        x = np.asarray(info["state"], dtype=np.float64)
        d = info["derived"]
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
        ref = self._reference(info)
        e1 = x1 - ref["z"]
        de1 = x2 - ref["dz"]
        e2 = x3 - ref["theta"]
        de2 = x4 - ref["dtheta"]
        delayed = self.history[-self.delay_steps]
        s1_h, s2_h, e1_h, de1_h, e2_h, de2_h = delayed
        t = self.time % self.period

        k1 = self._k_a(t, self.a1)
        k2 = self._k_a(t, self.a2)
        dk1 = (k1 - self.prev_k1) / self.dt
        dk2 = (k2 - self.prev_k2) / self.dt
        self.prev_k1 = k1
        self.prev_k2 = k2

        s1 = de1 + self.a1 * e1 + k1 * e1_h
        s2 = de2 + self.a2 * e2 + k2 * e2_h

        ro1 = self._ro1(d)
        ro2 = self._ro2(d)
        f1 = self._f1(d)
        f2 = self._f2(d)
        delta_g1 = (p.mb_max - p.mb) / p.mb_min
        delta_g2 = (p.Ip_max - p.Ip) / p.Ip_min

        ub0 = -self.a1 / (2 * (1 - self.tau1)) * s1 - self._k_ah(t, self.a1) / (
            2 * (1 - self.tau1)
        ) * np.sign(s1) * abs(s1) ** (2 * self.tau1 - 1) * abs(s1_h) ** (2 - 2 * self.tau1)
        base1 = (
            -2 / np.pi * ro1 * np.arctan(ro1 * s1 / self.eps)
            - self.a1 * de1
            - dk1 * e1_h
            - k1 * de1_h
            + ub0
        )
        gamma1 = 1 / (1 - delta_g1) * delta_g1 * abs(base1)
        vb0 = -2 / np.pi * gamma1 * np.arctan(gamma1 * s1 / self.eps)
        vb = p.mb * (ref["ddz"] + base1 + vb0)
        ub = vb + f1

        utheta0 = -self.a2 / (2 * (1 - self.tau2)) * s2 - self._k_ah(t, self.a2) / (
            2 * (1 - self.tau2)
        ) * np.sign(s2) * abs(s2) ** (2 * self.tau2 - 1) * abs(s2_h) ** (2 - 2 * self.tau2)
        base2 = (
            -2 / np.pi * ro2 * np.arctan(ro2 * s2 / self.eps)
            - self.a2 * de2
            - dk2 * e2_h
            - k2 * de2_h
            + utheta0
        )
        gamma2 = 1 / (1 - delta_g2) * delta_g2 * abs(base2)
        vtheta0 = -2 / np.pi * gamma2 * np.arctan(gamma2 * s2 / self.eps)
        vtheta = p.Ip * (ref["ddtheta"] + base2 + vtheta0)
        utheta = vtheta + f2

        self.history.append(np.asarray([s1, s2, e1, de1, e2, de2], dtype=np.float64))
        self.time += self.dt
        return split_generalized_force(ub, utheta, p.a, p.b, p.fmax)

    def _reference(self, info: dict) -> dict[str, float]:
        if not self.uses_preview:
            self.last_reference = self._reference_payload(0.0, 0.0)
            return self.last_reference
        command_z, command_theta = self._preview_command(info)
        self.ref_z, self.ref_dz, self.ref_ddz = self._advance_axis(
            q=self.ref_z,
            dq=self.ref_dz,
            command=command_z,
            kp=self.ref_body_kp,
            kd=self.ref_body_kd,
            acc_limit=self.ref_body_acc_limit,
            vel_limit=self.ref_body_vel_limit,
            abs_limit=self.ref_body_abs_limit,
        )
        self.ref_theta, self.ref_dtheta, self.ref_ddtheta = self._advance_axis(
            q=self.ref_theta,
            dq=self.ref_dtheta,
            command=command_theta,
            kp=self.ref_pitch_kp,
            kd=self.ref_pitch_kd,
            acc_limit=self.ref_pitch_acc_limit,
            vel_limit=self.ref_pitch_vel_limit,
            abs_limit=self.ref_pitch_abs_limit,
        )
        self.last_reference = self._reference_payload(command_z, command_theta)
        return self.last_reference

    def _preview_command(self, info: dict) -> tuple[float, float]:
        p = self.p
        road_now = np.asarray(info.get("road", np.zeros(4)), dtype=np.float64).reshape(-1)
        current = np.asarray(
            [[road_now[0] if road_now.size > 0 else 0.0, road_now[2] if road_now.size > 2 else 0.0]]
        )
        preview = np.asarray(info.get("road_preview", np.empty((0, 2))), dtype=np.float64)
        if preview.size == 0:
            road_heights = current
        else:
            preview = preview.reshape(-1, 2)
            road_heights = np.vstack([current, preview])
        horizon = min(max(1, self.ref_lookahead_steps + 1), road_heights.shape[0])
        road_heights = road_heights[:horizon]

        z_front = road_heights[:, 0]
        z_rear = road_heights[:, 1]
        front_body = np.clip(0.0, z_front - self.ref_stroke_limit, z_front + self.ref_stroke_limit)
        rear_body = np.clip(0.0, z_rear - self.ref_stroke_limit, z_rear + self.ref_stroke_limit)
        wheelbase = max(p.a + p.b, 1e-9)
        raw_z = (p.b * front_body + p.a * rear_body) / wheelbase
        raw_theta = (front_body - rear_body) / wheelbase
        return self._select_preview_value(raw_z), self._select_preview_value(raw_theta)

    def _select_preview_value(self, values: np.ndarray) -> float:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size == 0:
            return 0.0
        weights = np.linspace(1.0, 0.35, values.size)
        mean_value = float(np.average(values, weights=weights))
        peak_value = float(values[int(np.argmax(np.abs(values)))])
        return (1.0 - self.ref_peak_blend) * mean_value + self.ref_peak_blend * peak_value

    def _advance_axis(
        self,
        q: float,
        dq: float,
        command: float,
        kp: float,
        kd: float,
        acc_limit: float,
        vel_limit: float,
        abs_limit: float,
    ) -> tuple[float, float, float]:
        ddq = float(np.clip(kp * (command - q) - kd * dq, -acc_limit, acc_limit))
        dq_next = float(np.clip(dq + ddq * self.dt, -vel_limit, vel_limit))
        q_next = q + dq_next * self.dt
        if q_next > abs_limit:
            q_next = abs_limit
            dq_next = min(dq_next, 0.0)
        elif q_next < -abs_limit:
            q_next = -abs_limit
            dq_next = max(dq_next, 0.0)
        return float(q_next), float(dq_next), ddq

    def _reference_payload(self, command_z: float, command_theta: float) -> dict[str, float]:
        p = self.p
        return {
            "z": float(self.ref_z),
            "dz": float(self.ref_dz),
            "ddz": float(self.ref_ddz),
            "theta": float(self.ref_theta),
            "dtheta": float(self.ref_dtheta),
            "ddtheta": float(self.ref_ddtheta),
            "command_z": float(command_z),
            "command_theta": float(command_theta),
            "front_body": float(self.ref_z + p.a * self.ref_theta),
            "rear_body": float(self.ref_z - p.b * self.ref_theta),
            "stroke_limit": float(self.ref_stroke_limit),
        }

    def _k_a(self, t: float, a: float) -> float:
        if (t % (4 * self.h)) < 2 * self.h:
            return 0.0
        return self._k_ah(t, a)

    def _k_ah(self, t: float, a: float) -> float:
        t1 = t % (2 * self.h)
        if t1 < self.h:
            return 0.0
        return float(
            (
                4
                * a
                * np.sin(np.pi * t1 / self.h) ** 2
                * np.exp(-a * (3 * self.h - 2 * t1))
                * (a**2 * self.h**2 + np.pi**2)
            )
            / (np.pi**2 * (np.exp(2 * a * self.h) - 1))
        )

    def _f1(self, d: dict) -> float:
        p = self.p
        dyf, ddyf, dyr, ddyr = d["delta_yf"], d["delta_dyf"], d["delta_yr"], d["delta_dyr"]
        return (
            p.kf1 * dyf
            + p.knf1 * dyf**3
            + p.kr1 * dyr
            + p.knr1 * dyr**3
            + 0.5 * ((p.be + p.bc) + (p.be - p.bc) * np.sign(ddyf)) * ddyf
            + 0.5 * ((p.be + p.bc) + (p.be - p.bc) * np.sign(ddyr)) * ddyr
        )

    def _f2(self, d: dict) -> float:
        p = self.p
        dyf, ddyf, dyr, ddyr = d["delta_yf"], d["delta_dyf"], d["delta_yr"], d["delta_dyr"]
        return p.a * (
            p.kf1 * dyf
            + p.knf1 * dyf**3
            + 0.5 * ((p.be + p.bc) + (p.be - p.bc) * np.sign(ddyf)) * ddyf
        ) - p.b * (
            p.kr1 * dyr
            + p.knr1 * dyr**3
            + 0.5 * ((p.be + p.bc) + (p.be - p.bc) * np.sign(ddyr)) * ddyr
        )

    def _ro1(self, d: dict) -> float:
        p = self.p
        dyf, ddyf, dyr, ddyr = d["delta_yf"], d["delta_dyf"], d["delta_yr"], d["delta_dyr"]
        return (
            1
            / p.mb_min
            * (
                abs(p.delta_kf1 * dyf)
                + abs(p.delta_knf1 * dyf**3)
                + abs(max(p.delta_be, p.delta_bc) * ddyf)
                + abs(p.delta_kr1 * dyr)
                + abs(p.delta_knr1 * dyr**3)
                + abs(max(p.delta_be, p.delta_bc) * ddyr)
            )
        )

    def _ro2(self, d: dict) -> float:
        p = self.p
        dyf, ddyf, dyr, ddyr = d["delta_yf"], d["delta_dyf"], d["delta_yr"], d["delta_dyr"]
        return (
            1
            / p.Ip_min
            * (
                (p.a + p.delta_ab)
                * (
                    abs(p.delta_kf1 * dyf)
                    + abs(p.delta_knf1 * dyf**3)
                    + abs(max(p.delta_be, p.delta_bc) * ddyf)
                )
                + (p.b + p.delta_ab)
                * (
                    abs(p.delta_kr1 * dyr)
                    + abs(p.delta_knr1 * dyr**3)
                    + abs(max(p.delta_be, p.delta_bc) * ddyr)
                )
                + p.delta_ab * abs(self._f1(d))
            )
        )
