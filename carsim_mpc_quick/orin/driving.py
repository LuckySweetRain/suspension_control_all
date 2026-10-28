"""Shared trajectory/PID contract, independent of CARLA and model dependencies.

All poses use metres/radians in a right-handed ground frame: x forward, y left,
yaw positive left. Convert CARLA's y/yaw at the boundary. A model plan is anchored
to the measured pose and simulation time of its input, never to its reply time.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


def _finite(values):
    return all(math.isfinite(v) for v in values)


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    yaw: float

    def __post_init__(self):
        if not _finite((self.x, self.y, self.yaw)):
            raise ValueError("Pose must be finite")

    def world_point(self, x, y):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return self.x + c * x - s * y, self.y + s * x + c * y

    def local_point(self, x, y):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx, dy = x - self.x, y - self.y
        return c * dx + s * dy, -s * dx + c * dy


@dataclass(frozen=True)
class Trajectory:
    """Future ego-frame (time from capture [s], x [m], y [m]) samples.

    A t=0 pose may be present; positive first timestamps are supported too.
    Model/checkpoint identity is required so logs cannot silently label a route
    generator as neural inference. Identity itself does not prove authenticity.
    """

    model: str
    checkpoint: str
    sequence: int
    captured_at: float
    origin: Pose
    points: tuple[tuple[float, float, float], ...]

    def __post_init__(self):
        if not self.model or not self.checkpoint:
            raise ValueError("Model and checkpoint identity are required")
        if self.sequence < 0 or not math.isfinite(self.captured_at):
            raise ValueError("Invalid trajectory sequence or simulation timestamp")
        # Freeze nested inputs too, rather than retaining a mutable network payload.
        points = tuple(tuple(float(v) for v in p) for p in self.points)
        if len(points) < 2 or any(len(p) != 3 or not _finite(p) for p in points):
            raise ValueError("Expected at least two finite (time, x, y) points")
        if points[0][0] < 0 or any(b[0] <= a[0] for a, b in zip(points, points[1:])):
            raise ValueError("Trajectory times must be nonnegative and strictly increasing")
        object.__setattr__(self, "points", points)


@dataclass(frozen=True)
class DrivingControl:
    """Road-wheel command [rad], normalized pedals, and diagnostic target speed.

    Steering is positive left. This is not a VehicleSim import buffer: the adapter
    must map to verified steering-wheel angle and brake pressure units.
    """

    steer_rad: float
    throttle: float
    brake: float
    target_speed_mps: float
    reason: str = "tracking"


class PID:
    def __init__(self, kp, ki, kd, limit):
        if not _finite((kp, ki, kd, limit)) or limit <= 0:
            raise ValueError("Invalid PID parameters")
        self.kp, self.ki, self.kd, self.limit = kp, ki, kd, limit
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.previous = None

    def step(self, error, dt):
        if not _finite((error, dt)) or dt <= 0:
            raise ValueError("PID requires finite error and positive dt")
        derivative = 0.0 if self.previous is None else (error - self.previous) / dt
        proposed = self.integral + error * dt
        raw = self.kp * error + self.ki * proposed + self.kd * derivative
        # Conditional integration allows unwinding a saturated integrator.
        if abs(raw) <= self.limit or raw * error < 0:
            self.integral = proposed
        self.previous = error
        return max(-self.limit, min(self.limit, raw))


class TrajectoryTracker:
    """100 Hz PID consumer of a slower model plan, with explicit stale-plan stop.

    Gains below are initial integration settings, not a calibrated CarSim result.
    Model runs may pause the synchronous simulator while awaiting inference; in
    that mode plan age is still measured in simulation seconds, not wall time.
    """

    def __init__(self, max_age=0.5, max_speed_mps=12.0, max_steer_rad=0.45):
        if (
            not _finite((max_age, max_speed_mps, max_steer_rad))
            or min(max_age, max_speed_mps, max_steer_rad) <= 0
        ):
            raise ValueError("Tracker limits must be positive and finite")
        self.max_age = max_age
        self.max_speed = max_speed_mps
        self.steering = PID(0.8, 0.01, 0.03, max_steer_rad)
        self.speed = PID(0.3, 0.08, 0.0, 1.0)

    def reset(self):
        self.steering.reset()
        self.speed.reset()

    def stop(self, reason):
        self.reset()
        return DrivingControl(0.0, 0.0, 1.0, 0.0, reason)

    def step(self, plan, pose, speed_mps, sim_time, dt=0.01):
        if not _finite((speed_mps, sim_time, dt)) or dt <= 0:
            raise ValueError("Invalid measured speed, time, or control period")
        if plan is None:
            return self.stop("missing_plan")
        age = sim_time - plan.captured_at
        if age < -1e-6 or age > self.max_age:
            return self.stop("future_plan" if age < 0 else "stale_plan")
        remaining = [p for p in plan.points if p[0] > age]
        if len(remaining) < 2:
            return self.stop("exhausted_plan")
        a, b = remaining[:2]
        target_speed = min(self.max_speed, math.hypot(b[1] - a[1], b[2] - a[2]) / (b[0] - a[0]))
        if target_speed < 0.3:
            return self.stop("planned_stop")
        lookahead = max(2.0, max(0.0, speed_mps) * 0.5)
        aim = None
        for point in remaining:
            local = pose.local_point(*plan.origin.world_point(point[1], point[2]))
            if local[0] > 0:
                aim = local
                if math.hypot(*local) >= lookahead:
                    break
        if aim is None:
            return self.stop("plan_behind_vehicle")
        steer = self.steering.step(math.atan2(aim[1], aim[0]), dt)
        effort = self.speed.step(target_speed - speed_mps, dt)
        return DrivingControl(steer, max(0.0, effort), max(0.0, -effort), target_speed)


class DrivingSession:
    """Optional per-MPC-connection driving state; all PID calculations stay on Orin."""

    def __init__(self):
        self.tracker = TrajectoryTracker()
        self.plan = None
        self.tcp_control = None

    def step(self, request, sim_time):
        if "cruise_speed_mps" in request:
            raise ValueError("Cruise speed override has been removed")
        packet = request.get("plan")
        if packet is not None:
            plan = Trajectory(
                packet["model"],
                packet["checkpoint"],
                packet["seq"],
                packet["captured_at"],
                Pose(*packet["origin"]),
                packet["points"],
            )
            if self.plan is not None and plan.sequence <= self.plan.sequence:
                raise ValueError("Driving plan sequence did not increase")
            self.plan = plan
            if plan.model == "tcp-third-party-reproduction":
                action = packet["tcp_control"]
                if action["policy"] != "tcp_official_agent":
                    raise ValueError("TCP requires its original agent policy")
                steer, throttle, brake = (
                    float(action[k]) for k in ("steer_right_positive", "throttle", "brake")
                )
                if not _finite((steer, throttle, brake)) or not (
                    -1 <= steer <= 1 and 0 <= throttle <= 0.75 and 0 <= brake <= 1
                ):
                    raise ValueError("Invalid native TCP actuator command")
                self.tcp_control = dict(action)
        if self.plan is not None and self.plan.model == "tcp-third-party-reproduction":
            if not -1e-6 <= sim_time - self.plan.captured_at <= self.tracker.max_age:
                raise ValueError("Stale or future native TCP command")
            action = self.tcp_control
            # CARLA steer is right-positive and normalized; CarSim wants a
            # left-positive steering-wheel angle. Keep the existing provisional
            # max road-wheel angle, steering ratio and hydraulic pressure map.
            steer_rad = -float(action["steer_right_positive"]) * 0.45
            return dict(
                steer_rad=steer_rad,
                throttle=float(action["throttle"]),
                brake=float(action["brake"]),
                target_speed_mps=action["desired_speed_mps"],
                reason="tcp_original",
                control_source="tcp_official_agent",
                native_steer_right_positive=action["steer_right_positive"],
                carsim_inputs=[
                    math.degrees(steer_rad) * 16.0,
                    action["throttle"],
                    action["brake"] * 6.0,
                ],
                actuator_calibration="provisional_ratio16_brake6MPa_max_roadwheel0.45rad",
                plan_seq=self.plan.sequence,
            )
        control = self.tracker.step(
            self.plan,
            Pose(*request["pose"]),
            float(request["speed_mps"]),
            sim_time,
        )
        # Initial integration calibration: actual D-class steering kinematics and
        # pressure response must be tuned before claiming driving performance.
        steer_deg = math.degrees(control.steer_rad) * 16.0
        result = asdict(control)
        result["carsim_inputs"] = [steer_deg, control.throttle, control.brake * 6.0]
        result["actuator_calibration"] = "provisional_ratio16_brake6MPa"
        result["plan_seq"] = None if self.plan is None else self.plan.sequence
        return result
