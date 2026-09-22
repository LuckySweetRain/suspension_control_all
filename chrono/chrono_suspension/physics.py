"""Chrono plant: 5 bodies, 23 independent constraints, 7 mechanical DOFs.

Axes are x forward, y left, z up. Positive roll raises the left side;
positive right-handed pitch lowers the front. Commands are positive when
pushing the chassis away from the wheel. Actuator states are not mechanical
degrees of freedom. No collision forces are added on top of the tire law.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pychrono as ch

from .roads import FourWheelRoad


def vector(v) -> np.ndarray:
    return np.array([v.x, v.y, v.z], dtype=float)


class SuspensionForce(ch.ForceFunctor):
    def __init__(self, model: SevenDOFModel, index: int):
        super().__init__()
        self.model, self.index = model, index

    def evaluate(self, time, rest_length, length, vel, link):
        m, i = self.model, self.index
        extension = length - rest_length
        cfg = m.config["vehicle"]
        factor = cfg["rebound_damping_ratio"] if vel > 0 else cfg["compression_damping_ratio"]
        force = m.preload[i] - m.ks[i] * extension - m.cs[i] * factor * vel
        force -= cfg["spring_cubic_npm3"] * extension**3
        excess = max(0.0, abs(extension) - cfg["bump_stop_start_m"])
        force -= np.sign(extension) * cfg["bump_stop_npm2"] * excess**2
        return float(force + m.actuator_force[i])


class TireForce(ch.ForceFunctor):
    def __init__(self, model: SevenDOFModel, index: int):
        super().__init__()
        self.model, self.index = model, index

    def evaluate(self, time, rest_length, length, vel, link):
        m, i = self.model, self.index
        road, road_vel = m.road_sample(time)
        compression = m.radius + road[i] - length
        if compression <= 0:
            return 0.0
        return float(max(0.0, m.kt[i] * compression + m.ct[i] * (road_vel[i] - vel)))


class SevenDOFModel:
    def __init__(self, config: dict, road: FourWheelRoad):
        self.config, self.road = config, road
        p, v, a = config["physics"], config["vehicle"], config["actuator"]
        self.dt = float(p["dt"])
        self.radius = float(v["tire_radius_m"])
        self.ks = np.asarray(v["suspension_stiffness_npm"], dtype=float)
        self.cs = np.asarray(v["suspension_damping_nspm"], dtype=float)
        self.kt = np.asarray(v["tire_stiffness_npm"], dtype=float)
        self.ct = np.asarray(v["tire_damping_nspm"], dtype=float)
        self.mu = np.asarray(v["unsprung_mass_kg"], dtype=float)
        front, rear = v["front_distance_m"], v["rear_distance_m"]
        self.xy = np.array(
            [
                [front, v["front_half_track_m"]],
                [front, -v["front_half_track_m"]],
                [-rear, v["rear_half_track_m"]],
                [-rear, -v["rear_half_track_m"]],
            ]
        )
        self.preload = (
            v["sprung_mass_kg"] * 9.81 / (front + rear) * np.array([rear, rear, front, front]) / 2
        )
        self.static_load = self.preload + self.mu * 9.81
        self.wheel_z0 = self.radius - self.static_load / self.kt
        self.body_z0 = float(v["body_height_m"])
        self.top_local = [ch.ChVector3d(x, y, -0.1) for x, y in self.xy]
        self.actuator_force = np.zeros(4)
        self.delay_steps = int(round(a["delay_s"] / self.dt))
        self.delay_buffer = deque(np.zeros(4) for _ in range(self.delay_steps))
        self._road_cache_time = None
        self._road_cache_value = None
        self._build()

    def _build(self) -> None:
        v = self.config["vehicle"]
        self.system = ch.ChSystemNSC()
        self.system.SetNumThreads(1, 1, 1)
        self.system.SetGravitationalAcceleration(ch.ChVector3d(0, 0, -9.81))
        self.system.SetTimestepperType(ch.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)
        self.system.SetSolverType(ch.ChSolver.Type_MINRES)
        solver = self.system.GetSolver()
        if hasattr(solver, "SetMaxIterations"):
            solver.SetMaxIterations(150)
        if hasattr(solver, "SetTolerance"):
            solver.SetTolerance(1e-10)
        self.ground = ch.ChBody()
        self.ground.SetFixed(True)
        self.system.AddBody(self.ground)
        self.chassis = ch.ChBody()
        self.chassis.SetName("sprung_body")
        self.chassis.SetMass(v["sprung_mass_kg"])
        self.chassis.SetInertiaXX(ch.ChVector3d(*v["inertia_kgm2"]))
        self.chassis.SetPos(ch.ChVector3d(0, 0, self.body_z0))
        self.system.AddBody(self.chassis)
        self.constraints = []
        planar = ch.ChLinkMateGeneric(True, True, False, False, False, False)
        planar.Initialize(self.chassis, self.ground, ch.ChFramed(self.chassis.GetPos()))
        self.system.AddLink(planar)
        self.constraints.append(planar)
        # body X orthogonal to world Y gives sin(yaw)*cos(pitch)=0.
        # In the allowed pitch range this is yaw=0, without constraining roll.
        yaw = ch.ChLinkMateOrthogonal()
        yaw.Initialize(
            self.chassis,
            self.ground,
            True,
            ch.ChVector3d(0, 0, 0),
            ch.ChVector3d(0, 0, 0),
            ch.ChVector3d(1, 0, 0),
            ch.ChVector3d(0, 1, 0),
        )
        self.system.AddLink(yaw)
        self.constraints.append(yaw)
        self.wheels, self.suspensions, self.tires, self.functors = [], [], [], []
        for i, (x, y) in enumerate(self.xy):
            wheel = ch.ChBody()
            wheel.SetName(("FL", "FR", "RL", "RR")[i])
            wheel.SetMass(float(self.mu[i]))
            wheel.SetInertiaXX(ch.ChVector3d(1, 1, 1))  # all rotations constrained
            wheel.SetPos(ch.ChVector3d(x, y, self.wheel_z0[i]))
            self.system.AddBody(wheel)
            guide = ch.ChLinkMateGeneric(True, True, False, True, True, True)
            guide.Initialize(wheel, self.ground, ch.ChFramed(wheel.GetPos()))
            self.system.AddLink(guide)
            self.constraints.append(guide)
            self.wheels.append(wheel)
            suspension = ch.ChLinkTSDA()
            suspension.Initialize(
                self.chassis, wheel, True, self.top_local[i], ch.ChVector3d(0, 0, 0)
            )
            spring_functor = SuspensionForce(self, i)
            suspension.RegisterForceFunctor(spring_functor)
            self.system.AddLink(suspension)
            self.suspensions.append(suspension)
            tire = ch.ChLinkTSDA()
            tire.Initialize(wheel, self.ground, False, wheel.GetPos(), ch.ChVector3d(x, y, 0))
            tire_functor = TireForce(self, i)
            tire.RegisterForceFunctor(tire_functor)
            self.system.AddLink(tire)
            self.tires.append(tire)
            self.functors.extend([spring_functor, tire_functor])
        self.system.Setup()
        self.mechanical_dof = int(
            self.system.GetNumCoordsVelLevel() - self.system.GetNumConstraints()
        )
        if self.mechanical_dof != 7:
            raise RuntimeError(f"Unexpected mechanical DOF count: {self.mechanical_dof}")

    def road_sample(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        if self._road_cache_time != time:
            self._road_cache_time = time
            self._road_cache_value = self.road.sample(time)
        return self._road_cache_value

    @property
    def time(self) -> float:
        return float(self.system.GetChTime())

    def coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        c = self.chassis
        quaternion = c.GetRot()
        w, x, y, z = quaternion.e0, quaternion.e1, quaternion.e2, quaternion.e3
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        # At yaw=0: omega_world=(roll_dot*cos(pitch), pitch_dot,
        #                     -roll_dot*sin(pitch)).
        omega = c.GetAngVelParent()
        roll_dot = omega.x * np.cos(pitch) - omega.z * np.sin(pitch)
        q = np.array(
            [c.GetPos().z - self.body_z0, roll, pitch]
            + [body.GetPos().z - z0 for body, z0 in zip(self.wheels, self.wheel_z0, strict=True)]
        )
        qd = np.array([c.GetPosDt().z, roll_dot, omega.y] + [b.GetPosDt().z for b in self.wheels])
        return q, qd

    def step(self, command_n: np.ndarray) -> None:
        a = self.config["actuator"]
        desired = np.clip(command_n, -a["force_limit_n"], a["force_limit_n"])
        self.delay_buffer.append(desired.copy())
        delayed = self.delay_buffer.popleft()
        velocity = np.array([s.GetVelocity() for s in self.suspensions])
        power_bound = a["power_limit_w"] / np.maximum(np.abs(velocity), 1e-6)
        # Derate the target instead of instantly clipping the actual force;
        # the latter would violate the slew limit following a velocity jump.
        delayed = np.clip(delayed, -power_bound, power_bound)
        delta = (delayed - self.actuator_force) * (-np.expm1(-self.dt / a["time_constant_s"]))
        self.actuator_force += np.clip(
            delta, -a["rate_limit_nps"] * self.dt, a["rate_limit_nps"] * self.dt
        )
        self.system.DoStepDynamics(self.dt)

    def metrics(self) -> dict:
        q, _ = self.coordinates()
        travel = np.array([s.GetLength() - s.GetRestLength() for s in self.suspensions])
        rates = np.array([s.GetVelocity() for s in self.suspensions])
        tire_load = np.array(
            [
                f.evaluate(self.time, 0, tire.GetLength(), tire.GetVelocity(), tire)
                for f, tire in zip(self.functors[1::2], self.tires, strict=True)
            ]
        )
        return {
            "body_acc_mps2": float(self.chassis.GetPosDt2().z),
            "roll_rad": float(q[1]),
            "pitch_rad": float(q[2]),
            "suspension_travel_m": travel,
            "suspension_velocity_mps": rates,
            "tire_load_n": tire_load,
            "actuator_force_n": self.actuator_force.copy(),
            "mechanical_power_w": self.actuator_force * rates,
        }

    def export_frame(self) -> dict:
        c, rot = self.chassis, self.chassis.GetRot()
        frame = {
            "time_s": self.time,
            "body_pos": vector(c.GetPos()).tolist(),
            "body_quat": [rot.e0, rot.e1, rot.e2, rot.e3],
            "wheel_pos": [vector(b.GetPos()).tolist() for b in self.wheels],
            "road_m": self.road_sample(self.time)[0].tolist(),
            "suspension_top_pos": [vector(s.GetPoint1Abs()).tolist() for s in self.suspensions],
            "suspension_bottom_pos": [vector(s.GetPoint2Abs()).tolist() for s in self.suspensions],
        }
        frame.update(
            {
                k: value.tolist() if isinstance(value, np.ndarray) else value
                for k, value in self.metrics().items()
            }
        )
        return frame

    def close(self) -> None:
        """Release C++ links and director callbacks before rebuilding an episode."""
        self.system.Clear()
        for functor in self.functors:
            functor.model = None
        self.functors.clear()
        self.suspensions.clear()
        self.tires.clear()
        self.wheels.clear()
        self.constraints.clear()
