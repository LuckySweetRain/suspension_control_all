"""Passive, active skyhook and a linear full-state LQR reference."""

from __future__ import annotations

import numpy as np

from .constitutive import damper_tangent


def linear_matrices(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Static tangent x=[q(7), qdot(7)], four suspension force inputs [N].

    The stiffness includes the geometric effect of the static TSDA preload.
    With asymmetric compression/rebound damping, the velocity derivative is
    not unique at rest; use the mean of the two one-sided slopes for LQR.
    Actuator delay, lag and saturation remain outside this mechanical model.
    """
    v = config["vehicle"]
    x = np.array([v["front_distance_m"]] * 2 + [-v["rear_distance_m"]] * 2)
    y = np.array(
        [
            v["front_half_track_m"],
            -v["front_half_track_m"],
            v["rear_half_track_m"],
            -v["rear_half_track_m"],
        ]
    )
    jac = np.column_stack([np.ones(4), y, -x, -np.eye(4)])
    mass = np.array([v["sprung_mass_kg"], *v["inertia_kgm2"][:2], *v["unsprung_mass_kg"]])
    stiffness = jac.T @ np.diag(v["suspension_stiffness_npm"]) @ jac
    for axle, half_track in enumerate((v["front_half_track_m"], v["rear_half_track_m"])):
        difference = jac[2 * axle] - jac[2 * axle + 1]
        cross_rate = v["anti_roll_stiffness_nmprad"][axle] / (2 * half_track) ** 2
        stiffness += cross_rate * np.outer(difference, difference)
    # K = J.T diag(k) J - sum_i preload_i * Hessian(length_i).
    # At equilibrium each TSDA is vertical. For the yaw-free Ry(pitch)Rx(roll)
    # geometry, only the roll/roll and pitch/pitch Hessian entries are nonzero:
    # d2(length)/d(angle)2 = -h + h**2 / length, h = upper mount local z.
    mount_height = v["suspension_mount_height_m"]
    gravity = config["physics"]["gravity_mps2"]
    wheel_mass = np.asarray(v["unsprung_mass_kg"], dtype=float)
    preload = (
        v["sprung_mass_kg"]
        * gravity
        / (2 * (v["front_distance_m"] + v["rear_distance_m"]))
        * np.array([v["rear_distance_m"]] * 2 + [v["front_distance_m"]] * 2)
    )
    wheel_height = v["tire_radius_m"] - (preload + wheel_mass * gravity) / np.asarray(
        v["tire_stiffness_npm"]
    )
    installed_length = v["body_height_m"] + mount_height - wheel_height
    geometric_stiffness = np.sum(preload * (-mount_height + mount_height**2 / installed_length))
    stiffness[1, 1] -= geometric_stiffness
    stiffness[2, 2] -= geometric_stiffness
    damping = jac.T @ np.diag(damper_tangent(v)) @ jac
    stiffness[3:, 3:] += np.diag(v["tire_stiffness_npm"])
    damping[3:, 3:] += np.diag(v["tire_damping_nspm"])
    a = np.block(
        [[np.zeros((7, 7)), np.eye(7)], [-stiffness / mass[:, None], -damping / mass[:, None]]]
    )
    b = np.vstack([np.zeros((7, 4)), jac.T / mass[:, None]])
    return a, b, jac


def lqr_gain(config: dict) -> np.ndarray:
    from scipy.linalg import solve_continuous_are

    a, b, _ = linear_matrices(config)
    q = np.diag([1e3, 1e4, 1e4, 1e3, 1e3, 1e3, 1e3, 3e4, 5e4, 5e4, 1e2, 1e2, 1e2, 1e2])
    r = np.eye(4) * 0.001
    p = solve_continuous_are(a, b, q, r)
    return np.linalg.solve(r, b.T @ p)
