"""Wheel-coordinate force tables in SI units, with linear end extrapolation.

Positive displacement and velocity mean suspension compression (jounce).
Positive force pushes the chassis away from the wheel. These curves preserve
source data nodes but do not reproduce CarSim's proprietary spline algorithm.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite

import numpy as np


class PiecewiseLinearCurve:
    """Compile a force table once for use inside a physics force callback.

    ``axis`` names either ``velocity_mps`` or ``compression_m``; ordinate
    ``force_n`` is always force at the wheel, including any motion-ratio mapping.
    End slopes are extended without clipping the input or capping the force.
    """

    def __init__(self, table: Mapping, axis: str):
        if not isinstance(table, Mapping):
            raise ValueError("Force table must be a mapping")
        if axis not in table or "force_n" not in table:
            raise ValueError(f"Force table requires {axis} and force_n")
        try:
            self.x = np.array(table[axis], dtype=float, copy=True)
            self.force = np.array(table["force_n"], dtype=float, copy=True)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Force table {axis} and force_n must be numeric arrays") from error
        if self.x.ndim != 1 or self.x.size < 2 or self.force.shape != self.x.shape:
            raise ValueError(
                f"Force table {axis} and force_n need matching 1D arrays of length >= 2"
            )
        if not np.all(np.isfinite(self.x)) or not np.all(np.isfinite(self.force)):
            raise ValueError("Force table nodes must be finite")
        if np.any(np.diff(self.x) <= 0):
            raise ValueError(f"Force table {axis} must be strictly increasing")
        self.slopes = np.diff(self.force) / np.diff(self.x)
        if not np.all(np.isfinite(self.slopes)):
            raise ValueError("Force table slopes must be finite")
        for values in (self.x, self.force, self.slopes):
            values.setflags(write=False)

    def __call__(self, value: float | np.ndarray) -> float | np.ndarray:
        if np.isscalar(value):
            position = float(value)
            if not isfinite(position):
                raise ValueError("Force table query must be finite")
            index = min(
                max(int(np.searchsorted(self.x, position, side="right")) - 1, 0),
                self.slopes.size - 1,
            )
            return float(self.force[index] + self.slopes[index] * (position - self.x[index]))
        position = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(position)):
            raise ValueError("Force table queries must be finite")
        index = np.clip(
            np.searchsorted(self.x, position, side="right") - 1, 0, self.slopes.size - 1
        )
        return self.force[index] + self.slopes[index] * (position - self.x[index])

    def slopes_at(self, value: float = 0.0) -> tuple[float, float]:
        """Return left and right derivatives, including at a node or beyond the table."""
        if not isfinite(value):
            raise ValueError("Force table derivative query must be finite")
        indices = [
            min(max(int(np.searchsorted(self.x, value, side=side)) - 1, 0), self.slopes.size - 1)
            for side in ("left", "right")
        ]
        return float(self.slopes[indices[0]]), float(self.slopes[indices[1]])


def evaluate_curve(table: Mapping, x: float | np.ndarray, axis: str) -> float | np.ndarray:
    """Evaluate a table; use ``PiecewiseLinearCurve`` for repeated physics queries."""
    return PiecewiseLinearCurve(table, axis)(x)


def _curves(vehicle: Mapping, name: str, axis: str) -> list[PiecewiseLinearCurve] | None:
    tables = vehicle.get(name)
    if tables is None:
        return None
    if not isinstance(tables, (list, tuple)) or len(tables) != 4:
        raise ValueError(f"vehicle.{name} requires four tables in FL FR RL RR order")
    curves = []
    for index, table in enumerate(tables):
        try:
            curves.append(PiecewiseLinearCurve(table, axis))
        except ValueError as error:
            raise ValueError(f"vehicle.{name}[{index}]: {error}") from error
    return curves


def validate_force_tables(vehicle: Mapping) -> None:
    """Reject nonpassive curves, unintended static stop forces, and partial stop sets.

    Damper nodes must span both velocity signs and include exactly (0, 0).
    Monotonicity then ensures dissipative interpolation and end extrapolation.
    Each stop must have a zero-force neighborhood around zero jounce and a flat
    inactive end, so extrapolation cannot pull in the wrong travel direction.
    """
    dampers = _curves(vehicle, "damper_tables", "velocity_mps")
    if dampers is not None:
        for index, curve in enumerate(dampers):
            label = f"vehicle.damper_tables[{index}]"
            zero = np.flatnonzero(curve.x == 0)
            if curve.x[0] >= 0 or curve.x[-1] <= 0 or zero.size != 1:
                raise ValueError(f"{label} must span both velocity signs and contain (0, 0)")
            if curve.force[zero[0]] != 0:
                raise ValueError(f"{label} must contain exactly (0, 0)")
            if np.any(curve.slopes < 0) or np.any(curve.x * curve.force < 0):
                raise ValueError(
                    f"{label} must be monotone and dissipative (velocity * force >= 0)"
                )

    jounce = _curves(vehicle, "jounce_stop_tables", "compression_m")
    rebound = _curves(vehicle, "rebound_stop_tables", "compression_m")
    if (jounce is None) != (rebound is None):
        raise ValueError("jounce_stop_tables and rebound_stop_tables must be supplied together")
    for name, curves, sign in (
        ("jounce_stop_tables", jounce, 1),
        ("rebound_stop_tables", rebound, -1),
    ):
        if curves is None:
            continue
        for index, curve in enumerate(curves):
            label = f"vehicle.{name}[{index}]"
            if np.any(curve.slopes < 0) or np.any(sign * curve.force < 0):
                raise ValueError(
                    f"{label} must be monotone and act only in its restoring direction"
                )
            if curve(0.0) != 0 or curve.slopes_at(0.0) != (0.0, 0.0):
                raise ValueError(f"{label} must have a zero-force neighborhood around zero jounce")
            inactive_slope = curve.slopes[0] if sign > 0 else curve.slopes[-1]
            if inactive_slope != 0:
                raise ValueError(f"{label} needs a flat inactive end for physical extrapolation")


def damper_tangent(vehicle: Mapping) -> np.ndarray:
    """Return four wheel damping tangents in N s/m for a linearized controller.

    At a zero-velocity corner, use the arithmetic mean of its two one-sided
    slopes. Without tables, use the mean of the configured compression/rebound
    multipliers applied to ``suspension_damping_nspm``.
    """
    validate_force_tables(vehicle)
    dampers = _curves(vehicle, "damper_tables", "velocity_mps")
    if dampers is not None:
        return np.array([sum(curve.slopes_at()) / 2 for curve in dampers], dtype=float)
    damping = np.asarray(vehicle["suspension_damping_nspm"], dtype=float)
    ratios = np.array(
        [vehicle.get("compression_damping_ratio", 1.0), vehicle.get("rebound_damping_ratio", 1.0)],
        dtype=float,
    )
    if damping.shape != (4,) or not np.all(np.isfinite(damping)) or np.any(damping < 0):
        raise ValueError("suspension_damping_nspm requires four finite nonnegative values")
    if not np.all(np.isfinite(ratios)) or np.any(ratios < 0):
        raise ValueError("Damping ratios must be finite and nonnegative")
    return damping * float(np.mean(ratios))
