"""Inverse-tanh observation coordinates, not a physical state safety filter."""

from __future__ import annotations

import numpy as np


def inverse_tanh_state(value, lower, upper, epsilon=1e-4):
    """Map an open physical interval to R; report every numerical saturation.

    Outside/boundary values are saturated only for network input. The caller must
    keep original states for dynamics, rewards and independent safety metrics.
    """
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("State mapping requires finite measurements")
    if not np.isfinite([lower, upper, epsilon]).all() or not lower < upper:
        raise ValueError("State mapping requires finite ordered scalar bounds")
    if not 0 < epsilon < 1:
        raise ValueError("State mapping epsilon must lie in (0, 1)")
    ratio = 2 * (value - lower) / (upper - lower) - 1
    saturated = np.clip(ratio, -1 + epsilon, 1 - epsilon)
    return np.arctanh(saturated), {
        "outside_bounds": int(np.count_nonzero((value < lower) | (value > upper))),
        "at_or_outside_bounds": int(np.count_nonzero((value <= lower) | (value >= upper))),
        "numerical_saturations": int(np.count_nonzero(ratio != saturated)),
    }
