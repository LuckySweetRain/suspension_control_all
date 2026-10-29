"""Exercise standstill, jitter, restart and image-preview zero-speed limits.

Run from the repository root; this is a standalone check, not a pytest suite.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from controller import Controller
from depth_preview import DepthPreview


def verify(data, fixture):
    reference = loadmat(fixture, simplify_cells=True)
    controller = Controller(str(data))
    sample = reference["exports"][:, 0].copy()
    controller.step(sample)
    origin = controller.origin
    controller.prev = np.array([1800.0, -1400.0])
    previous = controller.prev.copy()
    sample[38] = 0
    sample[0] += 0.1  # State/camera bias must not actuate a parked vehicle.
    controller.wx.fill(1)
    controller.wz.fill(1)
    controller.wy.fill(1)
    solve = controller.solve

    def forbidden_qp(*_args):
        raise AssertionError("A low-speed QP was solved")

    controller.solve = forbidden_qp
    stop_forces = []
    for k in range(150):
        sample[22:26] = np.array([8.0, -5.0, 3.0, -2.0]) * np.cos(k * 0.1)
        forces, diag, failed = controller.step(sample)
        assert not failed
        actual = np.array(diag[:2])
        np.testing.assert_array_less(np.abs(actual - previous), np.array([40.000001] * 2))
        assert np.all(np.abs(actual) <= np.abs(previous))
        stop_forces.append(actual)
        expected = np.clip(3 * sample[22:26] + 0.5 * actual[[0, 1, 0, 1]], -5000, 5000)
        np.testing.assert_allclose(forces, np.tile(expected, 2), atol=1e-10, rtol=0)
        previous = actual
    np.testing.assert_array_equal(stop_forces[49:], np.zeros((101, 2)))
    assert controller.origin == origin, "Stopping must not relearn the body equilibrium"
    assert not any(np.any(a) for a in [controller.wx, controller.wz, controller.wy])
    for speed in [0.21, 0.49, 0.3, 0.48, 0.25]:
        sample[38] = speed * 3.6
        _, diag, _ = controller.step(sample)
        assert diag[:2] == [0, 0] and controller.low_speed

    # Re-start on the original, feasible state; ramp with newly solved QPs.
    controller.solve = solve
    sample = reference["exports"][:, 0].copy()
    gain, previous = [], np.zeros(2)
    for _ in range(55):
        _, diag, failed = controller.step(sample)
        assert not failed
        gain.append(controller.active_gain)
        actual = np.asarray(diag[:2])
        if controller.active_gain < 1:
            assert np.max(np.abs(actual - previous)) <= 40.000001
        previous = actual
    assert gain[0] == 0.02 and gain[49] == 1.0
    assert controller.mode == "active" and not controller.low_speed
    assert np.max(np.abs(previous)) > 1, "Restart did not resume MPC actuation"

    # Total planar motion prevents false standstill during a sideways slide.
    sample[38] = 0
    controller.step(sample, lateral_speed_mps=1.0)
    assert not controller.low_speed
    controller.latched = True
    for speed in [0, 5]:
        sample[38] = speed * 3.6
        _, diag, failed = controller.step(sample)
        assert failed and diag[:2] == [0, 0]
    controller.reset()
    assert controller.origin is None and not controller.latched

    # A measured, non-flat road must remain stationary when actual Vx=Vy=0.
    sample = reference["exports"][:, 0].copy()
    sample[38] = 0
    xx, yy = np.meshgrid(np.arange(0, 14, 0.05), np.arange(-2, 2, 0.05))
    vision = DepthPreview()
    vision.points = np.column_stack((xx.ravel(), yy.ravel(), 0.02 * np.sin(xx.ravel())))
    vision.reference_plane = np.zeros(3)
    vision.initial_road = np.zeros(2)
    vision.observed_time = 0.0
    zero_preview_error = []
    for steering in [None, 0.0, 0.35, -0.35]:
        path = (
            None
            if steering is None
            else dict(pose=[0, 0, 0], steer_rad=steering, lateral_speed_mps=0.0)
        )
        preview = vision.preview(sample, controller.c, 0.01, path)
        w = preview[0].reshape(-1, 4)
        assert preview[2][:2] == [0, 0]
        np.testing.assert_array_equal(w[:, [1, 3]], np.zeros((controller.n, 2)))
        np.testing.assert_array_equal(w, np.tile(w[0], (controller.n, 1)))
        zero_preview_error.append(float(np.max(np.abs(w[:, [1, 3]]))))
    # A finite lateral-only displacement with steered wheels used to divide by zero.
    path["lateral_speed_mps"] = 0.7
    assert np.isfinite(vision.preview(sample, controller.c, 0.02, path)[0]).all()
    static_preview = controller.preview(sample)
    assert static_preview[2][:2] == [0, 0]
    np.testing.assert_array_equal(static_preview[0].reshape(-1, 4)[:, [1, 3]], 0)
    # Low speed must not hide malformed or stale perception inputs.
    invalid = (np.full(4 * controller.n, np.nan), static_preview[1], static_preview[2])
    try:
        controller.step(sample, invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("Standstill accepted a nonfinite preview")
    try:
        vision.preview(sample, controller.c, 0.2, path)
    except ValueError:
        pass
    else:
        raise AssertionError("Standstill accepted stale camera data")
    result = dict(
        passed=True,
        stationary_active_force_after_0_5s_N=np.max(np.abs(stop_forces[49:]), axis=0).tolist(),
        stop_slew_limit_N_per_step=40,
        passive_damping_preserved=True,
        equilibrium_preserved=True,
        hysteresis_and_restart_passed=True,
        fault_latch_preserved=True,
        zero_speed_preview_rate_max_mps=max(zero_preview_error),
        nonfinite_and_stale_preview_rejected=True,
    )
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("results/orin/quick_mpc_data.mat"))
    parser.add_argument("--fixture", type=Path, default=Path("results/orin/parity_fixture.mat"))
    args = parser.parse_args()
    verify(args.data, args.fixture)
