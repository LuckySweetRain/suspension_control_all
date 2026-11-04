"""Check hidden-state isolation, paired noise, and causal sensor history."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.cosim_observation import SensorObservation  # noqa: E402


def main():
    a = SensorObservation(dict(mode="measured", noise_seed=7))
    b = SensorObservation(dict(mode="estimated", noise_seed=7))
    isolated = SensorObservation(dict(mode="estimated", noise_seed=7))
    for i in range(100):
        y = np.zeros(77)
        y[40], y[42], y[76] = 0.03, 2, 40
        y[44:48] = i * np.array([0.1, 0.2, 0.1, 0.2])
        request = dict(exports=y, time=i * 0.01)
        oa, ob = a(request, [10, 20]), b(request, [10, 20])
        altered = y.copy()
        altered[39] = 10000  # Absolute body velocity must not leak.
        altered[48:56] = 10000  # True relative velocities and tire loads.
        oi = isolated(dict(exports=altered, time=i * 0.01), [10, 20])
        assert oa.shape == ob.shape == (100,)
        np.testing.assert_array_equal(ob, oi)
        np.testing.assert_array_equal(ob, b(request, [10, 20]))
        np.testing.assert_array_equal(oa[:96].reshape(8, 12)[:, :9], ob[:96].reshape(8, 12)[:, :9])
        assert not oa[:96].reshape(8, 12)[:, 9:].any()
        assert not oa[96:].any() and not ob[96:].any()
    assert abs(ob[93:96]).min() > 0
    print("PASS: hidden-state isolation, causal history, paired noise, estimate activation")


if __name__ == "__main__":
    main()
