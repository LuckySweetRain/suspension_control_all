"""Optional ctypes bridge for the numerically equivalent compiled ADMM loop."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np


class NativeSolver:
    def __init__(self, a, r):
        self.library = ctypes.CDLL(str(Path(__file__).with_name("qp_native.so")))
        self.function = self.library.suspension_qp
        array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        self.function.argtypes = [ctypes.c_int] * 3 + [ctypes.c_double] * 4 + [array] * 10
        self.function.restype = ctypes.c_int
        self.a = np.ascontiguousarray(a, dtype=np.float64)
        self.at = np.ascontiguousarray(a.T, dtype=np.float64)
        self.r = np.ascontiguousarray(r, dtype=np.float64)

    def solve(self, f, lo, hi, x, z, y, opt):
        stats = np.zeros(2)
        status = self.function(
            self.a.shape[1],
            self.a.shape[0],
            int(opt["max_iter"]),
            opt["rho"],
            opt["sigma"],
            opt["abs_tol"],
            opt["rel_tol"],
            self.a,
            self.at,
            self.r,
            np.ascontiguousarray(f),
            np.ascontiguousarray(lo),
            np.ascontiguousarray(hi),
            x,
            z,
            y,
            stats,
        )
        if status < 0:
            raise ValueError("Invalid native solver dimensions")
        return int(stats[0]), bool(stats[1])
