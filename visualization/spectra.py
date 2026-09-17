from __future__ import annotations

import numpy as np


def sampling_frequency(times: np.ndarray) -> float:
    """Infer Hz from finite, strictly increasing, uniformly sampled timestamps."""
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(times)):
        raise ValueError("PSD requires at least two finite timestamps")
    intervals = np.diff(times)
    dt = float(np.mean(intervals))
    if dt <= 0.0 or not np.allclose(intervals, dt, rtol=1e-6, atol=1e-12):
        raise ValueError("PSD timestamps must be strictly increasing and uniformly spaced")
    return 1.0 / dt


def one_sided_psd(times: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a demeaned periodogram whose integral equals the signal variance."""
    fs = sampling_frequency(times)
    values = np.asarray(values, dtype=np.float64)
    if values.shape != np.shape(times) or not np.all(np.isfinite(values)):
        raise ValueError("PSD values must be finite and match the timestamp shape")
    values = values - np.mean(values)
    spectrum = np.fft.rfft(values)
    power = np.abs(spectrum) ** 2 / (fs * values.size)
    if values.size % 2 == 0:
        power[1:-1] *= 2.0
    else:
        power[1:] *= 2.0
    return np.fft.rfftfreq(values.size, 1.0 / fs), power
