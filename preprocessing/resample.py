"""Resampling of irregular / differently-sampled signals onto a uniform 50 Hz grid."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def _interp(grid: np.ndarray, t: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(grid, t, v[:, c]) for c in range(v.shape[1])])


def resample_to_grid(t: np.ndarray, v: np.ndarray, grid: np.ndarray, fs_out: float) -> np.ndarray:
    """Linear interpolation onto `grid`. When the source rate is clearly higher than fs_out the
    signal is first low-pass filtered (zero-phase Butterworth, 0.4*fs_out) to avoid aliasing."""
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    if len(t) < 2:
        return np.repeat(v[:1], len(grid), axis=0)
    fs_in = 1.0 / float(np.median(np.diff(t)))
    if fs_in > 1.05 * fs_out and len(t) > 40:
        n_in = int(np.floor((t[-1] - t[0]) * fs_in)) + 1
        g_in = t[0] + np.arange(n_in) / fs_in
        u = _interp(g_in, t, v)
        wn = (0.4 * fs_out) / (fs_in / 2.0)
        if wn < 1.0:
            sos = butter(4, wn, output="sos")
            try:
                u = sosfiltfilt(sos, u, axis=0)
            except ValueError:   # segment too short for the filter's padding
                pass
        return _interp(grid, g_in, u)
    return _interp(grid, t, v)
