"""Synthetic signal generators used ONLY by unit tests (never by the application)."""
import numpy as np

_G = {"walking": (0.0, 1.0, 0.0), "sitting": (0.0, 0.6, 0.8), "standing": (0.0, 1.0, 0.05)}
_F = {"walking": 2.0, "sitting": 0.0, "standing": 0.0}


def acc_signal(activity, n, fs, rng, scale=1.0):
    t = np.arange(n) / fs
    g = np.array(_G[activity])
    amp = 0.0 if _F[activity] == 0 else 0.4
    osc = amp * np.sin(2 * np.pi * _F[activity] * t)
    sig = g[None, :] + np.stack([osc, osc * 0.5, osc * 0.2], 1) + rng.normal(0, 0.01, (n, 3))
    return t, sig * scale


def gyro_signal(n, rng):
    return rng.normal(0, 0.05, (n, 3))
