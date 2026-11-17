"""Unit heuristics. Internal standard: accelerometer in g, gyroscope in rad/s."""
from __future__ import annotations

import numpy as np


def guess_acc_unit(v: np.ndarray) -> str:
    """Guess accelerometer unit from the 99th percentile of |a|. The user can always override."""
    mag = np.linalg.norm(v, axis=1)
    p99 = float(np.percentile(mag, 99)) if len(mag) else 1.0
    if p99 > 300:
        return "mg"
    if p99 > 6:
        return "m/s2"
    return "g"


def guess_gyro_unit(v: np.ndarray) -> str:
    """Gyroscope in deg/s typically reaches tens-hundreds; in rad/s rarely exceeds ~10."""
    p99 = float(np.percentile(np.abs(v), 99)) if v.size else 1.0
    return "deg/s" if p99 > 20 else "rad/s"
