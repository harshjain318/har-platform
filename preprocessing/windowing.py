"""Sliding-window segmentation. Windows never cross segment boundaries."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def sliding_windows(a: np.ndarray, window: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """a: (n, C) -> windows (k, window, C) and start offsets (k,). Empty when n < window."""
    n, c = a.shape
    if n < window:
        return np.zeros((0, window, c), dtype=np.float32), np.zeros(0, dtype=int)
    starts = np.arange(0, n - window + 1, stride)
    idx = starts[:, None] + np.arange(window)[None, :]
    return a[idx].astype(np.float32), starts
