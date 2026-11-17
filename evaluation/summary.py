"""Descriptive summaries of window-level predictions (no ground truth required)."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from config.settings import STRIDE, TARGET_FS


def smooth_probs(probs: np.ndarray, segment: np.ndarray, k: int) -> np.ndarray:
    """Moving average of class probabilities along time WITHIN each segment (display-only smoothing)."""
    if k <= 1:
        return probs
    out = probs.copy()
    for s in np.unique(segment):
        m = segment == s
        out[m] = uniform_filter1d(probs[m], size=k, axis=0, mode="nearest")
    return out


def activity_summary(pred_labels: np.ndarray, classes: List[str], stride: int = STRIDE, fs: float = TARGET_FS) -> pd.DataFrame:
    """Share of windows per predicted activity. `approx_seconds` = n_windows * stride/fs
    (windows overlap by 50 %, so this is the non-overlapping equivalent)."""
    n = len(pred_labels)
    counts = pd.Series(pred_labels).value_counts()
    rows = [{"activity": c, "windows": int(counts.get(c, 0)),
             "percent_of_windows": 100.0 * counts.get(c, 0) / max(n, 1),
             "approx_seconds": counts.get(c, 0) * stride / fs} for c in classes]
    return pd.DataFrame(rows).sort_values("windows", ascending=False).reset_index(drop=True)


def transition_timeline(pred_labels, conf, start_idx, segment, subject, window: int = 128, fs: float = TARGET_FS) -> pd.DataFrame:
    """Contiguous runs of the same predicted activity within a segment."""
    rows = []
    n = len(pred_labels)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and segment[j + 1] == segment[i] and pred_labels[j + 1] == pred_labels[i]:
            j += 1
        rows.append({"subject": subject[i], "segment": int(segment[i]), "activity": pred_labels[i],
                     "start_s": start_idx[i] / fs, "end_s": (start_idx[j] + window) / fs,
                     "n_windows": j - i + 1, "mean_confidence": float(np.mean(conf[i:j + 1]))})
        i = j + 1
    return pd.DataFrame(rows)
