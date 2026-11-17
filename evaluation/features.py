"""Hand-crafted window features (used for PCA / t-SNE / UMAP / MMD in a model-independent space)."""
from __future__ import annotations

import numpy as np


def feature_names(channels):
    base = ["mean", "std", "min", "max", "energy", "dom_freq"]
    return [f"{c}_{b}" for b in base for c in channels] + ["acc_mag_mean", "acc_mag_std"]


def window_features(X: np.ndarray, fs: float = 50.0) -> np.ndarray:
    """X: (N, T, C) in standard units (first three channels = accelerometer)."""
    X = np.asarray(X, dtype=np.float64)
    N, T, C = X.shape
    mean, std = X.mean(1), X.std(1)
    mn, mx, energy = X.min(1), X.max(1), (X ** 2).mean(1)
    spec = np.abs(np.fft.rfft(X - mean[:, None, :], axis=1))
    freqs = np.fft.rfftfreq(T, 1.0 / fs)
    dom = freqs[np.argmax(spec[:, 1:, :], axis=1) + 1]
    mag = np.linalg.norm(X[:, :, :3], axis=2)
    return np.hstack([mean, std, mn, mx, energy, dom, mag.mean(1, keepdims=True), mag.std(1, keepdims=True)])
