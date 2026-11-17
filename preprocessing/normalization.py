"""Channel-wise z-score normalisation. Statistics are fitted on TRAINING data only and stored
inside the model checkpoint, then re-used unchanged at inference time."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray
    channels: List[str]

    @classmethod
    def fit(cls, X: np.ndarray, channels: List[str]) -> "Normalizer":
        flat = X.reshape(-1, X.shape[-1]).astype(np.float64)
        std = flat.std(axis=0)
        std[std < 1e-8] = 1.0
        return cls(flat.mean(axis=0), std, list(channels))

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.shape[-1] != len(self.mean):
            raise ValueError(f"Normalizer expects {len(self.mean)} channels, got {X.shape[-1]}.")
        return ((X - self.mean) / self.std).astype(np.float32)

    def inverse(self, Xn: np.ndarray) -> np.ndarray:
        return (Xn * self.std + self.mean).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": [float(x) for x in self.mean], "std": [float(x) for x in self.std],
                "channels": list(self.channels)}

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        return cls(np.asarray(d["mean"], float), np.asarray(d["std"], float), list(d["channels"]))
