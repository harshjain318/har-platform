"""Window-level prediction for processed files."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from config.settings import LOW_CONFIDENCE, TARGET_FS
from evaluation.metrics import evaluate_predictions, expected_calibration_error
from preprocessing.pipeline import ProcessedFile
from .registry import LoadedModel
from .trainer import forward_numpy, softmax_np, timed_forward


@dataclass
class Prediction:
    file: str
    classes: List[str]
    probs: np.ndarray            # (N, K) softmax outputs (NOT calibrated)
    pred_idx: np.ndarray
    conf: np.ndarray             # max probability
    entropy: np.ndarray
    y_true: np.ndarray           # object array, canonical label or None
    subject: np.ndarray
    segment: np.ndarray
    start: np.ndarray            # start index (samples) in the file's concatenated signal
    inference_ms_per_window: float
    fs: float = TARGET_FS

    @property
    def pred(self) -> np.ndarray:
        return np.array(self.classes, dtype=object)[self.pred_idx]

    @property
    def start_s(self) -> np.ndarray:
        return self.start / self.fs

    def low_conf_mask(self, thr: float = LOW_CONFIDENCE) -> np.ndarray:
        return self.conf < thr

    def evaluate(self) -> Optional[Dict]:
        res = evaluate_predictions(self.y_true, self.pred, self.classes)
        if res is not None:
            valid = np.array([(t is not None) and (t in self.classes) for t in self.y_true])
            correct = (self.y_true[valid] == self.pred[valid]).astype(float)
            res["ece"] = expected_calibration_error(self.conf[valid], correct)
        return res


def validate_windows(X: np.ndarray, n_channels: int, window: int = 128) -> None:
    if X.ndim != 3 or X.shape[1] != window or X.shape[2] != n_channels:
        raise ValueError(f"Window tensor has shape {X.shape}; expected (N, {window}, {n_channels}).")
    if not np.isfinite(X).all():
        raise ValueError("Windows contain NaN/inf values.")


def predict_processed(loaded: LoadedModel, pf: ProcessedFile) -> Prediction:
    if pf.windows.n == 0:
        raise ValueError(f"'{pf.name}' produced no windows (signal too short?).")
    X = pf.windows.select(loaded.channels)
    validate_windows(X, len(loaded.channels))
    Xn = loaded.normalizer.transform(X)
    logits, ms = timed_forward(loaded.model, Xn, loaded.device)
    probs = softmax_np(logits)
    idx = probs.argmax(1)
    ent = -(probs * np.log(probs + 1e-12)).sum(1)
    return Prediction(file=pf.name, classes=loaded.classes, probs=probs, pred_idx=idx, conf=probs.max(1),
                      entropy=ent, y_true=pf.windows.y, subject=pf.windows.subject, segment=pf.windows.segment,
                      start=pf.windows.start, inference_ms_per_window=ms, fs=pf.fs)


def extract_layer_features(loaded: LoadedModel, X_raw: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-layer pooled features for raw (un-normalised) windows already channel-selected."""
    _, layers = forward_numpy(loaded.model, loaded.normalizer.transform(X_raw), loaded.device, return_layers=True)
    return layers
