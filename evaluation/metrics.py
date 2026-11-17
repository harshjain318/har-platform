"""Ground-truth evaluation. Only windows whose true label is a class known to the model are scored."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def evaluate_predictions(y_true: Sequence, y_pred: Sequence, model_classes: List[str]) -> Optional[Dict]:
    """Return None when no window has a usable ground-truth label.

    * Windows whose label is None / not in `model_classes` are excluded (and counted).
    * Macro-F1 is averaged over the classes PRESENT in the ground truth.
    * per-class accuracy == recall of that class.
    """
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    valid = np.array([(t is not None) and (t in model_classes) for t in y_true])
    if valid.sum() == 0:
        return None
    yt, yp = y_true[valid].astype(str), y_pred[valid].astype(str)
    true_labels = [c for c in model_classes if c in set(yt)]
    cm_labels = [c for c in model_classes if c in set(yt) | set(yp)]
    p, r, f, s = precision_recall_fscore_support(yt, yp, labels=true_labels, zero_division=0)
    per_class = pd.DataFrame({"class": true_labels, "precision": p, "recall (per-class accuracy)": r,
                              "f1": f, "support": s})
    excluded = pd.Series([str(t) for t in y_true[~valid]]).value_counts().to_dict()
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, labels=true_labels, average="macro", zero_division=0)),
        "n_evaluated": int(valid.sum()), "n_excluded": int((~valid).sum()), "excluded_labels": excluded,
        "per_class": per_class, "cm_labels": cm_labels,
        "confusion_matrix": confusion_matrix(yt, yp, labels=cm_labels),
        "true_labels_present": true_labels,
    }


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """Top-label ECE. Only meaningful when ground truth exists; reported as a diagnostic."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)
