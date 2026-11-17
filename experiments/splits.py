"""Leakage-safe splitting. Splits are by GROUP (subject / file+subject), never by window.
When fewer than 3 groups exist, a purged temporal split inside each segment is used instead."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

Split = Tuple[np.ndarray, np.ndarray, np.ndarray]


def group_split(groups: Sequence, seed: int, fractions=(0.6, 0.15, 0.25)) -> Split:
    groups = np.asarray(groups, dtype=object)
    ug = np.array(sorted(set(groups)), dtype=object)
    if len(ug) < 3:
        raise ValueError("Need at least 3 groups for a group split.")
    rng = np.random.default_rng(seed)
    ug = ug[rng.permutation(len(ug))]
    n_te = max(1, int(round(len(ug) * fractions[2])))
    n_va = max(1, int(round(len(ug) * fractions[1])))
    te, va, tr = ug[:n_te], ug[n_te:n_te + n_va], ug[n_te + n_va:]
    if len(tr) == 0:
        raise ValueError("Not enough groups to leave any for training.")
    f = lambda gs: np.flatnonzero(np.isin(groups, gs))
    return f(tr), f(va), f(te)


def group_kfold(groups: Sequence, n_folds: int, seed: int, val_groups: float = 0.2) -> List[Split]:
    groups = np.asarray(groups, dtype=object)
    ug = np.array(sorted(set(groups)), dtype=object)
    if len(ug) < max(n_folds, 3):
        raise ValueError(f"{len(ug)} groups available; need at least {max(n_folds, 3)} for {n_folds}-fold CV.")
    rng = np.random.default_rng(seed)
    ug = ug[rng.permutation(len(ug))]
    folds = np.array_split(ug, n_folds)
    out = []
    for k in range(n_folds):
        test_g = folds[k]
        rest = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        n_va = max(1, int(round(len(rest) * val_groups)))
        va_g, tr_g = rest[:n_va], rest[n_va:]
        f = lambda gs: np.flatnonzero(np.isin(groups, gs))
        out.append((f(tr_g), f(va_g), f(test_g)))
    return out


def temporal_split(segment_key: Sequence, pos: Sequence, fractions=(0.6, 0.15, 0.25), gap: int = 1) -> Split:
    """Per segment: first 60 % train | gap | next 15 % val | gap | rest test. A one-window gap makes
    50 %-overlapping windows from different partitions share no samples."""
    key = np.asarray(segment_key, dtype=object)
    pos = np.asarray(pos)
    tr, va, te = [], [], []
    for k in sorted(set(key)):
        idx = np.flatnonzero(key == k)
        idx = idx[np.argsort(pos[idx])]
        n = len(idx)
        a = int(n * fractions[0])
        b = a + gap + int(n * fractions[1])
        tr.append(idx[:a])
        va.append(idx[a + gap:b])
        te.append(idx[b + gap:])
    cat = lambda xs: np.concatenate(xs) if xs else np.array([], dtype=int)
    return cat(tr), cat(va), cat(te)


def stratified_fraction(idx: np.ndarray, y: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Random class-stratified subsample of `idx`, at least one window per present class."""
    if fraction >= 1.0 or len(idx) == 0:
        return idx
    chosen = []
    for c in sorted(set(y[idx])):
        ci = idx[y[idx] == c]
        k = max(1, int(round(len(ci) * fraction)))
        chosen.append(rng.choice(ci, size=min(k, len(ci)), replace=False))
    return np.sort(np.concatenate(chosen))


def train_val_split(groups: Sequence, seed: int, val_fraction: float = 0.2):
    """Subject-wise two-way split (used when a separate official test set exists)."""
    groups = np.asarray(groups, dtype=object)
    ug = np.array(sorted(set(groups)), dtype=object)
    if len(ug) < 2:
        raise ValueError("Need at least 2 groups.")
    rng = np.random.default_rng(seed)
    ug = ug[rng.permutation(len(ug))]
    n_va = max(1, int(round(len(ug) * val_fraction)))
    f = lambda gs: np.flatnonzero(np.isin(groups, gs))
    return f(ug[n_va:]), f(ug[:n_va])
