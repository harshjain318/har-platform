"""Pool of labelled windows from several processed files, with group ids for leak-free splitting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from preprocessing.pipeline import ProcessedFile


@dataclass
class WindowPool:
    X: np.ndarray            # (N, T, C) raw standard units
    y: np.ndarray            # (N,) str
    group: np.ndarray        # (N,) "file::subject"
    file: np.ndarray
    seg_key: np.ndarray      # "file::segment"
    pos: np.ndarray          # start index (for temporal ordering)
    dropped: Dict[str, int]

    @property
    def n(self) -> int:
        return len(self.X)


def build_pool(files: Sequence[ProcessedFile], channels: List[str], common: List[str]) -> WindowPool:
    Xs, ys, gs, fs, sk, ps = [], [], [], [], [], []
    dropped: Dict[str, int] = {}
    for pf in files:
        w = pf.windows
        if w.n == 0:
            continue
        keep = np.array([(v is not None) and (v in common) for v in w.y])
        for v in w.y[~keep]:
            k = str(v)
            dropped[k] = dropped.get(k, 0) + 1
        if keep.sum() == 0:
            continue
        Xs.append(w.select(channels)[keep])
        ys.append(w.y[keep].astype(str))
        gs.append(np.array([f"{pf.name}::{s}" for s in w.subject[keep]], dtype=object))
        fs.append(np.full(keep.sum(), pf.name, dtype=object))
        sk.append(np.array([f"{pf.name}::{s}" for s in w.segment[keep]], dtype=object))
        ps.append(w.start[keep])
    if not Xs:
        raise ValueError("No labelled windows with the selected common activities were found.")
    return WindowPool(np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs), np.concatenate(fs),
                      np.concatenate(sk), np.concatenate(ps), dropped)
