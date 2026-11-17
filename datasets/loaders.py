"""Dataset loaders used for TRAINING. Place raw datasets under datasets/ (see datasets/README.md).

* UCI HAR  : the official 'Inertial Signals' files (already 50 Hz, 128-sample windows).
* CSV spec : any set of CSV files processed with the same preprocessing pipeline as the app.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from preprocessing.column_mapper import ColumnMapping
from preprocessing.pipeline import process_dataframe
from preprocessing.reader import read_csv_bytes

UCI_LABELS = {1: "walking", 2: "upstairs", 3: "downstairs", 4: "sitting", 5: "standing", 6: "lying"}


def _read_signal(path: Path) -> np.ndarray:
    return pd.read_csv(path, sep=r"\s+", header=None, engine="python").to_numpy(dtype=np.float32)


def load_uci_har(root, use_gyro: bool = False) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Returns {'train': (X, y, subject), 'test': (X, y, subject)}; X: (N,128,C) in g (+ rad/s)."""
    root = Path(root)
    if not (root / "train").exists():
        cand = list(root.glob("*/train"))
        if cand:
            root = cand[0].parent
    out = {}
    for split in ("train", "test"):
        base = root / split
        sig = base / "Inertial Signals"
        if not sig.exists():
            raise FileNotFoundError(f"{sig} not found. Download UCI HAR Dataset and point --uci-har-dir at it.")
        names = [f"total_acc_{a}_{split}.txt" for a in "xyz"]
        if use_gyro:
            names += [f"body_gyro_{a}_{split}.txt" for a in "xyz"]
        X = np.stack([_read_signal(sig / n) for n in names], axis=-1)
        y = pd.read_csv(base / f"y_{split}.txt", header=None)[0].map(UCI_LABELS).to_numpy(dtype=object)
        subj = pd.read_csv(base / f"subject_{split}.txt", header=None)[0].astype(str).to_numpy(dtype=object)
        out[split] = (X, y, np.array([f"uci:{s}" for s in subj], dtype=object))
    return out


def load_csv_spec(spec_path, channels: List[str]):
    """spec.json: list of {"path","name","source","mapping":{...ColumnMapping fields...},"label_map":{raw:canonical}}.
    Returns (X, y, subject) with subjects prefixed by dataset name to avoid collisions."""
    spec = json.loads(Path(spec_path).read_text())
    Xs, ys, ss = [], [], []
    for item in spec:
        src = item.get("source", "Custom")
        df = read_csv_bytes(Path(item["path"]).read_bytes(), source=src)
        pf = process_dataframe(df, ColumnMapping.from_dict(item["mapping"]), item.get("name", Path(item["path"]).name),
                               src, label_map=item.get("label_map"))
        ok = np.array([v is not None for v in pf.windows.y])
        if ok.sum() == 0:
            raise ValueError(f"{item['path']}: no labelled windows after label mapping.")
        Xs.append(pf.windows.select(channels)[ok])
        ys.append(pf.windows.y[ok])
        ss.append(np.array([f"{pf.name}:{s}" for s in pf.windows.subject[ok]], dtype=object))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ss)
