"""Checkpoint save / load. Checkpoints hold weights, class names, channel list and the TRAINING
normalisation statistics; a sample of raw training windows is stored beside it for domain analysis."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from config.settings import CHECKPOINT_DIR
from preprocessing.normalization import Normalizer
from .har_cnn import HARNet


@dataclass
class LoadedModel:
    model: HARNet
    classes: List[str]
    channels: List[str]
    normalizer: Normalizer
    meta: dict
    path: Optional[Path] = None
    device: str = "cpu"

    def reference(self):
        """(X_raw, y) sample of training windows in standard units, or None if unavailable."""
        if self.path is None:
            return None
        p = Path(self.path).with_suffix(".reference.npz")
        if not p.exists():
            return None
        d = np.load(p, allow_pickle=False)
        return d["X"], d["y"].astype(object)


def list_checkpoints() -> List[Path]:
    return sorted(CHECKPOINT_DIR.glob("*.pt"))


def save_checkpoint(path: Path, model: HARNet, classes: List[str], channels: List[str], normalizer: Normalizer,
                    meta: Optional[dict] = None, reference_X: Optional[np.ndarray] = None,
                    reference_y: Optional[np.ndarray] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_clean = json.loads(json.dumps(meta or {}, default=float))
    torch.save({"state_dict": model.state_dict(), "config": model.config, "classes": list(classes),
                "channels": list(channels), "normalizer": normalizer.to_dict(), "meta": meta_clean}, path)
    if reference_X is not None:
        np.savez_compressed(path.with_suffix(".reference.npz"), X=reference_X.astype(np.float32),
                            y=np.asarray(reference_y).astype("<U16"))


def load_checkpoint(path, device: str = "cpu") -> LoadedModel:
    path = Path(path)
    # weights_only=True: the file may come from an untrusted source; only tensors/primitives are allowed.
    ck = torch.load(path, map_location=device, weights_only=True)
    cfg = dict(ck["config"])
    cfg["widths"] = tuple(cfg["widths"])
    model = HARNet(**cfg)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return LoadedModel(model=model, classes=list(ck["classes"]), channels=list(ck["channels"]),
                       normalizer=Normalizer.from_dict(ck["normalizer"]), meta=ck.get("meta", {}),
                       path=path, device=device)
