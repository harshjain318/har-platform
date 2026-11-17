"""Training strategies compared in the platform.

scratch            – new randomly initialised model trained on the labelled target subset only
source_only        – pretrained source model evaluated as-is (no target data used)
frozen_extractor   – pretrained; every layer frozen except the classification head
full_finetune      – pretrained; all layers trained with one learning rate
adaptive_layerwise – pretrained; per-layer learning rate and L2-SP anchor strength driven by the
                     measured source→target feature discrepancy (MMD) of each layer.
                     (This is a concrete, documented implementation of "adaptive layer-wise
                     adaptation"; swap in your own method by editing `_adaptive_layerwise`.)
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from evaluation.domain import mmd_test
from models.har_cnn import HARNet, count_parameters
from models.registry import LoadedModel
from models.trainer import forward_numpy, softmax_np, timed_forward, train_model
from preprocessing.normalization import Normalizer

STRATEGIES = ["scratch", "source_only", "frozen_extractor", "full_finetune", "adaptive_layerwise"]
STRATEGY_LABELS = {
    "scratch": "Trained from scratch", "source_only": "Source-only pretrained",
    "frozen_extractor": "Frozen feature extractor", "full_finetune": "Full fine-tuning",
    "adaptive_layerwise": "Adaptive layer-wise adaptation",
}


@dataclass
class AdaptConfig:
    epochs: int = 40
    batch_size: int = 32
    lr_scratch: float = 1e-3
    lr_finetune: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 8
    l2sp_lambda: float = 0.1       # base strength of the L2-SP anchor (adaptive strategy)
    min_lr_scale: float = 0.1      # lowest per-layer lr multiplier (adaptive strategy)
    device: str = "cpu"


@dataclass
class FittedModel:
    name: str
    model: HARNet
    classes: List[str]
    normalizer: Normalizer
    trainable_params: int
    train_seconds: float
    info: dict = field(default_factory=dict)


def _idx(labels: Sequence, classes: List[str]) -> np.ndarray:
    return np.array([classes.index(l) for l in labels], dtype=np.int64)


def _freeze_all_but_head(model: HARNet) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True


def layer_discrepancy(source: LoadedModel, model: HARNet, X_src_raw: np.ndarray, X_tgt_raw: np.ndarray,
                      device: str) -> Dict[str, float]:
    """Per-layer unbiased MMD² between source reference windows and target training windows."""
    _, fs = forward_numpy(model, source.normalizer.transform(X_src_raw), device, return_layers=True)
    _, ft = forward_numpy(model, source.normalizer.transform(X_tgt_raw), device, return_layers=True)
    out = {}
    for k in fs:
        r = mmd_test(fs[k], ft[k], n_perm=0, max_n=300)
        out[k] = max(0.0, r["mmd2"]) if np.isfinite(r["mmd2"]) else 0.0
    return out


def _adaptive_layerwise(model, source, Xtr_n, ytr_i, Xva_n, yva_i, X_src_raw, Xtr_raw, cfg, seed):
    if X_src_raw is not None and len(Xtr_raw) >= 8 and len(X_src_raw) >= 8:
        d = layer_discrepancy(source, model, X_src_raw, Xtr_raw, cfg.device)
        mx = max(d.values()) if d else 0.0
        w = {k: (v / mx if mx > 0 else 0.5) for k, v in d.items()}
        basis = "mmd"
    else:   # too little data to estimate discrepancy reliably -> neutral weights
        w = {k: 0.5 for k in model.layer_names}
        d, basis = {}, "neutral (n<8 target windows or no source reference)"
    w["head"] = 1.0
    model.to(cfg.device)          # anchors (L2-SP) must live on the same device as the parameters
    groups, l2sp = [], []
    lp = model.layer_parameters()
    for name, params in lp.items():
        scale = cfg.min_lr_scale + (1 - cfg.min_lr_scale) * w[name]
        groups.append({"params": params, "lr": cfg.lr_finetune * scale})
        if name != "head":
            lam = cfg.l2sp_lambda * (1.0 - w[name])
            l2sp += [(p, p.detach().clone(), lam) for p in params]
    log = train_model(model, Xtr_n, ytr_i, Xva_n, yva_i, epochs=cfg.epochs, lr=cfg.lr_finetune,
                      weight_decay=cfg.weight_decay, batch_size=cfg.batch_size, param_groups=groups, l2sp=l2sp,
                      seed=seed, device=cfg.device, patience=cfg.patience)
    return log, {"layer_mmd2": d, "layer_weights": w, "weight_basis": basis}


def fit_strategy(name: str, *, source: Optional[LoadedModel], Xtr: np.ndarray, ytr: np.ndarray,
                 Xva: Optional[np.ndarray], yva: Optional[np.ndarray], common: List[str], cfg: AdaptConfig,
                 seed: int, X_src_raw: Optional[np.ndarray] = None) -> FittedModel:
    """Xtr/Xva: RAW windows (standard units, channels already matching the source model). ytr/yva: label strings.
    Normalisation: pretrained strategies keep the SOURCE statistics; scratch fits its own on Xtr only."""
    t0 = time.perf_counter()
    info: dict = {}
    if name == "scratch":
        arch = dict(source.model.config) if source is not None else {}
        arch.update(in_channels=Xtr.shape[-1], n_classes=len(common))
        arch["widths"] = tuple(arch.get("widths", (64, 128, 128)))
        model = HARNet(**arch)
        norm = Normalizer.fit(Xtr, [f"ch{i}" for i in range(Xtr.shape[-1])]) if len(Xtr) else \
            Normalizer(np.zeros(Xtr.shape[-1]), np.ones(Xtr.shape[-1]), [])
        classes = list(common)
        log = train_model(model, norm.transform(Xtr), _idx(ytr, classes),
                          None if Xva is None or len(Xva) == 0 else norm.transform(Xva),
                          None if Xva is None or len(Xva) == 0 else _idx(yva, classes),
                          epochs=cfg.epochs, lr=cfg.lr_scratch, weight_decay=cfg.weight_decay,
                          batch_size=cfg.batch_size, seed=seed, device=cfg.device, patience=cfg.patience)
    else:
        if source is None:
            raise ValueError(f"Strategy '{name}' needs a pretrained source model.")
        model = copy.deepcopy(source.model)
        norm, classes = source.normalizer, list(source.classes)
        miss = [c for c in common if c not in classes]
        if miss:
            raise ValueError(f"Common activities {miss} are not outputs of the source model {classes}.")
        Xtr_n = norm.transform(Xtr)
        have_val = Xva is not None and len(Xva) > 0
        Xva_n = norm.transform(Xva) if have_val else None
        ytr_i = _idx(ytr, classes)
        yva_i = _idx(yva, classes) if have_val else None
        if name == "source_only":
            for p in model.parameters():
                p.requires_grad = False
            log = {"epochs_run": 0}
        elif name == "frozen_extractor":
            _freeze_all_but_head(model)
            log = train_model(model, Xtr_n, ytr_i, Xva_n, yva_i, epochs=cfg.epochs, lr=cfg.lr_finetune,
                              weight_decay=cfg.weight_decay, batch_size=cfg.batch_size, backbone_eval=True,
                              seed=seed, device=cfg.device, patience=cfg.patience)
        elif name == "full_finetune":
            log = train_model(model, Xtr_n, ytr_i, Xva_n, yva_i, epochs=cfg.epochs, lr=cfg.lr_finetune,
                              weight_decay=cfg.weight_decay, batch_size=cfg.batch_size, seed=seed,
                              device=cfg.device, patience=cfg.patience)
        elif name == "adaptive_layerwise":
            log, extra = _adaptive_layerwise(model, source, Xtr_n, ytr_i, Xva_n, yva_i, X_src_raw, Xtr, cfg, seed)
            info.update(extra)
        else:
            raise ValueError(f"Unknown strategy '{name}'.")
    info["epochs_run"] = log.get("epochs_run", 0)
    return FittedModel(name, model, classes, norm, count_parameters(model, trainable_only=True),
                       time.perf_counter() - t0, info)


def predict_common(fm: FittedModel, X: np.ndarray, common: List[str], device: str = "cpu"):
    """Probabilities restricted to the common activities (softmax over the allowed logits only)."""
    allowed = [fm.classes.index(c) for c in common]
    logits, ms = timed_forward(fm.model, fm.normalizer.transform(X), device)
    probs = softmax_np(logits, allowed)
    return probs, np.array(common, dtype=object)[probs.argmax(1)], ms
