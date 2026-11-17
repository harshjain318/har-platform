"""Generic training / inference helpers shared by source training and adaptation strategies."""
from __future__ import annotations

import copy
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def softmax_np(logits: np.ndarray, allowed: Optional[Sequence[int]] = None) -> np.ndarray:
    z = logits if allowed is None else logits[:, list(allowed)]
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@torch.no_grad()
def forward_numpy(model: nn.Module, X: np.ndarray, device: str = "cpu", batch_size: int = 512,
                  return_layers: bool = False):
    """Returns logits (N, K) [and dict of per-layer feature arrays]. Model is put in eval mode."""
    model.eval()
    model.to(device)
    logits, layers = [], {}
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch_size])).float().to(device)
        if return_layers:
            lg, ly = model(xb, return_layers=True)
            for k, v in ly.items():
                layers.setdefault(k, []).append(v.cpu().numpy())
        else:
            lg = model(xb)
        logits.append(lg.cpu().numpy())
    out = np.concatenate(logits) if logits else np.zeros((0, model.head.out_features), dtype=np.float32)
    if return_layers:
        return out, {k: np.concatenate(v) for k, v in layers.items()}
    return out


def timed_forward(model: nn.Module, X: np.ndarray, device: str = "cpu") -> Tuple[np.ndarray, float]:
    """Logits and inference latency in milliseconds per window (after one warm-up pass)."""
    if len(X) == 0:
        return forward_numpy(model, X, device), float("nan")
    forward_numpy(model, X[:min(len(X), 32)], device)
    t0 = time.perf_counter()
    lg = forward_numpy(model, X, device)
    return lg, 1000.0 * (time.perf_counter() - t0) / len(X)


def train_model(model: nn.Module, Xtr: np.ndarray, ytr: np.ndarray, Xva: Optional[np.ndarray] = None,
                yva: Optional[np.ndarray] = None, *, epochs: int = 40, lr: float = 1e-3,
                weight_decay: float = 1e-4, batch_size: int = 64, param_groups: Optional[List[dict]] = None,
                l2sp: Optional[List[Tuple[nn.Parameter, torch.Tensor, float]]] = None,
                freeze_bn: bool = False, backbone_eval: bool = False, seed: int = 0, device: str = "cpu",
                patience: int = 8) -> Dict:
    """Mini-batch AdamW training with early stopping on validation macro-F1 (if validation data given).
    Xtr must already be normalised; ytr are integer class indices of the model head."""
    set_seed(seed)
    model.to(device)
    params = param_groups if param_groups is not None else [p for p in model.parameters() if p.requires_grad]
    n = len(Xtr)
    if n == 0 or not params:
        return {"epochs_run": 0, "best_val_f1": float("nan"), "history": []}
    bn_active = not (backbone_eval or freeze_bn)
    if n < 2 and bn_active:
        return {"epochs_run": 0, "best_val_f1": float("nan"), "history": []}
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    Xt = torch.from_numpy(np.ascontiguousarray(Xtr)).float().to(device)
    yt = torch.from_numpy(np.asarray(ytr)).long().to(device)
    rng = np.random.default_rng(seed)
    use_val = Xva is not None and yva is not None and len(Xva) > 0
    best_f1, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(epochs):
        model.train()
        if backbone_eval:
            model.eval()
        elif freeze_bn:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()
        perm = torch.from_numpy(rng.permutation(n))
        tot = 0.0
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size].to(device)
            if len(b) < 2 and bn_active:
                continue
            loss = F.cross_entropy(model(Xt[b]), yt[b])
            if l2sp:
                loss = loss + sum(c * ((p - p0) ** 2).sum() for p, p0, c in l2sp)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item()) * len(b)
        rec = {"epoch": ep + 1, "train_loss": tot / n}
        if use_val:
            pv = forward_numpy(model, Xva, device).argmax(1)
            f1 = float(f1_score(yva, pv, average="macro", labels=np.unique(yva), zero_division=0))
            rec["val_macro_f1"] = f1
            if f1 > best_f1 + 1e-6:
                best_f1, best_state, bad = f1, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    hist.append(rec)
                    break
        hist.append(rec)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"epochs_run": len(hist), "best_val_f1": best_f1 if use_val else float("nan"), "history": hist}
