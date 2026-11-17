"""Experiment runner: identical test windows for every strategy; leak-free splits; fractions x seeds x folds."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from adaptation.strategies import (STRATEGIES, STRATEGY_LABELS, AdaptConfig, fit_strategy, predict_common)
from evaluation.metrics import evaluate_predictions
from models.har_cnn import HARNet
from models.registry import LoadedModel
from models.trainer import train_model
from preprocessing.normalization import Normalizer
from .pool import WindowPool
from .splits import group_kfold, group_split, stratified_fraction, temporal_split, train_val_split
from .tracking import Tracker


@dataclass
class ExperimentConfig:
    strategies: List[str] = field(default_factory=lambda: list(STRATEGIES))
    fractions: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.2, 1.0])
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    n_folds: int = 1                      # 1 = single group split; >1 = group k-fold
    common: List[str] = field(default_factory=list)
    adapt: AdaptConfig = field(default_factory=AdaptConfig)


@dataclass
class ExperimentResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    split_mode: str
    notes: List[str]

    def summary(self) -> pd.DataFrame:
        g = self.metrics.groupby(["strategy", "fraction"])
        return g.agg(accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
                     macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
                     inference_ms=("inference_ms_per_window", "mean"), trainable_params=("trainable_params", "max"),
                     runs=("accuracy", "count")).reset_index()


def make_splits(pool: WindowPool, seed: int, n_folds: int):
    n_groups = len(set(pool.group))
    if n_groups >= max(3, n_folds):
        mode = "subject-wise"
        if n_folds > 1:
            return group_kfold(pool.group, n_folds, seed), mode
        return [group_split(pool.group, seed)], mode
    if n_folds > 1:
        raise ValueError(f"Only {n_groups} subject group(s): cross-validation needs at least {max(3, n_folds)}.")
    return [temporal_split(pool.seg_key, pool.pos)], "temporal (purged) — fewer than 3 subjects available"


def train_source_model(pool: WindowPool, channels: List[str], common: List[str], cfg: AdaptConfig, seed: int = 0) -> LoadedModel:
    """Train a source model on source-domain files: subject-wise train/val split, normaliser from train only."""
    if len(set(pool.group)) >= 2:
        tr, va = train_val_split(pool.group, seed, 0.2)
    else:
        tr, va, _ = temporal_split(pool.seg_key, pool.pos)
    norm = Normalizer.fit(pool.X[tr], channels)
    model = HARNet(in_channels=len(channels), n_classes=len(common))
    idx = {c: i for i, c in enumerate(common)}
    enc = lambda y: np.array([idx[v] for v in y])
    train_model(model, norm.transform(pool.X[tr]), enc(pool.y[tr]), norm.transform(pool.X[va]), enc(pool.y[va]),
                epochs=cfg.epochs, lr=cfg.lr_scratch, batch_size=64, seed=seed, device=cfg.device, patience=cfg.patience)
    return LoadedModel(model, list(common), list(channels), norm,
                       {"dataset": "uploaded source CSVs", "n_train_windows": int(len(tr))}, None, cfg.device)


def run_experiment(source: Optional[LoadedModel], target: WindowPool, cfg: ExperimentConfig,
                   source_ref_raw: Optional[np.ndarray] = None, tracker: Optional[Tracker] = None,
                   progress: Optional[Callable[[float, str], None]] = None, save_predictions: bool = True) -> ExperimentResult:
    if not cfg.common:
        raise ValueError("Select at least two common activities.")
    if source is None and any(n != "scratch" for n in cfg.strategies):
        raise ValueError("A source model is required for every strategy except 'scratch'.")
    notes: List[str] = []
    rows, pred_rows = [], []
    total = len(cfg.seeds) * max(cfg.n_folds, 1) * len(cfg.fractions) * len(cfg.strategies)
    done, split_mode = 0, ""
    if tracker:
        tracker.log_params({"strategies": cfg.strategies, "fractions": cfg.fractions, "seeds": cfg.seeds,
                            "n_folds": cfg.n_folds, "common": cfg.common, **{f"adapt_{k}": v for k, v in asdict(cfg.adapt).items()},
                            "n_target_windows": target.n, "dropped_labels": target.dropped})
    for seed in cfg.seeds:
        splits, split_mode = make_splits(target, seed, cfg.n_folds)
        for fold, (tr, va, te) in enumerate(splits):
            if len(te) == 0 or len(tr) == 0:
                notes.append(f"seed {seed} fold {fold}: empty train or test partition – skipped.")
                done += len(cfg.fractions) * len(cfg.strategies)
                continue
            for frac in cfg.fractions:
                rng = np.random.default_rng(seed * 1000 + fold)
                tr_s = stratified_fraction(tr, target.y, frac, rng)
                va_s = stratified_fraction(va, target.y, frac, np.random.default_rng(seed * 1000 + fold + 1)) if len(va) else va
                Xte, yte = target.X[te], target.y[te]
                for name in cfg.strategies:
                    done += 1
                    if progress:
                        progress(done / total, f"seed {seed} fold {fold} frac {frac:g} – {STRATEGY_LABELS[name]}")
                    try:
                        fm = fit_strategy(name, source=source, Xtr=target.X[tr_s], ytr=target.y[tr_s],
                                          Xva=target.X[va_s] if len(va_s) else None,
                                          yva=target.y[va_s] if len(va_s) else None, common=cfg.common,
                                          cfg=cfg.adapt, seed=seed, X_src_raw=source_ref_raw)
                    except Exception as e:      # keep going; report the failure
                        notes.append(f"{name} (seed {seed}, fold {fold}, frac {frac:g}) failed: {e}")
                        continue
                    probs, pred, ms = predict_common(fm, Xte, cfg.common, cfg.adapt.device)
                    ev = evaluate_predictions(yte, pred, cfg.common)
                    row = {"strategy": name, "strategy_label": STRATEGY_LABELS[name], "fraction": frac, "seed": seed,
                           "fold": fold, "accuracy": ev["accuracy"], "macro_f1": ev["macro_f1"],
                           "inference_ms_per_window": ms, "trainable_params": fm.trainable_params,
                           "train_seconds": fm.train_seconds, "n_train_windows": int(len(tr_s)),
                           "n_val_windows": int(len(va_s)), "n_test_windows": int(len(te)),
                           "epochs_run": fm.info.get("epochs_run", 0), "split_mode": split_mode}
                    rows.append(row)
                    if tracker:
                        tracker.log_metrics({f"{name}_f{frac:g}_acc": ev["accuracy"], f"{name}_f{frac:g}_f1": ev["macro_f1"]})
                    if save_predictions:
                        pred_rows.append(pd.DataFrame({
                            "strategy": name, "fraction": frac, "seed": seed, "fold": fold, "file": target.file[te],
                            "group": target.group[te], "y_true": yte, "y_pred": pred, "confidence": probs.max(1)}))
    metrics = pd.DataFrame(rows)
    preds = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    if tracker and len(metrics):
        tracker.log_table("metrics", metrics)
        if len(preds):
            tracker.log_table("predictions", preds)
        (tracker.dir / "notes.txt").write_text("\n".join(notes))
    return ExperimentResult(metrics, preds, split_mode, notes)
