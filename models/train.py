"""Train a source HAR model and write a checkpoint (weights + class names + normaliser + reference sample).

    python -m models.train --uci-har-dir "datasets/UCI HAR Dataset" --channels acc
    python -m models.train --csv-spec datasets/my_spec.json --channels acc
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import torch

from config.settings import CHANNELS_ACC, CHANNELS_GYRO, CHECKPOINT_DIR
from datasets.loaders import load_csv_spec, load_uci_har
from evaluation.metrics import evaluate_predictions
from experiments.splits import group_split, train_val_split
from preprocessing.normalization import Normalizer
from .har_cnn import HARNet
from .registry import save_checkpoint
from .trainer import forward_numpy, set_seed, train_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uci-har-dir")
    ap.add_argument("--csv-spec")
    ap.add_argument("--channels", choices=["acc", "acc_gyro"], default="acc",
                    help="acc (3 ch) is recommended: it works for datasets without a gyroscope")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(CHECKPOINT_DIR / "har_source.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if bool(a.uci_har_dir) == bool(a.csv_spec):
        ap.error("Provide exactly one of --uci-har-dir or --csv-spec")

    set_seed(a.seed)
    channels = CHANNELS_ACC + (CHANNELS_GYRO if a.channels == "acc_gyro" else [])
    if a.uci_har_dir:
        d = load_uci_har(a.uci_har_dir, use_gyro=a.channels == "acc_gyro")
        Xp, yp, sp = d["train"]
        Xte, yte, ste = d["test"]            # official held-out subjects
        tr, va = train_val_split(sp, a.seed, 0.2)   # subject-wise split of the official training subjects
        dataset_name = "UCI HAR"
        Xtr, ytr, Xva, yva = Xp[tr], yp[tr], Xp[va], yp[va]
        train_subj = len(set(sp[tr]))
    else:
        Xa, ya, sa = load_csv_spec(a.csv_spec, channels)
        tr, va, te = group_split(sa, a.seed)
        Xtr, ytr, Xva, yva, Xte, yte = Xa[tr], ya[tr], Xa[va], ya[va], Xa[te], ya[te]
        dataset_name, train_subj = Path(a.csv_spec).stem, len(set(sa[tr]))

    classes = [c for c in ["walking", "upstairs", "downstairs", "sitting", "standing", "running", "lying"]
               if c in set(ytr)]
    cidx = {c: i for i, c in enumerate(classes)}
    keep = lambda y: np.array([v in cidx for v in y])
    ktr, kva, kte = keep(ytr), keep(yva), keep(yte)
    Xtr, ytr, Xva, yva, Xte, yte = Xtr[ktr], ytr[ktr], Xva[kva], yva[kva], Xte[kte], yte[kte]

    norm = Normalizer.fit(Xtr, channels)          # statistics from TRAINING subjects only
    f = lambda X: norm.transform(X)
    model = HARNet(in_channels=len(channels), n_classes=len(classes))
    enc = lambda y: np.array([cidx[v] for v in y])
    log = train_model(model, f(Xtr), enc(ytr), f(Xva), enc(yva), epochs=a.epochs, lr=a.lr,
                      batch_size=a.batch_size, seed=a.seed, device=a.device)
    pred = np.array(classes, dtype=object)[forward_numpy(model, f(Xte), a.device).argmax(1)]
    test = evaluate_predictions(yte, pred, classes)
    meta = {"dataset": dataset_name, "classes": classes, "channels": channels, "seed": a.seed,
            "epochs_run": log["epochs_run"], "best_val_macro_f1": log["best_val_f1"],
            "n_train_windows": int(len(Xtr)), "n_val_windows": int(len(Xva)), "n_test_windows": int(len(Xte)),
            "n_train_subjects": train_subj, "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
            "torch": torch.__version__,
            "test_accuracy": test["accuracy"] if test else None, "test_macro_f1": test["macro_f1"] if test else None}
    rng = np.random.default_rng(a.seed)
    ref = rng.choice(len(Xtr), size=min(2000, len(Xtr)), replace=False)
    save_checkpoint(Path(a.out), model, classes, channels, norm, meta, Xtr[ref], ytr[ref])
    print(f"Saved {a.out}")
    print(f"  classes: {classes}")
    print(f"  held-out test: accuracy={meta['test_accuracy']}  macro-F1={meta['test_macro_f1']}")


if __name__ == "__main__":
    main()
