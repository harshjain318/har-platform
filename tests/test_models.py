"""Torch-dependent tests (skipped automatically when torch is missing). Uses SYNTHETIC data purely to test the
training / checkpoint / strategy machinery — the application itself never uses synthetic data."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from adaptation.strategies import STRATEGIES, AdaptConfig, fit_strategy, predict_common   # noqa: E402
from experiments.pool import build_pool                                                       # noqa: E402
from experiments.runner import ExperimentConfig, run_experiment                               # noqa: E402
from models.har_cnn import HARNet                                                             # noqa: E402
from models.registry import load_checkpoint, save_checkpoint                                  # noqa: E402
from models.trainer import forward_numpy, train_model                                         # noqa: E402
from preprocessing.normalization import Normalizer                                            # noqa: E402

CLASSES = ["walking", "sitting", "standing"]


def _data(n_per=60, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(128) / 50
    X, y = [], []
    for k, c in enumerate(CLASSES):
        for _ in range(n_per):
            g = np.array([0, 1, 0]) if c != "sitting" else np.array([0, .6, .8])
            osc = 0.5 * np.sin(2 * np.pi * 2 * t)[:, None] * (c == "walking")
            X.append(g + shift + osc + rng.normal(0, .05, (128, 3)))
            y.append(c)
    return np.array(X, dtype=np.float32), np.array(y, dtype=object)


def test_forward_shapes_and_layers():
    m = HARNet(3, 3)
    lg, ly = m(torch.randn(4, 128, 3), return_layers=True)
    assert lg.shape == (4, 3) and set(ly) == {"block1", "block2", "block3", "embed"}


def test_train_and_checkpoint_roundtrip(tmp_path):
    X, y = _data()
    norm = Normalizer.fit(X, ["acc_x", "acc_y", "acc_z"])
    idx = np.array([CLASSES.index(v) for v in y])
    m = HARNet(3, 3)
    train_model(m, norm.transform(X), idx, epochs=8, batch_size=32)
    acc = (forward_numpy(m, norm.transform(X)).argmax(1) == idx).mean()
    assert acc > 0.9
    p = tmp_path / "m.pt"
    save_checkpoint(p, m, CLASSES, ["acc_x", "acc_y", "acc_z"], norm, {"dataset": "synthetic"}, X[:50], y[:50])
    lm = load_checkpoint(p)
    assert lm.classes == CLASSES and lm.reference()[0].shape[0] == 50
    assert np.allclose(forward_numpy(lm.model, norm.transform(X[:5])), forward_numpy(m, norm.transform(X[:5])), atol=1e-5)


def test_all_strategies_run(tmp_path):
    X, y = _data()
    norm = Normalizer.fit(X, ["acc_x", "acc_y", "acc_z"])
    m = HARNet(3, 3)
    train_model(m, norm.transform(X), np.array([CLASSES.index(v) for v in y]), epochs=5, batch_size=32)
    p = tmp_path / "m.pt"
    save_checkpoint(p, m, CLASSES, ["acc_x", "acc_y", "acc_z"], norm, {}, X, y)
    src = load_checkpoint(p)
    Xt, yt = _data(20, shift=0.1, seed=5)
    cfg = AdaptConfig(epochs=3, batch_size=16)
    for s in STRATEGIES:
        fm = fit_strategy(s, source=src, Xtr=Xt, ytr=yt, Xva=Xt[:10], yva=yt[:10], common=CLASSES, cfg=cfg, seed=0, X_src_raw=X)
        probs, pred, ms = predict_common(fm, Xt, CLASSES)
        assert probs.shape == (len(Xt), 3) and np.allclose(probs.sum(1), 1, atol=1e-5)
        if s == "source_only":
            assert fm.trainable_params == 0
        if s == "frozen_extractor":
            assert fm.trainable_params == sum(q.numel() for q in src.model.head.parameters())
    assert fm.info["weight_basis"] == "mmd"
