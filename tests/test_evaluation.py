import numpy as np

from evaluation.domain import domain_shift_findings, js_distance, mean_psd, mmd_test, summarize_windows
from evaluation.features import window_features
from evaluation.metrics import evaluate_predictions, expected_calibration_error
from evaluation.summary import activity_summary, smooth_probs, transition_timeline
from experiments.splits import group_kfold, group_split, stratified_fraction, temporal_split
from preprocessing.normalization import Normalizer

CL = ["walking", "sitting", "standing"]


def test_metrics_exact_values():
    yt = ["walking"] * 4 + ["sitting"] * 4
    yp = ["walking", "walking", "walking", "sitting", "sitting", "sitting", "sitting", "sitting"]
    r = evaluate_predictions(yt, yp, CL)
    assert abs(r["accuracy"] - 7 / 8) < 1e-9
    # walking: P=1, R=.75, F1=.857 ; sitting: P=.8, R=1, F1=.889
    assert abs(r["macro_f1"] - (0.857142857 + 0.888888889) / 2) < 1e-6
    assert r["true_labels_present"] == ["walking", "sitting"]
    assert r["confusion_matrix"].sum() == 8


def test_metrics_no_ground_truth_returns_none():
    assert evaluate_predictions([None, None], ["walking", "sitting"], CL) is None
    assert evaluate_predictions(["running", "running"], ["walking", "sitting"], CL) is None   # unsupported by model


def test_metrics_excludes_unsupported_labels():
    r = evaluate_predictions(["walking", "running", None], ["walking", "walking", "sitting"], CL)
    assert r["n_evaluated"] == 1 and r["n_excluded"] == 2 and r["accuracy"] == 1.0


def test_ece():
    assert expected_calibration_error(np.ones(10), np.ones(10)) < 1e-9
    assert abs(expected_calibration_error(np.full(10, 0.9), np.r_[np.ones(5), np.zeros(5)]) - 0.4) < 1e-9


def test_mmd_detects_shift_and_accepts_same():
    rng = np.random.default_rng(0)
    A = rng.normal(0, 1, (200, 5)); B = rng.normal(0, 1, (200, 5)); C = rng.normal(1.0, 1, (200, 5))
    same = mmd_test(A, B, n_perm=100)
    diff = mmd_test(A, C, n_perm=100)
    assert same["p_value"] > 0.05 and diff["p_value"] < 0.05 and diff["mmd2"] > same["mmd2"]


def test_domain_findings_flag_obvious_shift():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 0.1, (50, 128, 3)) + np.array([0, 1, 0])
    Xs = rng.normal(0, 0.1, (50, 128, 3)) + np.array([1, 0, 0])          # rotated gravity
    ref, tgt = summarize_windows(X, ["a", "b", "c"]), summarize_windows(Xs, ["a", "b", "c"])
    f = {x["aspect"]: x for x in domain_shift_findings(tgt, ref, orig_fs=25.0, acc_unit="m/s2", gyro_unit=None)}
    assert f["Sensor orientation / placement"]["severity"] == "high"
    assert f["Sampling frequency"]["severity"] == "high"
    ok = {x["aspect"]: x for x in domain_shift_findings(ref, ref, orig_fs=50.0, acc_unit="g", gyro_unit=None)}
    assert all(v["severity"] == "ok" for v in ok.values())


def test_psd_peak_location():
    t = np.arange(0, 20, 1 / 50)
    f, p = mean_psd(np.sin(2 * np.pi * 5 * t)[None, :].repeat(3, 0)[:, :512])
    assert abs(f[np.argmax(p)] - 5) < 0.6
    assert js_distance(np.array([1, 0]), np.array([0, 1])) > 0.99


def test_features_shape():
    assert window_features(np.random.randn(7, 128, 6)).shape == (7, 6 * 6 + 2)


def test_normalizer_roundtrip():
    X = np.random.default_rng(0).normal(5, 3, (40, 128, 3))
    n = Normalizer.fit(X, ["a", "b", "c"])
    Z = n.transform(X)
    assert abs(Z.mean()) < 1e-3 and abs(Z.std() - 1) < 1e-2
    n2 = Normalizer.from_dict(n.to_dict())
    assert np.allclose(n2.transform(X), Z)
    assert np.allclose(n.inverse(Z), X, atol=1e-3)


def test_group_split_no_subject_leak():
    groups = np.repeat([f"s{i}" for i in range(10)], 20)
    tr, va, te = group_split(groups, 0)
    g = lambda i: set(groups[i])
    assert not (g(tr) & g(va)) and not (g(tr) & g(te)) and not (g(va) & g(te))
    assert len(tr) + len(va) + len(te) == len(groups)
    for a, b, c in group_kfold(groups, 5, 0):
        assert not (g(a) & g(c)) and not (g(b) & g(c)) and not (g(a) & g(b))


def test_temporal_split_purged():
    key = np.repeat(["seg0", "seg1"], 20)
    pos = np.tile(np.arange(20) * 64, 2)
    tr, va, te = temporal_split(key, pos)
    for seg in ("seg0", "seg1"):
        m = lambda idx: pos[idx][key[idx] == seg]
        a, b, c = m(tr), m(va), m(te)
        assert a.max() + 128 <= b.min() and b.max() + 128 <= c.min()      # no shared samples


def test_stratified_fraction_keeps_all_classes():
    y = np.array(["a"] * 100 + ["b"] * 100 + ["c"] * 5)
    idx = stratified_fraction(np.arange(len(y)), y, 0.01, np.random.default_rng(0))
    assert set(y[idx]) == {"a", "b", "c"}


def test_summary_helpers():
    pred = np.array(["walking"] * 6 + ["sitting"] * 4, dtype=object)
    s = activity_summary(pred, CL)
    assert s.iloc[0]["activity"] == "walking" and abs(s.iloc[0]["percent_of_windows"] - 60) < 1e-9
    seg = np.zeros(10, int)
    tl = transition_timeline(pred, np.full(10, .9), np.arange(10) * 64, seg, np.array(["u"] * 10, dtype=object))
    assert list(tl["activity"]) == ["walking", "sitting"] and tl.iloc[1]["start_s"] == 6 * 64 / 50
    P = np.random.default_rng(0).dirichlet(np.ones(3), 10)
    assert smooth_probs(P, seg, 3).shape == P.shape
