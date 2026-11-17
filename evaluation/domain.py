"""Cross-dataset domain analysis: summaries, MMD (+permutation test), PSD, embeddings, shift report."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy import stats as sps
from scipy.signal import welch
from scipy.spatial.distance import cdist, jensenshannon


# ----------------------------------------------------------------------------- MMD
def _rbf_kernel(Z: np.ndarray, gamma: float) -> np.ndarray:
    return np.exp(-gamma * cdist(Z, Z, "sqeuclidean"))


def _mmd2_from_kernel(K: np.ndarray, n: int) -> float:
    m = K.shape[0] - n
    Kxx, Kyy, Kxy = K[:n, :n], K[n:, n:], K[:n, n:]
    return float((Kxx.sum() - np.trace(Kxx)) / (n * (n - 1)) + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
                 - 2.0 * Kxy.mean())


def mmd_test(X: np.ndarray, Y: np.ndarray, n_perm: int = 200, max_n: int = 400, seed: int = 0,
             standardize: bool = True) -> Dict:
    """Unbiased RBF-MMD^2 (median heuristic) with a permutation p-value.
    X, Y: (n, d) feature matrices. Larger MMD^2 / small p => distributions differ."""
    rng = np.random.default_rng(seed)
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    if len(X) < 3 or len(Y) < 3:
        return {"mmd2": float("nan"), "p_value": float("nan"), "n_x": len(X), "n_y": len(Y), "note": "too few samples"}
    if len(X) > max_n:
        X = X[rng.choice(len(X), max_n, replace=False)]
    if len(Y) > max_n:
        Y = Y[rng.choice(len(Y), max_n, replace=False)]
    Z = np.vstack([X, Y])
    if standardize:
        mu, sd = X.mean(0), X.std(0)
        sd[sd < 1e-8] = 1.0
        Z = (Z - mu) / sd          # reference (X) statistics only
    d2 = cdist(Z, Z, "sqeuclidean")
    med = np.median(d2[np.triu_indices_from(d2, 1)])
    gamma = 1.0 / (med + 1e-12)
    K = np.exp(-gamma * d2)
    n = len(X)
    obs = _mmd2_from_kernel(K, n)
    cnt = 0
    for _ in range(n_perm):
        p = rng.permutation(len(Z))
        if _mmd2_from_kernel(K[np.ix_(p, p)], n) >= obs:
            cnt += 1
    return {"mmd2": obs, "p_value": (cnt + 1) / (n_perm + 1), "n_x": len(X), "n_y": len(Y), "note": ""}


def layerwise_mmd(feats_a: Dict[str, np.ndarray], feats_b: Dict[str, np.ndarray], **kw) -> Dict[str, Dict]:
    return {k: mmd_test(feats_a[k], feats_b[k], **kw) for k in feats_a if k in feats_b}


# ----------------------------------------------------------------------------- spectra / distributions
def mean_psd(x: np.ndarray, fs: float = 50.0):
    """Welch PSD. x may be 1-D (one signal) or 2-D (n_windows, T): PSDs are averaged over windows,
    so no spectral artefacts are introduced at segment boundaries."""
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[None, :]
    x = x - x.mean(axis=1, keepdims=True)
    f, p = welch(x, fs=fs, nperseg=min(128, x.shape[1]), axis=1)
    return f, p.mean(axis=0)


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    p, q = np.asarray(p, float), np.asarray(q, float)
    if p.sum() == 0 or q.sum() == 0:
        return float("nan")
    return float(jensenshannon(p / p.sum(), q / q.sum(), base=2))


def ks_test(a: np.ndarray, b: np.ndarray) -> Dict:
    r = sps.ks_2samp(a, b)
    return {"statistic": float(r.statistic), "p_value": float(r.pvalue)}


def summarize_windows(X: np.ndarray, channels: List[str], fs: float = 50.0, y: Optional[np.ndarray] = None) -> Dict:
    """Descriptive statistics of (N, T, C) windows in standard units."""
    X = np.asarray(X, float)
    acc = X[:, :, :3]
    mag = np.linalg.norm(acc, axis=2)
    centred = mag - mag.mean(1, keepdims=True)
    spec = np.abs(np.fft.rfft(centred, axis=1)) ** 2
    f = np.fft.rfftfreq(X.shape[1], 1 / fs)
    tot = spec.sum() + 1e-12
    out = {
        "n_windows": int(len(X)),
        "channel_mean": dict(zip(channels, X.reshape(-1, X.shape[-1]).mean(0).round(4))),
        "channel_std": dict(zip(channels, X.reshape(-1, X.shape[-1]).std(0).round(4))),
        "acc_mean_vector": acc.reshape(-1, 3).mean(0),
        "mag_mean": float(mag.mean()), "mag_std": float(mag.std()), "mag_p99": float(np.percentile(mag, 99)),
        "hf_fraction": float(spec[:, f > 10].sum() / tot),
        "acc_std_mean": float(acc.std(1).mean()),
    }
    if y is not None:
        yy = np.array([str(v) if v is not None else "" for v in y])
        out["class_gravity"] = {c: acc[yy == c].reshape(-1, 3).mean(0) for c in np.unique(yy) if c}
        out["class_dist"] = {c: float((yy == c).mean()) for c in np.unique(yy) if c}
    return out


def _angle(u: np.ndarray, v: np.ndarray) -> float:
    c = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def domain_shift_findings(tgt: Dict, ref: Dict, *, orig_fs: float, acc_unit: str, gyro_unit: Optional[str],
                          mmd_res: Optional[Dict] = None, tgt_dist: Optional[Dict[str, float]] = None,
                          ref_dist: Optional[Dict[str, float]] = None, target_fs: float = 50.0) -> List[Dict]:
    """Rule-based, evidence-backed domain-shift report. Causes are HYPOTHESES, not conclusions."""
    out: List[Dict] = []

    def add(aspect, sev, evidence, cause):
        out.append({"aspect": aspect, "severity": sev, "evidence": evidence, "possible_cause": cause})

    r = orig_fs / target_fs
    sev = "ok" if abs(r - 1) <= 0.1 else ("high" if (r >= 2 or r <= 0.5) else "warn")
    add("Sampling frequency", sev, f"Source ≈ {orig_fs:.1f} Hz vs {target_fs:.0f} Hz model rate (ratio {r:.2f}).",
        "Different device hardware/OS sampling policy; resampling changes temporal dynamics and bandwidth." if sev != "ok" else "—")

    mr = tgt["mag_mean"] / max(ref["mag_mean"], 1e-9)
    sev = "ok" if 0.8 <= mr <= 1.25 else "high"
    add("Units / gravity", sev,
        f"Declared acc unit '{acc_unit}'" + (f", gyro unit '{gyro_unit}'" if gyro_unit else "") +
        f"; mean |acc| = {tgt['mag_mean']:.2f} g vs {ref['mag_mean']:.2f} g in training data.",
        "Wrong unit selection, or gravity removed/added (body vs total acceleration)." if sev != "ok" else "—")

    ang, basis = None, ""
    if "class_gravity" in tgt and "class_gravity" in ref:
        common = [c for c in tgt["class_gravity"] if c in ref["class_gravity"] and c in ("sitting", "standing", "walking", "lying")]
        if common:
            ang = float(np.mean([_angle(tgt["class_gravity"][c], ref["class_gravity"][c]) for c in common]))
            basis = f"mean per-class gravity-direction angle over {common}"
    if ang is None:
        ang = _angle(tgt["acc_mean_vector"], ref["acc_mean_vector"])
        basis = "angle between overall mean acceleration vectors (confounded by activity mix when labels are absent)"
    sev = "ok" if ang < 25 else ("warn" if ang < 60 else "high")
    add("Sensor orientation / placement", sev, f"{ang:.0f}° — {basis}.",
        "Different body placement (wrist/pocket/waist) or device axis convention." if sev != "ok" else "—")

    hr = (tgt["hf_fraction"] + 1e-9) / (ref["hf_fraction"] + 1e-9)
    sev = "ok" if 0.5 <= hr <= 2.0 else "warn"
    add("High-frequency content (>10 Hz of |acc|)", sev,
        f"{tgt['hf_fraction']:.3f} vs {ref['hf_fraction']:.3f} (ratio {hr:.2f}).",
        "Different sensor noise floor / internal filtering of the device." if sev != "ok" else "—")

    sr = tgt["acc_std_mean"] / max(ref["acc_std_mean"], 1e-9)
    sev = "ok" if 0.5 <= sr <= 2.0 else "warn"
    add("Signal variability", sev, f"Mean within-window acc std {tgt['acc_std_mean']:.3f} g vs {ref['acc_std_mean']:.3f} g.",
        "Different subject population, activity intensity or sensor placement." if sev != "ok" else "—")

    if tgt_dist and ref_dist:
        cl = sorted(set(tgt_dist) | set(ref_dist))
        js = js_distance(np.array([tgt_dist.get(c, 0) for c in cl]), np.array([ref_dist.get(c, 0) for c in cl]))
        sev = "ok" if js < 0.2 else ("warn" if js < 0.4 else "high")
        add("Activity distribution", sev, f"Jensen-Shannon distance {js:.2f} between class proportions.",
            "Different protocol / activity mix (label shift)." if sev != "ok" else "—")

    if mmd_res and np.isfinite(mmd_res.get("p_value", np.nan)):
        sig = mmd_res["p_value"] < 0.05
        add("Feature-space distribution (MMD)", "high" if sig else "ok",
            f"MMD² = {mmd_res['mmd2']:.4f}, permutation p = {mmd_res['p_value']:.3f} "
            f"(n={mmd_res['n_x']} vs {mmd_res['n_y']}).",
            "Distributions differ significantly (note: with many windows even small shifts are significant)." if sig
            else "No significant difference detected at α=0.05.")
    return out


# ----------------------------------------------------------------------------- embeddings
def embed_pca(F: np.ndarray, n: int = 2):
    from sklearn.decomposition import PCA
    p = PCA(n_components=n, random_state=0)
    return p.fit_transform(F), p.explained_variance_ratio_


def embed_tsne(F: np.ndarray, seed: int = 0):
    from sklearn.manifold import TSNE
    perp = float(max(2, min(30, (len(F) - 1) / 3)))
    return TSNE(n_components=2, perplexity=perp, init="pca", random_state=seed).fit_transform(F)


def embed_umap(F: np.ndarray, seed: int = 0):
    try:
        import umap
    except ImportError as e:
        raise ImportError("umap-learn is not installed (pip install umap-learn).") from e
    return umap.UMAP(n_components=2, random_state=seed, n_neighbors=min(15, len(F) - 1)).fit_transform(F)
