"""Page 5: cross-dataset domain analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from evaluation.domain import (domain_shift_findings, embed_pca, embed_tsne, embed_umap, layerwise_mmd, mmd_test,
                               summarize_windows)
from evaluation.features import window_features
from evaluation.report import findings_markdown
from models.predict import extract_layer_features
from visualization import plots
from . import state

ICON = {"ok": "🟢", "warn": "🟡", "high": "🔴"}


def render(model) -> None:
    st.header("5 · Cross-dataset domain analysis")
    model = state.need_model(model)
    files = state.need_files()
    ref = model.reference()
    if ref is None:
        st.error("The checkpoint has no reference sample of training windows (`*.reference.npz`). "
                 "Re-train with `python -m models.train` to enable domain analysis against the training data.")
        return
    Xref, yref = ref
    names = st.multiselect("Uploaded files to analyse", [n for n in files if files[n].windows.n > 0],
                           default=[n for n in files if files[n].windows.n > 0][:4])
    c = st.columns(4)
    space = c[0].selectbox("Embedding feature space", ["Hand-crafted window features", "Model embedding (penultimate layer)"])
    method = c[1].selectbox("2-D projection", ["PCA", "t-SNE", "UMAP"])
    n_perm = c[2].number_input("MMD permutations", 50, 2000, 200, 50)
    cap = c[3].number_input("Max windows per domain", 100, 2000, 400, 100)
    if not names:
        return
    if not st.button("▶ Run domain analysis", type="primary"):
        st.info("Press the button to compute MMD tests, embeddings and the domain-shift report.")
        return
    rng = np.random.default_rng(0)
    ri = rng.choice(len(Xref), min(len(Xref), int(cap)), replace=False)
    Xr, yr = Xref[ri], yref[ri]
    Fr_hand = window_features(Xr)
    Fr_emb = extract_layer_features(model, Xr)
    ref_sum = summarize_windows(Xr, model.channels, y=None)
    ref_sum_y = summarize_windows(Xref, model.channels, y=yref)
    ref_sum["class_gravity"], ref_dist = ref_sum_y["class_gravity"], ref_sum_y["class_dist"]

    results, layer_rows, findings_md = {}, [], []
    feats_pool = [Fr_hand if space.startswith("Hand") else Fr_emb["embed"]]
    dom, act = ["training reference"] * len(Xr), list(yr)
    for n in names:
        pf = files[n]
        X = pf.windows.select(model.channels)
        idx = rng.choice(len(X), min(len(X), int(cap)), replace=False)
        Xs, ys = X[idx], pf.windows.y[idx]
        F_hand, F_emb = window_features(Xs), extract_layer_features(model, Xs)
        Fa = F_hand if space.startswith("Hand") else F_emb["embed"]
        mm_hand = mmd_test(Fr_hand, F_hand, n_perm=int(n_perm))
        mm_emb = mmd_test(Fr_emb["embed"], F_emb["embed"], n_perm=int(n_perm))
        lw = layerwise_mmd(Fr_emb, F_emb, n_perm=int(n_perm))
        for lname, r in lw.items():
            layer_rows.append({"file": n, "layer": lname, "mmd2": max(r["mmd2"], 0.0), "p_value": r["p_value"]})
        if pf.has_labels:
            gt = pf.windows.y
            tsum = summarize_windows(pf.windows.select(model.channels), model.channels, y=gt)
            tdist = tsum["class_dist"]
            basis = "ground-truth labels"
        else:
            p = state.get_prediction(pf, model)
            tsum = summarize_windows(pf.windows.select(model.channels), model.channels)
            tdist = {c_: float((p.pred == c_).mean()) for c_ in model.classes}
            basis = "PREDICTED labels (no ground truth)"
        f = domain_shift_findings(tsum, ref_sum, orig_fs=pf.stats["original_fs_hz"], acc_unit=pf.stats["acc_unit"],
                                  gyro_unit=pf.stats["gyro_unit"], mmd_res=mm_hand, tgt_dist=tdist, ref_dist=ref_dist)
        results[n] = {"mm_hand": mm_hand, "mm_emb": mm_emb, "findings": f, "basis": basis, "sum": tsum}
        feats_pool.append(Fa)
        dom += [n] * len(Xs)
        act += [v if v is not None else "" for v in ys]
        findings_md.append(findings_markdown(n, f))

    st.subheader("Does the uploaded data differ from the training data?")
    tab = pd.DataFrame([{"file": n, "MMD² (hand-crafted)": r["mm_hand"]["mmd2"], "p (hand-crafted)": r["mm_hand"]["p_value"],
                         "MMD² (embedding)": r["mm_emb"]["mmd2"], "p (embedding)": r["mm_emb"]["p_value"],
                         "significant (p<0.05, either space)": bool(min(r["mm_hand"]["p_value"], r["mm_emb"]["p_value"]) < 0.05)}
                        for n, r in results.items()])
    st.dataframe(tab, use_container_width=True, hide_index=True)
    st.caption("Unbiased RBF-MMD² with median-heuristic bandwidth; permutation p-value. Features standardised with the "
               "training-reference statistics. MMD detects *any* distribution difference — it cannot say which factor caused it.")

    t1, t2, t3, t4 = st.tabs(["Embedding", "Distributions", "Layer-wise MMD", "Domain-shift report"])
    with t1:
        Z = np.vstack(feats_pool)
        mu, sd = Z[: len(Xr)].mean(0), Z[: len(Xr)].std(0)
        sd[sd < 1e-8] = 1.0
        Z = (Z - mu) / sd
        try:
            if method == "PCA":
                emb, ev = embed_pca(Z)
                title = f"PCA ({100 * ev[0]:.0f}% + {100 * ev[1]:.0f}% variance)"
            elif method == "t-SNE":
                emb, title = embed_tsne(Z), "t-SNE"
            else:
                emb, title = embed_umap(Z), "UMAP"
            st.plotly_chart(plots.fig_embedding(emb, dom, f"{title} — {space}", act), use_container_width=True)
        except Exception as e:
            st.error(str(e))
    with t2:
        mags = {"training reference": np.linalg.norm(Xref[:, :, :3], axis=2).ravel()}
        wm = {"training reference": np.linalg.norm(Xref[:, :, :3], axis=2)}
        for n in names:
            X = files[n].windows.select(model.channels)
            mags[n] = np.linalg.norm(X[:, :, :3], axis=2).ravel()
            wm[n] = np.linalg.norm(X[:, :, :3], axis=2)
        st.plotly_chart(plots.fig_magnitude_hist(mags), use_container_width=True)
        st.plotly_chart(plots.fig_psd(wm), use_container_width=True)
        rows = [{"domain": "training reference", **{f"{k}": v for k, v in ref_sum["channel_mean"].items()}}]
        rows += [{"domain": n, **results[n]["sum"]["channel_mean"]} for n in names]
        st.markdown("**Per-channel mean (standard units)**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        rows = [{"domain": "training reference", **ref_sum["channel_std"]}] + [{"domain": n, **results[n]["sum"]["channel_std"]} for n in names]
        st.markdown("**Per-channel standard deviation**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with t3:
        lt = pd.DataFrame(layer_rows)
        st.plotly_chart(plots.fig_layer_mmd(lt), use_container_width=True)
        st.dataframe(lt, use_container_width=True, hide_index=True)
        st.caption("Pooled global-average features of each layer, extracted from the loaded model.")
    with t4:
        for n in names:
            st.markdown(f"#### {n}")
            st.caption(f"Class-distribution comparison uses {results[n]['basis']}.")
            st.dataframe(pd.DataFrame([{"": ICON[f["severity"]], "aspect": f["aspect"], "evidence": f["evidence"],
                                        "possible cause (hypothesis)": f["possible_cause"]} for f in results[n]["findings"]]),
                         use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download domain-shift report (Markdown)", "\n".join(findings_md).encode(), "domain_shift_report.md")
        st.session_state["domain_report_md"] = "\n".join(findings_md)
