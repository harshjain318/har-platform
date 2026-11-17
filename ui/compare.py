"""Page 4: multi-CSV / cross-dataset comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from evaluation.domain import js_distance
from evaluation.summary import activity_summary
from visualization import plots
from . import state
from .predict import render_evaluation


def render(model) -> None:
    st.header("4 · Multi-CSV comparison")
    model = state.need_model(model)
    files = state.need_files()
    names = st.multiselect("Files to compare", list(files), default=list(files))
    if not names:
        return
    preds, evs, rows = {}, {}, []
    for n in names:
        pf = files[n]
        try:
            p = state.get_prediction(pf, model)
        except Exception as e:
            st.error(f"{n}: {e}")
            continue
        preds[n], evs[n] = p, p.evaluate()
        s = activity_summary(p.pred, p.classes)
        ev = evs[n]
        rows.append({"Dataset": pf.source, "File": n, "Predicted activity (most frequent)": s.iloc[0]["activity"],
                     "% windows": round(float(s.iloc[0]["percent_of_windows"]), 1), "Windows": len(p.pred_idx),
                     "Mean confidence": round(float(p.conf.mean()), 3),
                     "Accuracy": round(ev["accuracy"], 4) if ev else "n/a (no ground truth)",
                     "Macro-F1": round(ev["macro_f1"], 4) if ev else "n/a (no ground truth)",
                     "Evaluated windows": ev["n_evaluated"] if ev else 0})
    if not rows:
        return
    table = pd.DataFrame(rows)
    st.subheader("Dataset-wise comparison")
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption("Accuracy and macro-F1 are computed only where valid ground-truth labels exist; otherwise 'n/a'.")
    st.download_button("⬇️ Download comparison table (CSV)", table.to_csv(index=False).encode(), "comparison.csv")
    st.session_state["comparison_table"] = table

    tabs = st.tabs(["Performance", "Confusion matrices", "Signal distributions", "Cross-dataset prediction differences"])
    labeled = [n for n in preds if evs[n] is not None]
    with tabs[0]:
        if not labeled:
            st.info("None of the selected files has usable ground truth.")
        else:
            pc = pd.concat([evs[n]["per_class"].assign(dataset=n) for n in labeled])
            pc = pc.rename(columns={"recall (per-class accuracy)": "per_class_accuracy"})
            st.plotly_chart(plots.fig_per_class(pc, "per_class_accuracy", "Per-class accuracy (recall)"), use_container_width=True)
            st.plotly_chart(plots.fig_per_class(pc, "f1", "Per-class F1"), use_container_width=True)
            st.dataframe(pc[["dataset", "class", "precision", "per_class_accuracy", "f1", "support"]], use_container_width=True, hide_index=True)
    with tabs[1]:
        if not labeled:
            st.info("Confusion matrices need ground-truth labels.")
        else:
            sel = st.selectbox("File", labeled)
            render_evaluation(evs[sel], key=f"cmp{sel}")
    with tabs[2]:
        mags = {n: np.linalg.norm(files[n].signal[:, :3], axis=1) for n in preds}
        st.plotly_chart(plots.fig_magnitude_hist(mags), use_container_width=True)
        wm = {n: np.linalg.norm(files[n].windows.X[:, :, :3], axis=2) for n in preds}
        st.plotly_chart(plots.fig_psd(wm), use_container_width=True)
        stats = []
        for n in preds:
            s = files[n].signal
            for j, c in enumerate(files[n].channels):
                stats.append({"file": n, "channel": c, "mean": s[:, j].mean(), "std": s[:, j].std()})
        st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True)
    with tabs[3]:
        dist = pd.DataFrame({n: activity_summary(preds[n].pred, model.classes).set_index("activity")["percent_of_windows"]
                             for n in preds}).T[model.classes]
        st.plotly_chart(plots.fig_stacked_pred_dist(dist), use_container_width=True)
        if len(preds) > 1:
            js = pd.DataFrame([[js_distance(dist.loc[a].values, dist.loc[b].values) for b in dist.index] for a in dist.index],
                              index=dist.index, columns=dist.index)
            st.plotly_chart(plots.fig_heatmap(js, "Jensen-Shannon distance between predicted-activity distributions (0 = identical)"),
                            use_container_width=True)
            st.caption("Differences can reflect real activity differences between files, not only domain shift.")
