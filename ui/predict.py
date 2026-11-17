"""Page 3: predictions for one file."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config.settings import LOW_CONFIDENCE
from evaluation.report import evaluation_dict, prediction_table
from evaluation.summary import activity_summary, smooth_probs, transition_timeline
from visualization import plots
from . import state


def render_evaluation(ev, key: str = "") -> None:
    if ev is None:
        st.info("No valid ground-truth labels for this file → accuracy / macro-F1 are **not** computed "
                "(nothing is fabricated). Only descriptive prediction statistics are shown.")
        return
    c = st.columns(4)
    c[0].metric("Accuracy", f"{ev['accuracy']:.3f}")
    c[1].metric("Macro-F1", f"{ev['macro_f1']:.3f}")
    c[2].metric("Windows evaluated", ev["n_evaluated"])
    c[3].metric("Excluded (no/unsupported label)", ev["n_excluded"])
    if ev["excluded_labels"]:
        st.caption(f"Excluded labels: {ev['excluded_labels']}")
    st.caption("Macro-F1 is averaged over the classes present in the ground truth. "
               f"Expected calibration error of the softmax confidence on these windows: {ev.get('ece', float('nan')):.3f} "
               "(confidence is shown as raw model output, not a calibrated probability).")
    st.dataframe(ev["per_class"], use_container_width=True, hide_index=True)
    norm = st.checkbox("Row-normalise confusion matrix", key=f"cmnorm{key}")
    st.plotly_chart(plots.fig_confusion(ev["confusion_matrix"], ev["cm_labels"], normalize=norm), use_container_width=True)


def render(model) -> None:
    st.header("3 · Activity prediction")
    model = state.need_model(model)
    files = state.need_files()
    name = st.selectbox("File", list(files))
    pf = files[name]
    try:
        pred = state.get_prediction(pf, model)
    except Exception as e:
        st.error(str(e))
        return
    c1, c2 = st.columns(2)
    thr = c1.slider("Low-confidence threshold (max softmax probability)", 0.0, 1.0, LOW_CONFIDENCE, 0.05)
    k = c2.slider("Display smoothing: moving average over N windows (timeline only)", 1, 15, 1, 2)
    smoothed = smooth_probs(pred.probs, pred.segment, k).argmax(1) if k > 1 else pred.pred_idx
    summ = activity_summary(pred.pred, pred.classes)
    low = pred.low_conf_mask(thr)

    m = st.columns(5)
    m[0].metric("Most frequent activity", summ.iloc[0]["activity"])
    m[1].metric("Windows", len(pred.pred_idx))
    m[2].metric("Mean confidence", f"{pred.conf.mean():.2f}")
    m[3].metric("Low-confidence windows", f"{int(low.sum())} ({100 * low.mean():.0f}%)")
    m[4].metric("Inference", f"{pred.inference_ms_per_window:.3f} ms/window")
    st.caption("Confidence = maximum softmax probability. It has not been calibrated and should not be read as the "
               "probability that the prediction is correct.")

    tabs = st.tabs(["Summary", "Timeline", "Window table", "Ground-truth evaluation"])
    with tabs[0]:
        st.plotly_chart(plots.fig_activity_share(summ), use_container_width=True)
        st.dataframe(summ, use_container_width=True, hide_index=True)
        st.markdown("**Activity transition timeline** (runs of identical predictions within a segment)")
        tl = transition_timeline(np.array(pred.classes, dtype=object)[smoothed], pred.conf, pred.start, pred.segment, pred.subject)
        st.dataframe(tl, use_container_width=True, hide_index=True)
    with tabs[1]:
        t_end = float(pred.start_s.max() + 2.56)
        lo, hi = st.slider("Interval (s)", 0.0, t_end, (0.0, min(t_end, 120.0)), step=0.5, key="pred_iv")
        f = plots.fig_predictions_timeline(pred, smoothed, thr)
        f.update_xaxes(range=[lo, hi])
        st.plotly_chart(f, use_container_width=True)
        f2 = plots.fig_prob_heatmap(pred)
        f2.update_xaxes(range=[lo, hi])
        st.plotly_chart(f2, use_container_width=True)
    with tabs[2]:
        tbl = prediction_table(pred)
        tbl["low_confidence"] = low
        only_low = st.checkbox("Show only low-confidence windows")
        st.dataframe(tbl[tbl["low_confidence"]] if only_low else tbl, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download window-level predictions (CSV)", tbl.to_csv(index=False).encode(),
                           f"{name}_predictions.csv", "text/csv")
    with tabs[3]:
        ev = pred.evaluate()
        render_evaluation(ev, key=name)
        if ev is not None:
            import json
            st.download_button("⬇️ Download evaluation (JSON)", json.dumps(evaluation_dict(ev), indent=2, default=str).encode(),
                               f"{name}_evaluation.json")
