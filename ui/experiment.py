"""Page 6 (Model comparison) and Page 7 (Research experiment mode). Both use experiments.runner."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from adaptation.strategies import STRATEGIES, STRATEGY_LABELS, AdaptConfig
from config.settings import CANONICAL_ACTIVITIES
from experiments.pool import build_pool
from experiments.runner import ExperimentConfig, run_experiment, train_source_model
from experiments.tracking import HAS_MLFLOW, Tracker
from visualization import plots
from . import state

FRACTIONS = [0.01, 0.05, 0.1, 0.2, 1.0]


def _labels_in(files, names):
    out = set()
    for n in names:
        out |= {v for v in files[n].windows.y if v is not None}
    return sorted(out)


def _show(res, key):
    if res.notes:
        with st.expander(f"{len(res.notes)} note(s)/warning(s)"):
            for n in res.notes:
                st.write("• " + n)
    if res.metrics.empty:
        st.error("No successful runs.")
        return
    st.caption(f"Split mode: **{res.split_mode}**. Every strategy in a given (seed, fold, fraction) is evaluated on the "
               "identical held-out test windows.")
    summ = res.summary()
    st.subheader("Summary (mean ± std over seeds/folds)")
    st.dataframe(summ, use_container_width=True, hide_index=True)
    st.plotly_chart(plots.fig_model_comparison(summ, "macro_f1_mean"), use_container_width=True)
    st.plotly_chart(plots.fig_model_comparison(summ, "accuracy_mean"), use_container_width=True)
    with st.expander("All individual runs"):
        st.dataframe(res.metrics, use_container_width=True, hide_index=True)
    c = st.columns(2)
    c[0].download_button("⬇️ Metrics (CSV)", res.metrics.to_csv(index=False).encode(), "experiment_metrics.csv", key=f"dm{key}")
    if len(res.predictions):
        c[1].download_button("⬇️ Window-level predictions (CSV)", res.predictions.to_csv(index=False).encode(),
                             "experiment_predictions.csv", key=f"dp{key}")


def _form(model, advanced: bool):
    files = state.need_files()
    labeled = [n for n in files if files[n].has_labels]
    if not labeled:
        st.warning("None of the processed files has usable ground-truth labels (select a label column on page 1).")
        return
    src_files, source, src_ref = [], model, None
    if advanced:
        st.markdown("##### Source domain")
        src_mode = st.radio("Source model", ["Use the selected pretrained checkpoint", "Train a source model from source CSVs now"],
                            horizontal=True)
        if src_mode.startswith("Train"):
            src_files = st.multiselect("Source-domain files", labeled)
    st.markdown("##### Target domain")
    tgt = st.multiselect("Target-domain files (labelled)", [n for n in labeled if n not in src_files],
                         default=[n for n in labeled if n not in src_files][:1])
    if not tgt:
        return
    avail = _labels_in(files, tgt)
    if model is not None and not src_files:
        avail = [c for c in avail if c in model.classes]
    common = st.multiselect("Common activities", CANONICAL_ACTIVITIES, default=avail)
    c = st.columns(4)
    fracs = c[0].multiselect("Target training fractions", FRACTIONS, default=[0.05, 0.2, 1.0] if not advanced else FRACTIONS,
                             format_func=lambda f: f"{int(f * 100)}%")
    seeds = [int(x) for x in c[1].text_input("Random seeds", "0,1,2" if advanced else "0").replace(" ", "").split(",") if x]
    folds = c[2].number_input("Cross-validation folds (1 = single split)", 1, 10, 1) if advanced else 1
    epochs = c[3].number_input("Max epochs", 1, 300, 40)
    strategies = st.multiselect("Strategies", STRATEGIES, default=STRATEGIES, format_func=STRATEGY_LABELS.get)
    with st.expander("Adaptation hyper-parameters"):
        h = st.columns(4)
        bs = h[0].number_input("Batch size", 4, 256, 32)
        lrf = h[1].number_input("Fine-tune LR", 1e-5, 1e-2, 3e-4, format="%.5f")
        l2 = h[2].number_input("L2-SP λ (adaptive)", 0.0, 10.0, 0.1)
        dev = h[3].selectbox("Device", ["cpu", "cuda"])
    track = st.checkbox("Save results to results/ and log to MLflow" if HAS_MLFLOW else "Save results to results/ (MLflow not installed)", value=advanced)

    if not st.button("▶ Run", type="primary", key=f"run{advanced}"):
        return
    if len(common) < 2 or not fracs or not seeds or not strategies:
        st.error("Select ≥2 common activities, ≥1 fraction, ≥1 seed and ≥1 strategy.")
        return
    cfg = ExperimentConfig(strategies=strategies, fractions=sorted(fracs), seeds=seeds, n_folds=int(folds), common=common,
                           adapt=AdaptConfig(epochs=int(epochs), batch_size=int(bs), lr_finetune=float(lrf), l2sp_lambda=float(l2), device=dev))
    tracker = Tracker("har_experiment") if track else None
    bar = st.progress(0.0, text="Starting…")
    try:
        channels = (model.channels if model is not None else ["acc_x", "acc_y", "acc_z"])
        if src_files:
            spool = build_pool([files[n] for n in src_files], channels, common)
            source = train_source_model(spool, channels, common, cfg.adapt, seeds[0])
            src_ref = spool.X
            st.info(f"Source model trained on {spool.n} windows from {len(set(spool.group))} subject group(s).")
        elif model is not None:
            ref = model.reference()
            src_ref = ref[0] if ref is not None else None
        pool = build_pool([files[n] for n in tgt], channels, common)
        if pool.dropped:
            st.caption(f"Target windows excluded (label not among common activities / unlabeled): {pool.dropped}")
        if source is None and any(s != "scratch" for s in strategies):
            st.error("A pretrained checkpoint is required for the selected strategies.")
            return
        res = run_experiment(source, pool, cfg, src_ref, tracker, lambda p, t: bar.progress(min(p, 1.0), text=t))
        bar.progress(1.0, text="Done")
        if tracker:
            tracker.end()
            st.success(f"Saved to `{tracker.dir}`")
        st.session_state["exp_res"] = res
    except Exception as e:
        st.error(f"Experiment failed: {e}")
        return
    return


def render_compare(model) -> None:
    st.header("6 · Model comparison")
    st.caption("Compares scratch / source-only / frozen extractor / full fine-tuning / adaptive layer-wise adaptation on "
               "identical, subject-wise held-out test windows from your labelled target CSVs.")
    model = state.need_model(model)
    _form(model, advanced=False)
    if "exp_res" in st.session_state:
        _show(st.session_state["exp_res"], "c")


def render_experiment(model) -> None:
    st.header("7 · Research experiment mode")
    st.caption("Subject-wise train/validation/test separation. Normalisation statistics come from the source model "
               "(pretrained strategies) or from the target TRAINING subset only (scratch); validation windows are "
               "taken from training subjects, never from test subjects.")
    if model is None:
        st.info("No pretrained checkpoint available: only the 'scratch' strategy or training a source model from CSVs is possible.")
    _form(model, advanced=True)
    if "exp_res" in st.session_state:
        _show(st.session_state["exp_res"], "e")
