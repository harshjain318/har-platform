"""Page 2: interactive signal visualisation."""
from __future__ import annotations

import streamlit as st

from visualization import plots
from . import state


def render(model) -> None:
    st.header("2 · Signal visualisation")
    files = state.need_files()
    name = st.selectbox("File", list(files))
    pf = files[name]
    t_end = float(pf.timeline[-1]) if len(pf.timeline) else 1.0
    lo, hi = st.slider("Time interval to inspect (s, resampled timeline)", 0.0, t_end, (0.0, min(t_end, 60.0)), step=0.5)
    rng = (lo, hi)
    st.caption("The timeline concatenates the segments of the file (subject × activity runs); "
               "segment boundaries are listed below.")
    tabs = st.tabs(["Accelerometer X/Y/Z", "Combined 3-axis", "Gyroscope", "Magnitude", "FFT", "Window predictions"])
    with tabs[0]:
        for c in ("acc_x", "acc_y", "acc_z"):
            st.plotly_chart(plots.fig_axis(pf, c, rng), use_container_width=True)
    with tabs[1]:
        st.plotly_chart(plots.fig_combined(pf, "acc", rng), use_container_width=True)
    with tabs[2]:
        if pf.has_gyro:
            st.plotly_chart(plots.fig_combined(pf, "gyro", rng), use_container_width=True)
        else:
            st.info("This file has no gyroscope channels.")
    with tabs[3]:
        st.plotly_chart(plots.fig_magnitude(pf, rng), use_container_width=True)
    with tabs[4]:
        st.plotly_chart(plots.fig_fft(pf, rng, "acc"), use_container_width=True)
        if pf.has_gyro:
            st.plotly_chart(plots.fig_fft(pf, rng, "gyro"), use_container_width=True)
    with tabs[5]:
        if model is None:
            st.warning(state.NO_MODEL_MSG)
        elif pf.windows.n == 0:
            st.warning("No windows in this file.")
        else:
            try:
                pred = state.get_prediction(pf, model)
                f = plots.fig_predictions_timeline(pred)
                f.update_xaxes(range=[lo, hi])
                st.plotly_chart(f, use_container_width=True)
            except Exception as e:
                st.error(str(e))
    with st.expander("Segments in this file"):
        import pandas as pd
        st.dataframe(pd.DataFrame([{"segment": i, "subject": s.subject, "label (raw)": s.raw_label,
                                    "label (mapped)": s.label, "start_s": s.start / pf.fs, "duration_s": s.n / pf.fs,
                                    "windows": s.n_windows} for i, s in enumerate(pf.segments)]),
                     use_container_width=True, hide_index=True)
