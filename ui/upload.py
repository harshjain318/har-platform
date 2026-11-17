"""Page 1: upload, column mapping, preprocessing."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config.settings import CANONICAL_ACTIVITIES, DATASET_SOURCES
from preprocessing.column_mapper import ColumnMapping, detect_mapping
from preprocessing.labels import auto_label_map
from preprocessing.pipeline import process_dataframe
from preprocessing.reader import read_csv_bytes
from . import state

NONE = "(none)"
EXCLUDE = "(exclude)"


@st.cache_data(show_spinner=False)
def _read(data: bytes, source: str, hdr: str) -> pd.DataFrame:
    return read_csv_bytes(data, source, {"auto-detect": None, "has header": True, "no header": False}[hdr])


def _sel(container, label, cols, default, key, help=None):
    opts = [NONE] + list(cols)
    idx = opts.index(default) if default in opts else 0
    v = container.selectbox(label, opts, index=idx, key=key, help=help)
    return None if v == NONE else v


def _axes(container, title, cols, det_triple, triples, k):
    container.markdown(f"**{title}**")
    presets = ["(manual)"] + list(triples)
    preset = container.selectbox("Quick-fill from detected axis group", presets, key=f"{k}|preset")
    if preset != "(manual)" and st.session_state.get(f"{k}|applied") != preset:
        for ax, c in zip("xyz", triples[preset]):
            st.session_state[f"{k}|{ax}"] = c
        st.session_state[f"{k}|applied"] = preset
    c = container.columns(3)
    out = [_sel(c[i], f"{ax.upper()} axis", cols, det_triple[i] if det_triple else None, f"{k}|{ax}")
           for i, ax in enumerate("xyz")]
    return tuple(out) if all(out) else None


def _panel(name: str, data: bytes) -> None:
    proc = state.processed()
    with st.expander(f"📄 {name}", expanded=name not in proc):
        c1, c2 = st.columns(2)
        source = c1.selectbox("Dataset source", DATASET_SOURCES, key=f"{name}|source",
                              help="Used for label harmonisation defaults (e.g. UCI numeric codes, WISDM letters).")
        hdr = c2.selectbox("Header row", ["auto-detect", "has header", "no header"], key=f"{name}|hdr")
        try:
            df = _read(data, source, hdr)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            return
        st.dataframe(df.head(20), use_container_width=True)
        st.caption(f"{len(df):,} rows × {df.shape[1]} columns")
        det = detect_mapping(df, source)
        for n in det.notes:
            st.info(n)
        d, cols, k = det.mapping, list(df.columns), f"{name}|{source}|{hdr}"

        st.markdown("##### Column mapping (auto-detected — edit if wrong)")
        c = st.columns(4)
        time_col = _sel(c[0], "Time column", cols, d.time, f"{k}|time")
        subj_col = _sel(c[1], "Subject column", cols, d.subject, f"{k}|subj")
        lab_col = _sel(c[2], "Activity-label column (ground truth)", cols, d.label, f"{k}|lab")
        sens_col = _sel(c[3], "Sensor-type column (long format)", cols, d.sensor_type_col, f"{k}|sens")
        acc_val = gyr_val = None
        if sens_col:
            vals = [str(v) for v in df[sens_col].dropna().unique()][:50]
            cc = st.columns(2)
            acc_val = _sel(cc[0], "Value = accelerometer", vals, d.acc_sensor_value, f"{k}|accv")
            gyr_val = _sel(cc[1], "Value = gyroscope", vals, d.gyro_sensor_value, f"{k}|gyrv")
        a1, a2 = st.columns(2)
        acc = _axes(a1, "Accelerometer", cols, d.acc, det.triples, f"{k}|acc")
        gyro = _axes(a2, "Gyroscope (optional)", cols, d.gyro, det.triples, f"{k}|gyro")
        grav = None
        if d.gravity and st.checkbox("Add detected gravity columns to the (gravity-free) acceleration", value=True, key=f"{k}|grav"):
            grav = d.gravity

        st.markdown("##### Timing & units")
        u = st.columns(4)
        t_unit = u[0].selectbox("Time unit", ["auto", "s", "ms", "us", "ns", "datetime"], key=f"{k}|tu")
        fs = u[1].number_input("Sampling freq. (Hz) — only if no usable time column", 0.0, 10000.0, 0.0, key=f"{k}|fs")
        acc_u = u[2].selectbox("Accelerometer unit", ["auto", "g", "m/s2", "mg"], key=f"{k}|au")
        gyr_u = u[3].selectbox("Gyroscope unit", ["auto", "rad/s", "deg/s"], key=f"{k}|gu")

        label_map = None
        if lab_col:
            uniq = [str(v).strip() for v in pd.Series(df[lab_col]).astype(str).str.strip().unique()]
            if len(uniq) > 60:
                st.error("The label column has more than 60 distinct values — wrong column?")
                return
            auto = auto_label_map(uniq, source)
            ed = pd.DataFrame({"raw label": list(auto), "activity": [v or EXCLUDE for v in auto.values()]})
            st.markdown("##### Label mapping (ground truth → model vocabulary)")
            ed = st.data_editor(ed, hide_index=True, disabled=["raw label"], key=f"{k}|lmap", use_container_width=True,
                                column_config={"activity": st.column_config.SelectboxColumn(
                                    "activity", options=CANONICAL_ACTIVITIES + [EXCLUDE], required=True)})
            label_map = {r["raw label"]: (None if r["activity"] == EXCLUDE else r["activity"]) for _, r in ed.iterrows()}

        if st.button("⚙️ Process file", key=f"{k}|go", type="primary"):
            mapping = ColumnMapping(time=time_col, subject=subj_col, label=lab_col, acc=acc, gyro=gyro, gravity=grav,
                                    sensor_type_col=sens_col, acc_sensor_value=acc_val, gyro_sensor_value=gyr_val,
                                    time_unit=t_unit, fs=fs or None, acc_unit=acc_u, gyro_unit=gyr_u)
            try:
                with st.spinner("Preprocessing…"):
                    proc[name] = process_dataframe(df, mapping, name, source, label_map)
                st.session_state.get("preds", {}).clear()
            except Exception as e:
                proc.pop(name, None)
                st.error(f"Preprocessing failed: {e}")
        pf = proc.get(name)
        if pf is not None:
            s = pf.stats
            m = st.columns(6)
            m[0].metric("Original samples", f"{s['original_acc_samples']:,}")
            m[1].metric("Resampled samples", f"{s['resampled_samples']:,}")
            m[2].metric("Source fs (Hz)", s["original_fs_hz"])
            m[3].metric("Windows (128, 50 %)", s["n_windows"])
            m[4].metric("Missing values", s["missing_values"])
            m[5].metric("Segments / subjects", f"{s['n_segments']} / {s['n_subjects']}")
            st.write(f"**Selected channels:** {', '.join(s['channels'])} · acc unit `{s['acc_unit']}`"
                     + (f" · gyro unit `{s['gyro_unit']}`" if s["gyro_unit"] else "") + f" · {s['time']}"
                     f" · duplicate timestamps merged: {s['duplicates_removed']}")
            for w in pf.warnings:
                st.warning(w)
            st.success("Ready ✔  Windows never cross subject or activity boundaries.")


def render() -> None:
    st.header("1 · Upload & preprocess")
    st.caption("Any CSV with accelerometer (and optionally gyroscope) columns. Everything is converted to g / rad/s, "
               "resampled to 50 Hz and cut into 128-sample windows with 50 % overlap inside each subject × activity segment.")
    ups = st.file_uploader("Upload one or more CSV files", type=["csv", "txt"], accept_multiple_files=True)
    raw = {u.name: u.getvalue() for u in ups}
    proc = state.processed()
    for n in [n for n in proc if n not in raw]:
        del proc[n]
    if not raw:
        st.info("Upload files to begin. Supported layouts include Formats A–D from the specification, UCI HAR, WISDM, "
                "MotionSense, HHAR and custom files.")
        return
    for name, data in raw.items():
        _panel(name, data)
    if proc:
        st.subheader("Preprocessing summary")
        st.dataframe(pd.DataFrame([{
            "file": n, "source": p.source, "original samples": p.stats["original_acc_samples"],
            "source fs (Hz)": p.stats["original_fs_hz"], "resampled samples": p.stats["resampled_samples"],
            "windows": p.stats["n_windows"], "missing values": p.stats["missing_values"],
            "channels": ", ".join(p.channels), "labels": "yes" if p.has_labels else "no"} for n, p in proc.items()]),
            use_container_width=True, hide_index=True)
