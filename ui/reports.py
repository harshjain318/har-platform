"""Page 8: bundle downloadable reports."""
from __future__ import annotations

import io
import json
import zipfile

import streamlit as st

from evaluation.report import evaluation_dict, prediction_table, run_report
from evaluation.summary import activity_summary
from . import state


def render(model) -> None:
    st.header("8 · Reports")
    model = state.need_model(model)
    files = state.need_files()
    names = st.multiselect("Files to include", list(files), default=list(files))
    if not names:
        return
    buf = io.BytesIO()
    per_file = {}
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            pf = files[n]
            try:
                p = state.get_prediction(pf, model)
            except Exception as e:
                st.warning(f"{n}: {e}")
                continue
            ev = p.evaluate()
            per_file[n] = {"stats": pf.stats, "summary": activity_summary(p.pred, p.classes), "evaluation": evaluation_dict(ev)}
            z.writestr(f"{n}/predictions.csv", prediction_table(p).to_csv(index=False))
            z.writestr(f"{n}/preprocessing.json", json.dumps(pf.stats, indent=2, default=str))
            z.writestr(f"{n}/evaluation.json", json.dumps(per_file[n]["evaluation"], indent=2, default=str))
        z.writestr("report.md", run_report(per_file, model.meta))
        if "comparison_table" in st.session_state:
            z.writestr("comparison.csv", st.session_state["comparison_table"].to_csv(index=False))
        if "domain_report_md" in st.session_state:
            z.writestr("domain_shift_report.md", st.session_state["domain_report_md"])
    st.write("The bundle contains, per file: window-level predictions, preprocessing statistics and ground-truth evaluation "
             "(only when labels exist), plus a Markdown report and, if generated, the comparison table and domain-shift report.")
    st.download_button("⬇️ Download report bundle (ZIP)", buf.getvalue(), "har_report.zip", "application/zip", type="primary")
