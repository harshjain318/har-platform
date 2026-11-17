"""HAR Platform – Streamlit entry point.   Launch:  streamlit run app.py"""
import streamlit as st

from config.settings import ensure_dirs

st.set_page_config(page_title="HAR Platform", page_icon="🏃", layout="wide")
ensure_dirs()

from ui import compare, domain, experiment, predict, reports, signals, state, upload  # noqa: E402

PAGES = {
    "1 · Upload & preprocess": lambda m: upload.render(),
    "2 · Signal visualisation": signals.render,
    "3 · Activity prediction": predict.render,
    "4 · Multi-CSV comparison": compare.render,
    "5 · Domain analysis": domain.render,
    "6 · Model comparison": experiment.render_compare,
    "7 · Research experiments": experiment.render_experiment,
    "8 · Reports": reports.render,
}

st.sidebar.title("🏃 HAR Platform")
page = st.sidebar.radio("Navigation", list(PAGES))
model = state.current_model()
if model is not None:
    with st.sidebar.expander("Model info", expanded=False):
        st.write(f"**Classes:** {', '.join(model.classes)}")
        st.write(f"**Input channels:** {', '.join(model.channels)}")
        meta = model.meta
        st.write(f"**Trained on:** {meta.get('dataset', 'n/a')} ({meta.get('n_train_subjects', '?')} subjects)")
        if meta.get("test_macro_f1") is not None:
            st.write(f"Held-out test at training time: acc {meta['test_accuracy']:.3f}, macro-F1 {meta['test_macro_f1']:.3f}")
        st.caption("Window 128 @ 50 Hz, 50 % overlap. Normalisation statistics are stored in the checkpoint.")
else:
    st.sidebar.warning("No checkpoint found – see README (train a source model).")
PAGES[page](model)
