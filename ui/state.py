"""Shared Streamlit session helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import streamlit as st

from config.settings import CHECKPOINT_DIR
from models.predict import Prediction, predict_processed
from models.registry import LoadedModel, list_checkpoints, load_checkpoint
from preprocessing.pipeline import ProcessedFile

NO_MODEL_MSG = (
    "No pretrained model found in `models/checkpoints/`. This project does not ship weights and never fabricates "
    "predictions. Train a source model first, e.g.\n\n"
    "`python -m models.train --uci-har-dir \"datasets/UCI HAR Dataset\" --channels acc`\n\n"
    "(see README.md), then reload this page."
)


def processed() -> Dict[str, ProcessedFile]:
    return st.session_state.setdefault("processed", {})


@st.cache_resource(show_spinner="Loading model…")
def _load(path: str, mtime: float) -> LoadedModel:
    return load_checkpoint(Path(path))


def current_model() -> Optional[LoadedModel]:
    ckpts = list_checkpoints()
    if not ckpts:
        return None
    choice = st.sidebar.selectbox("Model checkpoint", [p.name for p in ckpts], key="ckpt")
    p = CHECKPOINT_DIR / choice
    return _load(str(p), p.stat().st_mtime)


def need_model(model: Optional[LoadedModel]) -> LoadedModel:
    if model is None:
        st.warning(NO_MODEL_MSG)
        st.stop()
    return model


def need_files(min_files: int = 1) -> Dict[str, ProcessedFile]:
    files = processed()
    if len(files) < min_files:
        st.info("Upload and process CSV file(s) on page **1 · Upload & preprocess** first.")
        st.stop()
    return files


def get_prediction(pf: ProcessedFile, model: LoadedModel) -> Prediction:
    cache = st.session_state.setdefault("preds", {})
    key = (pf.uid, str(model.path), model.path.stat().st_mtime if model.path else 0)
    if key not in cache:
        cache[key] = predict_processed(model, pf)
    return cache[key]
