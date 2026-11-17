"""Build downloadable prediction / evaluation reports. Predictions and ground-truth metrics are kept apart."""
from __future__ import annotations

import datetime as dt
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def prediction_table(pred) -> pd.DataFrame:
    df = pd.DataFrame({
        "window": np.arange(len(pred.pred_idx)), "subject": pred.subject, "segment": pred.segment,
        "start_s": pred.start_s, "predicted_activity": pred.pred, "confidence_max_softmax": pred.conf,
        "entropy": pred.entropy,
        "ground_truth": [v if v is not None else "" for v in pred.y_true],
    })
    for k, c in enumerate(pred.classes):
        df[f"p_{c}"] = pred.probs[:, k]
    return df


def evaluation_dict(ev: Optional[Dict]) -> Dict:
    if ev is None:
        return {"ground_truth_available": False,
                "note": "No valid ground-truth labels: accuracy/macro-F1 intentionally not reported."}
    return {"ground_truth_available": True, "accuracy": ev["accuracy"], "macro_f1": ev["macro_f1"],
            "n_evaluated_windows": ev["n_evaluated"], "n_excluded_windows": ev["n_excluded"],
            "excluded_labels": ev["excluded_labels"], "expected_calibration_error": ev.get("ece"),
            "per_class": ev["per_class"].to_dict(orient="records"),
            "confusion_matrix": {"labels": ev["cm_labels"], "matrix": ev["confusion_matrix"].tolist()},
            "macro_f1_definition": "mean F1 over classes present in the ground truth"}


def findings_markdown(name: str, findings: List[Dict]) -> str:
    icon = {"ok": "🟢", "warn": "🟡", "high": "🔴"}
    lines = [f"## Domain-shift report — {name}", ""]
    for f in findings:
        lines += [f"- {icon.get(f['severity'], '')} **{f['aspect']}** — {f['evidence']}",
                  f"  - Possible cause (hypothesis): {f['possible_cause']}"]
    return "\n".join(lines) + "\n"


def run_report(files: Dict[str, dict], model_meta: dict) -> str:
    """files: name -> {'stats', 'summary': DataFrame, 'evaluation': dict}"""
    out = ["# HAR platform report", f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}", "",
           "Model training metadata (as stored in the checkpoint; held-out subjects at training time):", "```json",
           json.dumps(model_meta, indent=2, default=str), "```", ""]
    for n, d in files.items():
        out += [f"## {n}", "", "**Preprocessing**", "```json", json.dumps(d["stats"], indent=2, default=str), "```", "",
                "**Predicted activity distribution (model output, no ground truth involved)**", "",
                d["summary"].to_markdown(index=False) if hasattr(d["summary"], "to_markdown") else d["summary"].to_string(index=False), "",
                "**Ground-truth evaluation**", "```json",
                json.dumps({k: v for k, v in d["evaluation"].items() if k not in ("per_class", "confusion_matrix")}, indent=2, default=str),
                "```", ""]
    return "\n".join(out)
