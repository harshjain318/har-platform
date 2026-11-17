"""Plotly figures. Every function returns a plotly Figure (no Streamlit dependency)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from evaluation.domain import mean_psd

MAX_POINTS = 20000
PALETTE = px.colors.qualitative.Safe


def _dec(t: np.ndarray, y: np.ndarray, max_points: int = MAX_POINTS):
    if len(t) <= max_points:
        return t, y
    k = int(np.ceil(len(t) / max_points))
    return t[::k], y[::k]


def _slice(pf, t_range):
    t = pf.timeline
    if t_range is None:
        m = np.ones(len(t), bool)
    else:
        m = (t >= t_range[0]) & (t <= t_range[1])
    return t[m], pf.signal[m], m


def fig_axis(pf, channel: str, t_range=None) -> go.Figure:
    t, s, _ = _slice(pf, t_range)
    j = pf.channels.index(channel)
    tt, yy = _dec(t, s[:, j])
    unit = "g" if channel.startswith("acc") else "rad/s"
    f = go.Figure(go.Scattergl(x=tt, y=yy, mode="lines", name=channel))
    f.update_layout(title=f"{channel} vs time", xaxis_title="Timeline (s, resampled 50 Hz)", yaxis_title=unit,
                    height=300, margin=dict(l=10, r=10, t=40, b=10))
    return f


def fig_combined(pf, kind: str = "acc", t_range=None) -> go.Figure:
    t, s, _ = _slice(pf, t_range)
    chans = [c for c in pf.channels if c.startswith(kind)]
    f = go.Figure()
    for c in chans:
        tt, yy = _dec(t, s[:, pf.channels.index(c)])
        f.add_trace(go.Scattergl(x=tt, y=yy, mode="lines", name=c))
    f.update_layout(title=f"Combined {'accelerometer' if kind == 'acc' else 'gyroscope'} (3 axes)",
                    xaxis_title="Timeline (s)", yaxis_title="g" if kind == "acc" else "rad/s", height=340,
                    margin=dict(l=10, r=10, t=40, b=10))
    return f


def magnitude(pf, kind: str = "acc") -> np.ndarray:
    idx = [pf.channels.index(c) for c in pf.channels if c.startswith(kind)]
    return np.linalg.norm(pf.signal[:, idx], axis=1)


def fig_magnitude(pf, t_range=None) -> go.Figure:
    t, _, m = _slice(pf, t_range)
    f = go.Figure()
    tt, yy = _dec(t, magnitude(pf, "acc")[m])
    f.add_trace(go.Scattergl(x=tt, y=yy, mode="lines", name="|acc| (g)"))
    if pf.has_gyro:
        tt, yy = _dec(t, magnitude(pf, "gyro")[m])
        f.add_trace(go.Scattergl(x=tt, y=yy, mode="lines", name="|gyro| (rad/s)", yaxis="y2"))
        f.update_layout(yaxis2=dict(overlaying="y", side="right", title="rad/s"))
    f.update_layout(title="Signal magnitude", xaxis_title="Timeline (s)", yaxis_title="g", height=320,
                    margin=dict(l=10, r=10, t=40, b=10))
    return f


def fig_fft(pf, t_range=None, kind: str = "acc") -> go.Figure:
    _, s, m = _slice(pf, t_range)
    f = go.Figure()
    chans = [c for c in pf.channels if c.startswith(kind)]
    for c in chans:
        x = s[:, pf.channels.index(c)].astype(float)
        if len(x) < 8:
            continue
        x = (x - x.mean()) * np.hanning(len(x))
        spec = np.abs(np.fft.rfft(x)) / len(x)
        fr = np.fft.rfftfreq(len(x), 1 / pf.fs)
        f.add_trace(go.Scatter(x=fr, y=spec, mode="lines", name=c))
    f.update_layout(title="Frequency-domain representation (FFT magnitude, Hann window)", xaxis_title="Frequency (Hz)",
                    yaxis_title="Amplitude", height=320, margin=dict(l=10, r=10, t=40, b=10))
    return f


def fig_predictions_timeline(pred, smoothed_idx: Optional[np.ndarray] = None, conf_thr: float = 0.6) -> go.Figure:
    idx = pred.pred_idx if smoothed_idx is None else smoothed_idx
    classes = pred.classes
    t0 = pred.start_s
    f = go.Figure()
    for k, c in enumerate(classes):
        m = idx == k
        if m.any():
            f.add_trace(go.Scatter(x=t0[m], y=[c] * int(m.sum()), mode="markers", name=c,
                                   marker=dict(size=9, color=PALETTE[k % len(PALETTE)]),
                                   customdata=np.c_[pred.conf[m]], hovertemplate="t=%{x:.1f}s<br>conf=%{customdata[0]:.2f}"))
    low = pred.conf < conf_thr
    if low.any():
        f.add_trace(go.Scatter(x=t0[low], y=[classes[i] for i in idx[low]], mode="markers", name=f"conf < {conf_thr:.2f}",
                               marker=dict(size=14, symbol="circle-open", color="black", line=dict(width=2))))
    if any(v is not None for v in pred.y_true):
        yt = [v if v in classes else None for v in pred.y_true]
        f.add_trace(go.Scatter(x=t0, y=yt, mode="markers", name="ground truth",
                               marker=dict(size=6, symbol="line-ew", color="black", line=dict(width=2))))
    f.update_layout(title="Sliding-window activity predictions (window start time)", xaxis_title="Timeline (s)",
                    yaxis=dict(categoryorder="array", categoryarray=classes), height=380, margin=dict(l=10, r=10, t=40, b=10))
    return f


def fig_prob_heatmap(pred) -> go.Figure:
    f = go.Figure(go.Heatmap(z=pred.probs.T, x=pred.start_s, y=pred.classes, zmin=0, zmax=1, colorscale="Viridis",
                             colorbar=dict(title="P")))
    f.update_layout(title="Class probability per window (softmax output, not calibrated)", xaxis_title="Timeline (s)",
                    height=320, margin=dict(l=10, r=10, t=40, b=10))
    return f


def fig_activity_share(df: pd.DataFrame, title: str = "Share of windows per predicted activity") -> go.Figure:
    f = px.bar(df, x="activity", y="percent_of_windows", text=df["percent_of_windows"].round(1).astype(str) + "%",
               title=title)
    f.update_layout(height=320, yaxis_title="% of windows", margin=dict(l=10, r=10, t=40, b=10))
    return f


def fig_confusion(cm: np.ndarray, labels: List[str], title: str = "Confusion matrix", normalize: bool = False) -> go.Figure:
    z = cm.astype(float)
    if normalize:
        z = z / np.maximum(z.sum(1, keepdims=True), 1)
    text = np.array([[f"{v:.2f}" if normalize else f"{int(v)}" for v in row] for row in z])
    f = go.Figure(go.Heatmap(z=z, x=labels, y=labels, text=text, texttemplate="%{text}", colorscale="Blues"))
    f.update_layout(title=title, xaxis_title="Predicted", yaxis_title="True", height=420,
                    yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=40, b=10))
    return f


def fig_per_class(df: pd.DataFrame, metric: str, title: str) -> go.Figure:
    """df columns: dataset, class, <metric>."""
    f = px.bar(df, x="class", y=metric, color="dataset", barmode="group", title=title)
    f.update_layout(height=340, yaxis_range=[0, 1.05], margin=dict(l=10, r=10, t=40, b=10))
    return f


def fig_stacked_pred_dist(dist: pd.DataFrame) -> go.Figure:
    """dist: rows = file, columns = class, values = percent."""
    f = go.Figure()
    for k, c in enumerate(dist.columns):
        f.add_trace(go.Bar(name=c, x=dist.index, y=dist[c], marker_color=PALETTE[k % len(PALETTE)]))
    f.update_layout(barmode="stack", title="Predicted activity distribution per file", yaxis_title="% of windows",
                    height=360, margin=dict(l=10, r=10, t=40, b=10))
    return f


def fig_heatmap(mat: pd.DataFrame, title: str, colorscale: str = "Reds") -> go.Figure:
    f = go.Figure(go.Heatmap(z=mat.values, x=list(mat.columns), y=list(mat.index), colorscale=colorscale,
                             text=np.round(mat.values, 3), texttemplate="%{text}"))
    f.update_layout(title=title, height=380, yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=40, b=10))
    return f


def fig_magnitude_hist(series: Dict[str, np.ndarray]) -> go.Figure:
    f = go.Figure()
    for k, (name, v) in enumerate(series.items()):
        f.add_trace(go.Histogram(x=v[:: max(1, len(v) // 50000)], name=name, histnorm="probability density",
                                 opacity=0.55, nbinsx=80, marker_color=PALETTE[k % len(PALETTE)]))
    f.update_layout(barmode="overlay", title="Acceleration-magnitude distribution (|acc| in g)", xaxis_title="|acc| (g)",
                    height=340, margin=dict(l=10, r=10, t=40, b=10))
    return f


def fig_psd(series: Dict[str, np.ndarray], fs: float = 50.0) -> go.Figure:
    f = go.Figure()
    for k, (name, v) in enumerate(series.items()):
        fr, p = mean_psd(v, fs)
        f.add_trace(go.Scatter(x=fr, y=p, mode="lines", name=name, line=dict(color=PALETTE[k % len(PALETTE)])))
    f.update_layout(title="Power spectral density of |acc| (Welch)", xaxis_title="Frequency (Hz)", yaxis_title="PSD",
                    yaxis_type="log", height=340, margin=dict(l=10, r=10, t=40, b=10))
    return f


def fig_embedding(emb: np.ndarray, domain: Sequence[str], title: str, activity: Optional[Sequence[str]] = None) -> go.Figure:
    df = pd.DataFrame({"x": emb[:, 0], "y": emb[:, 1], "domain": domain})
    if activity is not None:
        df["activity"] = [a if a else "unlabelled" for a in activity]
        f = px.scatter(df, x="x", y="y", color="domain", symbol="activity", title=title, opacity=0.7)
    else:
        f = px.scatter(df, x="x", y="y", color="domain", title=title, opacity=0.7)
    f.update_layout(height=480, margin=dict(l=10, r=10, t=40, b=10))
    return f


def fig_layer_mmd(table: pd.DataFrame) -> go.Figure:
    """table columns: file, layer, mmd2."""
    f = px.bar(table, x="layer", y="mmd2", color="file", barmode="group", title="Layer-wise MMD² (training reference vs file)")
    f.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
    return f


def fig_model_comparison(summary: pd.DataFrame, metric: str = "macro_f1_mean") -> go.Figure:
    f = px.line(summary, x="fraction", y=metric, color="strategy", markers=True,
                title=f"{metric.replace('_mean', '').replace('_', ' ')} vs target training fraction")
    f.update_layout(xaxis_type="log", height=380, margin=dict(l=10, r=10, t=40, b=10))
    return f
