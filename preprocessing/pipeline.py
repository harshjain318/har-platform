"""Unified preprocessing: mapping -> units -> dedupe/sort -> segmentation -> 50 Hz -> windows."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import ACC_UNITS, GYRO_UNITS, STRIDE, TARGET_FS, WINDOW
from .column_mapper import ColumnMapping, parse_time_seconds, to_numeric_array
from .labels import auto_label_map
from .resample import resample_to_grid
from .units import guess_acc_unit, guess_gyro_unit
from .windowing import sliding_windows


@dataclass
class Segment:
    subject: str
    raw_label: str
    label: Optional[str]
    start: int          # offset in the concatenated signal
    n: int
    n_windows: int


@dataclass
class WindowSet:
    X: np.ndarray                 # (N, window, C) float32, standard units, NOT normalised
    y: np.ndarray                 # (N,) object: canonical label or None
    subject: np.ndarray           # (N,) str
    segment: np.ndarray           # (N,) int segment id
    start: np.ndarray             # (N,) start index in the concatenated signal
    channels: List[str]

    @property
    def n(self) -> int:
        return len(self.X)

    def select(self, wanted: List[str]) -> np.ndarray:
        missing = [c for c in wanted if c not in self.channels]
        if missing:
            raise ValueError(f"Required channels {missing} are not available in this file "
                             f"(available: {self.channels}).")
        return self.X[:, :, [self.channels.index(c) for c in wanted]]


@dataclass
class ProcessedFile:
    name: str
    source: str
    mapping: ColumnMapping
    channels: List[str]
    signal: np.ndarray            # (n, C) concatenated resampled segments
    sample_segment: np.ndarray    # (n,) segment id per sample
    segments: List[Segment]
    windows: WindowSet
    stats: dict
    warnings: List[str]
    label_map: Dict[str, Optional[str]]
    fs: float = TARGET_FS
    uid: str = field(default_factory=lambda: uuid.uuid4().hex)

    @property
    def has_labels(self) -> bool:
        return any(y is not None for y in self.windows.y)

    @property
    def has_gyro(self) -> bool:
        return "gyro_x" in self.channels

    @property
    def timeline(self) -> np.ndarray:
        return np.arange(len(self.signal)) / self.fs

    def sample_labels(self) -> np.ndarray:
        lab = np.array([s.label for s in self.segments], dtype=object)
        return lab[self.sample_segment] if len(lab) else np.array([], dtype=object)

    def sample_subjects(self) -> np.ndarray:
        sub = np.array([s.subject for s in self.segments], dtype=object)
        return sub[self.sample_segment] if len(sub) else np.array([], dtype=object)


def _time_axis(df, mapping, subject, n_rows):
    """Return time in seconds (file order) and a note."""
    if mapping.time is None:
        if not mapping.fs:
            raise ValueError("No time column mapped: please specify the sampling frequency.")
        counter = pd.Series(subject).groupby(subject).cumcount().to_numpy(dtype=float)
        return counter / float(mapping.fs), f"time reconstructed from fs={mapping.fs} Hz"
    sec, unit, is_index = parse_time_seconds(df[mapping.time], mapping.time_unit)
    if is_index:
        if not mapping.fs:
            raise ValueError("The time column looks like a sample index: please specify the sampling frequency.")
        return sec / float(mapping.fs), f"time column is a sample index; fs={mapping.fs} Hz used"
    return sec, f"time unit: {unit}"


def process_dataframe(df: pd.DataFrame, mapping: ColumnMapping, name: str = "file", source: str = "Custom",
                      label_map: Optional[Dict[str, Optional[str]]] = None, target_fs: float = TARGET_FS,
                      window: int = WINDOW, stride: int = STRIDE, max_gap_s: float = 1.0) -> ProcessedFile:
    if mapping.acc is None:
        raise ValueError("Accelerometer X/Y/Z columns must be mapped.")
    df = df.reset_index(drop=True)
    n_rows = len(df)
    warnings: List[str] = []
    missing = 0

    subject = (df[mapping.subject].astype(str).str.strip().to_numpy(dtype=object)
               if mapping.subject else np.full(n_rows, "subject_0", dtype=object))
    if mapping.label:
        raw_lab = df[mapping.label].astype(str).str.strip().to_numpy(dtype=object)
        lmap = label_map if label_map is not None else auto_label_map(np.unique(raw_lab), source)
    else:
        raw_lab = np.full(n_rows, "", dtype=object)
        lmap = {}
    time_s, time_note = _time_axis(df, mapping, subject, n_rows)

    long_fmt = mapping.sensor_type_col is not None
    acc_mask = gyro_mask = None
    if long_fmt:
        if mapping.time is None:
            raise ValueError("Long (sensor-type) format requires a time column.")
        stype = df[mapping.sensor_type_col].astype(str).str.strip().to_numpy(dtype=object)
        if mapping.acc_sensor_value is None:
            raise ValueError("Select which value of the sensor-type column is the accelerometer.")
        acc_mask = stype == str(mapping.acc_sensor_value)
        if mapping.gyro is not None and mapping.gyro_sensor_value is not None:
            gyro_mask = stype == str(mapping.gyro_sensor_value)
        if acc_mask.sum() == 0:
            raise ValueError(f"No rows with sensor type '{mapping.acc_sensor_value}'.")

    def extract(triple, mask, grav=None):
        nonlocal missing
        idx = np.arange(n_rows) if mask is None else np.flatnonzero(mask)
        v = np.column_stack([to_numeric_array(df[c])[idx] for c in triple])
        g = np.column_stack([to_numeric_array(df[c])[idx] for c in grav]) if grav else None
        t = time_s[idx]
        bad = ~np.isfinite(v).all(1) | ~np.isfinite(t)
        if g is not None:
            bad |= ~np.isfinite(g).all(1)
        missing += int((~np.isfinite(v)).sum() + (~np.isfinite(t)).sum())
        k = ~bad
        return idx[k], t[k], v[k], (g[k] if g is not None else None)

    def frame(idx, t, v, with_label=True):
        return pd.DataFrame({"s": subject[idx], "t": t, "l": raw_lab[idx] if with_label else "",
                             "a": v[:, 0], "b": v[:, 1], "c": v[:, 2]})

    def dedupe(fr):
        n0 = len(fr)
        out = fr.groupby(["s", "t"], sort=True, as_index=False).agg(
            l=("l", "first"), a=("a", "mean"), b=("b", "mean"), c=("c", "mean"))
        return out, n0 - len(out)

    # ---- accelerometer
    ia, ta, va, ga = extract(mapping.acc, acc_mask, mapping.gravity)
    if len(ia) < 2:
        raise ValueError("Fewer than 2 valid accelerometer samples after removing missing values.")
    acc_unit = mapping.acc_unit if mapping.acc_unit != "auto" else guess_acc_unit(va)
    f_acc = ACC_UNITS[acc_unit]
    va = va * f_acc + (ga * f_acc if ga is not None else 0.0)
    fa, dup_removed = dedupe(frame(ia, ta, va))
    n_acc_samples = len(ia)

    # ---- gyroscope (optional)
    gyro_by_subject: Dict[str, tuple] = {}
    gyro_unit = None
    if mapping.gyro is not None:
        ig, tg, vg, _ = extract(mapping.gyro, gyro_mask)
        if len(ig) >= 2:
            gyro_unit = mapping.gyro_unit if mapping.gyro_unit != "auto" else guess_gyro_unit(vg)
            vg = vg * GYRO_UNITS[gyro_unit]
            fg, _ = dedupe(frame(ig, tg, vg, with_label=False))
            for key, grp in fg.groupby("s", sort=False):
                gyro_by_subject[str(key)] = (grp["t"].to_numpy(), grp[["a", "b", "c"]].to_numpy())
        else:
            warnings.append("Gyroscope columns were mapped but contain too few valid samples; gyroscope ignored.")
    use_gyro = bool(gyro_by_subject)

    # ---- segmentation (subject change, label change, time gap)
    s, t, l = fa["s"].to_numpy(), fa["t"].to_numpy(), fa["l"].to_numpy()
    v_all = fa[["a", "b", "c"]].to_numpy()
    dt = np.diff(t)
    same = s[1:] == s[:-1]
    pos = dt[same & (dt > 0)]
    med_dt = float(np.median(pos)) if len(pos) else 1.0 / target_fs
    gap = max(max_gap_s, 5.0 * med_dt)
    brk = (~same) | (l[1:] != l[:-1]) | (dt > gap)
    starts = np.r_[0, np.flatnonzero(brk) + 1]
    ends = np.r_[starts[1:], len(t)]
    orig_fs = 1.0 / med_dt

    chunks, segs, sample_seg = [], [], []
    win_X, win_y, win_s, win_seg, win_start = [], [], [], [], []
    offset = 0
    n_short = n_gyro_drop = n_tiny = 0
    for a, b in zip(starts, ends):
        if b - a < 2:
            n_tiny += 1
            continue
        t0, t1 = t[a], t[b - 1]
        n_out = int(np.floor((t1 - t0) * target_fs + 1e-9)) + 1
        grid = t0 + np.arange(n_out) / target_fs
        arr = resample_to_grid(t[a:b], v_all[a:b], grid, target_fs)
        if use_gyro:
            gt, gv = gyro_by_subject.get(str(s[a]), (None, None))
            ok = False
            if gt is not None:
                lo, hi = np.searchsorted(gt, t0 - gap), np.searchsorted(gt, t1 + gap, side="right")
                gt_s, gv_s = gt[lo:hi], gv[lo:hi]
                ok = (len(gt_s) >= 2 and gt_s[0] <= t0 + gap and gt_s[-1] >= t1 - gap
                      and np.max(np.diff(gt_s)) <= gap)
            if not ok:
                n_gyro_drop += 1
                continue
            arr = np.hstack([arr, resample_to_grid(gt_s, gv_s, grid, target_fs)])
        sid = len(segs)
        W, st = sliding_windows(arr, window, stride)
        raw = str(l[a])
        canon = lmap.get(raw) if mapping.label else None
        segs.append(Segment(str(s[a]), raw, canon, offset, len(arr), len(W)))
        chunks.append(arr)
        sample_seg.append(np.full(len(arr), sid, dtype=np.int32))
        if len(W):
            win_X.append(W)
            win_y.extend([canon] * len(W))
            win_s.extend([str(s[a])] * len(W))
            win_seg.extend([sid] * len(W))
            win_start.extend((offset + st).tolist())
        else:
            n_short += 1
        offset += len(arr)

    if not chunks:
        raise ValueError("No usable segments were produced (check time column, gaps, subjects and gyroscope coverage).")
    signal = np.vstack(chunks).astype(np.float32)
    channels = ["acc_x", "acc_y", "acc_z"] + (["gyro_x", "gyro_y", "gyro_z"] if use_gyro else [])
    ws = WindowSet(
        X=np.concatenate(win_X) if win_X else np.zeros((0, window, len(channels)), dtype=np.float32),
        y=np.array(win_y, dtype=object), subject=np.array(win_s, dtype=object),
        segment=np.array(win_seg, dtype=int), start=np.array(win_start, dtype=int), channels=channels)

    # ---- diagnostics
    if abs(orig_fs / target_fs - 1) > 0.1:
        warnings.append(f"Estimated source sampling rate is {orig_fs:.1f} Hz; resampled to {target_fs:.0f} Hz.")
    if orig_fs < 0.8 * target_fs:
        warnings.append("Source rate is below 50 Hz: upsampling adds no new information.")
    if n_short:
        warnings.append(f"{n_short} segment(s) shorter than {window} samples produced no windows.")
    if n_gyro_drop:
        warnings.append(f"{n_gyro_drop} segment(s) dropped because gyroscope coverage was incomplete.")
    if ws.n == 0:
        warnings.append("No windows were generated: every segment is shorter than the window length.")
    mag = np.linalg.norm(signal[:, :3], axis=1)
    if len(mag) and not (0.5 <= float(np.mean(mag)) <= 1.6):
        warnings.append(f"Mean acceleration magnitude is {np.mean(mag):.2f} g after unit conversion "
                        "(expected ≈1 g for total acceleration). Check units / gravity removal.")
    if mapping.label and all(sg.label is None for sg in segs):
        warnings.append("Label column selected but no label maps to a supported activity: no ground-truth "
                        "evaluation is possible (edit the label mapping).")

    stats = {
        "original_rows": n_rows, "original_acc_samples": n_acc_samples,
        "duplicates_removed": dup_removed, "missing_values": missing,
        "original_fs_hz": round(orig_fs, 3), "resampled_samples": int(len(signal)),
        "resampled_fs_hz": target_fs, "n_windows": int(ws.n), "window": window, "stride": stride,
        "n_segments": len(segs), "n_subjects": len({sg.subject for sg in segs}),
        "channels": channels, "acc_unit": acc_unit, "gyro_unit": gyro_unit, "time": time_note,
    }
    return ProcessedFile(name=name, source=source, mapping=mapping, channels=channels, signal=signal,
                         sample_segment=np.concatenate(sample_seg), segments=segs, windows=ws,
                         stats=stats, warnings=warnings, label_map=lmap, fs=target_fs)
