"""Automatic detection of sensor / time / subject / label columns and time parsing."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

Triple = Tuple[str, str, str]


@dataclass
class ColumnMapping:
    time: Optional[str] = None
    subject: Optional[str] = None
    label: Optional[str] = None
    acc: Optional[Triple] = None
    gyro: Optional[Triple] = None
    gravity: Optional[Triple] = None          # added to a gravity-free acc triple
    sensor_type_col: Optional[str] = None     # long format (Format D)
    acc_sensor_value: Optional[str] = None
    gyro_sensor_value: Optional[str] = None
    time_unit: str = "auto"                   # auto | s | ms | us | ns | datetime
    fs: Optional[float] = None                # manual sampling frequency (Hz)
    acc_unit: str = "auto"                    # auto | g | m/s2 | mg
    gyro_unit: str = "auto"                   # auto | rad/s | deg/s

    @classmethod
    def from_dict(cls, d: dict) -> "ColumnMapping":
        d = dict(d)
        for k in ("acc", "gyro", "gravity"):
            if d.get(k) is not None:
                d[k] = tuple(d[k])
        return cls(**d)


@dataclass
class Detection:
    mapping: ColumnMapping
    triples: Dict[str, Triple] = field(default_factory=dict)
    kinds: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def to_numeric_array(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(s):
        return s.to_numpy(dtype=float)
    cleaned = s.astype(str).str.strip().str.rstrip(";").str.strip()
    return pd.to_numeric(cleaned, errors="coerce").to_numpy(dtype=float)


def _numeric_fraction(s: pd.Series) -> float:
    return float(np.isfinite(to_numeric_array(s.head(2000))).mean()) if len(s) else 0.0


_AXIS_RE = re.compile(r"^(?P<p>.*?)[_\-\.\s]*(?P<a>[xyz])$", re.I)
_TIME_EXACT = ["timestamp", "time", "datetime", "date_time", "creation_time", "ts", "t",
               "time_s", "time_ms", "arrival_time", "elapsed_time", "date"]
_SUBJECT = r"^(subject|subject_?id|subject_?number|user|user_?id|participant|participant_?id|person|volunteer|uid|subj)$"
_LABEL = r"^(activity|activity_?label|activity_?name|activity_?id|label|labels|gt|class|target|action|annotation|act)$"


def _find_time(cols, lower) -> Optional[str]:
    for name in _TIME_EXACT:
        for c in cols:
            if lower[c] == name:
                return c
    for c in cols:
        if "time" in lower[c] or "stamp" in lower[c]:
            return c
    return None


def _find(cols, lower, pattern, exclude=()) -> Optional[str]:
    for c in cols:
        if c in exclude:
            continue
        if re.match(pattern, lower[c]):
            return c
    return None


def _kind(prefix: str) -> str:
    p = prefix.lower()
    if any(w in p for w in ("gyro", "rotation", "angular")):
        return "gyro"
    if "grav" in p:
        return "gravity"
    if "mag" in p and "magnitude" not in p:
        return "mag"
    if "acc" in p:
        return "acc"
    if p in ("", "axis", "value", "val", "data", "sensor", "raw", "values"):
        return "generic"
    return "other"


def _acc_rank(prefix: str) -> int:
    p = prefix.lower()
    if "total" in p or "raw" in p:
        return 0
    if any(w in p for w in ("body", "user", "linear")):
        return 2    # gravity-free
    return 1


def detect_mapping(df: pd.DataFrame, source: str = "Custom") -> Detection:
    cols = list(df.columns)
    lower = {c: str(c).strip().lower() for c in cols}
    notes: List[str] = []
    m = ColumnMapping()
    m.time = _find_time(cols, lower)
    m.subject = _find(cols, lower, _SUBJECT, exclude={m.time})
    m.label = _find(cols, lower, _LABEL, exclude={m.time, m.subject})

    groups: Dict[str, Dict[str, str]] = {}
    for c in cols:
        if c in (m.time, m.subject, m.label):
            continue
        mt = _AXIS_RE.match(str(c).strip())
        if mt and _numeric_fraction(df[c]) >= 0.8:
            groups.setdefault(mt.group("p"), {})[mt.group("a").lower()] = c
    triples: Dict[str, Triple] = {}
    kinds: Dict[str, str] = {}
    for p, ax in groups.items():
        if set(ax) == {"x", "y", "z"}:
            key = p if p else "x/y/z"
            triples[key] = (ax["x"], ax["y"], ax["z"])
            kinds[key] = _kind(p)

    # long format: a column naming the sensor
    for c in cols:
        if c in (m.time, m.subject, m.label) or df[c].dtype.kind in "if":
            continue
        if re.search(r"sensor|stream|type", lower[c]) and df[c].nunique() <= 20:
            vals = [str(v) for v in df[c].dropna().unique()]
            acc_v = next((v for v in vals if "acc" in v.lower()), None)
            gyr_v = next((v for v in vals if "gyr" in v.lower()), None)
            if acc_v or gyr_v:
                m.sensor_type_col, m.acc_sensor_value, m.gyro_sensor_value = c, acc_v, gyr_v
                break

    accs = [k for k, v in kinds.items() if v == "acc"]
    gyros = [k for k, v in kinds.items() if v == "gyro"]
    gravs = [k for k, v in kinds.items() if v == "gravity"]
    generics = [k for k, v in kinds.items() if v == "generic"]

    if m.sensor_type_col:
        pool = generics or accs
        if pool:
            m.acc = triples[pool[0]]
            m.gyro = triples[pool[0]] if m.gyro_sensor_value else None
            notes.append(f"Long format detected: sensor column '{m.sensor_type_col}'.")
    else:
        if accs:
            best = min(accs, key=_acc_rank)
            m.acc = triples[best]
            if _acc_rank(best) == 2:
                if gravs:
                    m.gravity = triples[gravs[0]]
                    notes.append("Selected acceleration looks gravity-free; matching gravity columns will be added.")
                else:
                    notes.append("Selected acceleration looks gravity-free (body/user acc) and no gravity "
                                 "columns were found: models trained on total acceleration may misbehave.")
        elif generics:
            m.acc = triples[generics[0]]
        if gyros:
            m.gyro = triples[gyros[0]]
    if m.acc is None:
        notes.append("Could not detect accelerometer X/Y/Z columns automatically; please map them manually.")
    if m.time is None:
        notes.append("No time column found; a sampling frequency must be provided.")
    return Detection(mapping=m, triples=triples, kinds=kinds, notes=notes)


def parse_time_seconds(series: pd.Series, unit: str = "auto") -> Tuple[np.ndarray, str, bool]:
    """Return (seconds, unit_used, looks_like_index). Times are in file order (not sorted)."""
    num = to_numeric_array(series)
    is_numeric_col = np.isfinite(num).mean() > 0.9
    if unit == "datetime" or not is_numeric_col:
        ts = pd.to_datetime(series, errors="coerce", utc=True)
        sec = (ts - ts.min()).dt.total_seconds().to_numpy(dtype=float)
        return sec, "datetime", False
    scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
    d = np.diff(num[np.isfinite(num)])
    d = d[d > 0]
    if unit != "auto":
        return num * scale[unit], unit, False
    if len(d) == 0:
        raise ValueError("Cannot infer time unit (no increasing timestamps).")
    med = float(np.median(d))
    finite = num[np.isfinite(num)]
    if med == 1.0 and np.all(finite == np.round(finite)):
        return num, "index", True
    for u in ("s", "ms", "us", "ns"):
        fs = 1.0 / (med * scale[u])
        if 1.0 <= fs <= 1000.0:
            return num * scale[u], u, False
    raise ValueError("Could not infer the timestamp unit; please select it manually.")
