"""Robust CSV reading (separator sniffing, header-less files, WISDM-style trailing ';')."""
from __future__ import annotations

import io
from typing import Optional

import pandas as pd


def _sniff_sep(text: str) -> str:
    first = text.splitlines()[0] if text else ""
    counts = {s: first.count(s) for s in [",", ";", "\t", "|"]}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def _is_num(tok: str) -> bool:
    try:
        float(tok.strip().rstrip(";"))
        return True
    except ValueError:
        return False


def read_csv_bytes(data: bytes, source: str = "Custom", header: Optional[bool] = None) -> pd.DataFrame:
    """Read CSV bytes. header=None -> auto-detect (a first row that is >=50% numeric is data)."""
    text = data[:20000].decode("utf-8", errors="replace")
    sep = _sniff_sep(text)
    first = text.splitlines()[0] if text else ""
    toks = first.split(sep)
    has_header = header if header is not None else (sum(_is_num(t) for t in toks) / max(len(toks), 1) < 0.5)
    df = pd.read_csv(io.BytesIO(data), sep=sep, header=0 if has_header else None,
                     on_bad_lines="skip", low_memory=False)
    if not has_header:
        if source == "WISDM" and df.shape[1] == 6:
            df.columns = ["user", "activity", "timestamp", "x", "y", "z"]
        else:
            df.columns = [f"col_{i}" for i in range(df.shape[1])]
    df.columns = [str(c).strip() for c in df.columns]
    return df
