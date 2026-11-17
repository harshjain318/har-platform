"""Activity-label harmonisation across datasets."""
from __future__ import annotations

import re
from typing import Dict, Iterable, Optional

_ALIASES = {
    "walking": {"walking", "walk", "wlk", "walking_normal", "normal_walking"},
    "upstairs": {"walking_upstairs", "upstairs", "ups", "stairsup", "stairs_up", "upstair",
                 "walkingupstairs", "ascending", "climbing_up", "going_up"},
    "downstairs": {"walking_downstairs", "downstairs", "dws", "stairsdown", "stairs_down",
                   "downstair", "walkingdownstairs", "descending", "climbing_down", "going_down"},
    "sitting": {"sitting", "sit"},
    "standing": {"standing", "stand", "std"},
    "running": {"running", "run", "jogging", "jog"},
    "lying": {"lying", "laying", "lay", "lie", "lying_down", "laying_down"},
}
_UCI_NUM = {"1": "walking", "2": "upstairs", "3": "downstairs",
            "4": "sitting", "5": "standing", "6": "lying"}
# WISDM-2019 single-letter codes (C = generic "stairs" cannot be split into up/down -> excluded)
_WISDM_LETTER = {"a": "walking", "b": "running", "d": "sitting", "e": "standing"}


def _norm(v) -> str:
    s = str(v).strip().lower()
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return re.sub(r"[\s\-]+", "_", s)


_LOOKUP = {_norm(a): c for c, al in _ALIASES.items() for a in al}


def canonical_label(value, source: str = "Custom") -> Optional[str]:
    """Return the canonical activity name or None when unsupported/unknown."""
    s = _norm(value)
    if source == "UCI HAR" and s in _UCI_NUM:
        return _UCI_NUM[s]
    if source == "WISDM" and s in _WISDM_LETTER:
        return _WISDM_LETTER[s]
    return _LOOKUP.get(s)


def auto_label_map(values: Iterable, source: str = "Custom") -> Dict[str, Optional[str]]:
    """Map every raw label string to a canonical activity (None = excluded)."""
    out: Dict[str, Optional[str]] = {}
    for v in values:
        raw = str(v).strip()
        out[raw] = canonical_label(raw, source)
    return out
