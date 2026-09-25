"""Address-number relation features (task T3.1).

EDA (eda/FINDINGS.md §3): true pairs keep the house number, or **truncate** it
(7622 -> 762, 6665 -> 665), **zero-pad** it (2877 -> 002877) or drop it.
Lookalike distractors **shift** it by a small amount (805 -> 812, 5835 -> 5838).
"""

from __future__ import annotations

import math

# house-number relation codes (categorical feature)
MISSING, EQUAL, PADDED, TRUNCATED, SHIFTED, DIFFERENT = range(6)
RELATION_NAMES = ["missing", "equal", "padded", "truncated", "shifted", "different"]
SHIFT_MAX = 50  # |a - b| at or below this (same digit count) counts as a small shift


def relation(a: str, b: str) -> int:
    """Relation between two raw digit strings (e.g. house numbers)."""
    if not a or not b:
        return MISSING
    if a == b:
        return EQUAL
    sa, sb = a.lstrip("0") or "0", b.lstrip("0") or "0"
    if sa == sb:
        return PADDED
    if len(sa) != len(sb) and (sa.startswith(sb) or sb.startswith(sa) or
                               sa.endswith(sb) or sb.endswith(sa)):
        return TRUNCATED
    if len(sa) == len(sb) and abs(int(sa) - int(sb)) <= SHIFT_MAX:
        return SHIFTED
    return DIFFERENT


def number_features(q_nums: str, s_nums: str) -> tuple[int, float, int, int, int, float]:
    """Features for space-separated digit strings of the query and S1 address:
    (house-number relation, log1p |difference|, n common, n query-only,
    n S1-only, jaccard of the number sets)."""
    qn, sn = q_nums.split(), s_nums.split()
    rel = relation(qn[0] if qn else "", sn[0] if sn else "")
    diff = (math.log1p(abs(int(qn[0]) - int(sn[0]))) if qn and sn and
            len(qn[0]) < 12 and len(sn[0]) < 12 else -1.0)
    qs = {x.lstrip("0") or "0" for x in qn}
    ss = {x.lstrip("0") or "0" for x in sn}
    common = len(qs & ss)
    union = len(qs | ss)
    return rel, diff, common, len(qs - ss), len(ss - qs), (common / union if union else -1.0)
