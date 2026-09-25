"""Per-S1 expected-F0.5 subset selection (task T4.1).

For one S1 with candidate queries of match probability p_1 >= p_2 >= ... the
prediction is always a prefix (top-k). For k > 0 the expected F0.5 is
approximated by the ratio of expectations

    E[F0.5 | top-k]  ~  1.25 * sum(p_1..p_k) / (0.25 * sum(p_all) + k)

(sum(p_all) = expected number of true matches among the candidates), and for
k = 0 ("no match") it is the probability that the S1 has no true match at all,
prod(1 - p_i), times ``p_empty_prior`` for matches that no candidate
covers. The k with the highest value wins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BETA2 = 0.25


def best_prefix(p: np.ndarray, empty_prior: float = 1.0) -> tuple[int, float]:
    """(k, expected F0.5) for probabilities sorted in descending order."""
    if len(p) == 0:
        return 0, empty_prior
    csum = np.cumsum(p)
    total = csum[-1]
    k = np.arange(1, len(p) + 1)
    ef = (1 + BETA2) * csum / (BETA2 * total + k)
    e0 = float(np.prod(1.0 - p)) * empty_prior
    i = int(np.argmax(ef))
    return (0, e0) if e0 >= ef[i] else (i + 1, float(ef[i]))


def select(assigned: pd.DataFrame, empty_prior: float = 1.0) -> pd.DataFrame:
    """assigned: one row per query (s1_id, q_id, p), after one-owner assignment.
    Returns the kept rows."""
    a = assigned.sort_values(["s1_id", "p"], ascending=[True, False])
    s1 = a.s1_id.to_numpy()
    p = a.p.to_numpy(dtype=np.float64)
    starts = np.r_[0, np.flatnonzero(s1[1:] != s1[:-1]) + 1, len(s1)]
    keep = np.zeros(len(a), dtype=bool)
    for i in range(len(starts) - 1):
        lo, hi = starts[i], starts[i + 1]
        k, _ = best_prefix(p[lo:hi], empty_prior)
        keep[lo:lo + k] = True
    return a[keep]
