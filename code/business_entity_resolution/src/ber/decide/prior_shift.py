"""Prior-shift correction of pair probabilities per lookalike segment (v2.2).

Why: test has about twice as many lookalike distractors per S1 as train/val, mostly
"same name, same street, house number shifted by a few units" and legal-form swaps.
Among such pairs the share of true matches is
therefore lower on test than on val, so probabilities calibrated on val are too
high there. Within a segment we assume p(features | match) and p(features | no
match) are unchanged and only the class balance moved (label shift), the setting
of Saerens, Latinne & Decaestecker (2002): the test prior of the segment is
estimated by EM from the model's own test probabilities, and every probability of
the segment is rescaled by the prior odds ratio

    p' = r p / (r p + 1 - p),   r = odds(pi_test) / odds(pi_val).

Segments: house-number relation x legal-form conflict x near-copy (exact name key
and address token-set ratio >= 90), per country; pi_val is the mean val p2 of the
segment (pooled over val countries for a country absent from val, e.g. France).
Needs the segment columns that ``ber rerank`` copies into its score files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEG_COLS = ("hn_rel", "l_conflict", "k_eq", "a_tsr")


def segment_key(df: pd.DataFrame) -> pd.Series:
    near = ((df.k_eq == 1) & (df.a_tsr >= 90)).astype(int)
    return (df.hn_rel.astype(int).astype(str) + "|" + df.l_conflict.astype(int).astype(str)
            + "|" + near.astype(str))


def segment_priors(scores: pd.DataFrame) -> dict[str, list[float]]:
    """{segment: [mean p, n]} over pairs (val: out-of-fold p2)."""
    g = scores.assign(seg=segment_key(scores)).groupby("seg").p.agg(["mean", "size"])
    return {k: [float(r["mean"]), int(r["size"])] for k, r in g.iterrows()}


def _odds(x: float) -> float:
    x = min(max(x, 1e-6), 1 - 1e-6)
    return x / (1 - x)


def em_ratio(p: np.ndarray, pi_val: float, iters: int = 200, tol: float = 1e-7) -> tuple[float, float]:
    """EM estimate of the test prior of a segment from calibrated-on-val
    probabilities p; returns (odds ratio r, estimated test prior)."""
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    r, pi = 1.0, float(p.mean())
    for _ in range(iters):
        q = r * p / (r * p + 1 - p)
        pi_new = float(q.mean())
        r_new = _odds(pi_new) / _odds(pi_val)
        if abs(r_new - r) < tol:
            r, pi = r_new, pi_new
            break
        r, pi = r_new, pi_new
    return r, pi


def adjust(scores: pd.DataFrame, priors: dict[str, list[float]], min_pairs: int = 2000,
           r_bounds: tuple[float, float] = (0.2, 5.0)) -> tuple[pd.DataFrame, dict]:
    """Rescale ``scores.p`` segment by segment (one country's pairs). Returns the
    adjusted frame and a report {segment: {n, pi_val, pi_test, r}}."""
    s = scores.copy()
    seg = segment_key(s).to_numpy()
    p = s.p.to_numpy(dtype=np.float64)
    out = p.copy()
    report = {}
    for k in np.unique(seg):
        m = seg == k
        if k not in priors or m.sum() < min_pairs:
            continue
        pi_val = priors[k][0]
        r, pi_te = em_ratio(p[m], pi_val)
        r = float(np.clip(r, *r_bounds))
        out[m] = r * p[m] / (r * p[m] + 1 - p[m])
        report[k] = {"n": int(m.sum()), "pi_val": round(pi_val, 5), "pi_test": round(pi_te, 5),
                     "r": round(r, 4)}
    s["p"] = out.astype(np.float32)
    return s, report
