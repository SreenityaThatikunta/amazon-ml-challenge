"""Copy-group features for stage 2 (plan.md §0.1b N4, experiments E26).

On test, ~60-66% of the no-match records are S2/S3 copies of entities whose S1 is
missing. When a lookalike twin of the missing S1 is present, the whole copy group can
be linked to it. Such a group is closer to *itself* (one source entity, independent
noise) than to the twin, which differs systematically (another number or street).
Real copies of an S1 are each a noisy version of that S1, so they are about as close
to the S1 as to each other.

Per pair (record q, S1 s), over the other records claiming the same s (pairs with
p1 >= ``min_p``, at most ``cap`` of them per S1, highest p1 first):
  cp_cos_qs      cosine(q, s)
  cp_sib_max     max cosine(q, other claimant)
  cp_sib_gap     cp_sib_max - cp_cos_qs  (> 0: q is closer to a sibling than to s)
  cp_sib_n08     number of claimants with cosine(q, .) >= 0.8
  cp_sib_p       p1 of the closest claimant on s
  cp_sib_top_s   1 if s is also the closest claimant's own best S1
Records are binary character-trigram vectors (hashed, l2-normalised, no IDF), so the
values are defined the same way on every data set (no fitting, no density effect)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

COLS = ["cp_cos_qs", "cp_sib_max", "cp_sib_gap", "cp_sib_n08", "cp_sib_p", "cp_sib_top_s"]
_VEC = HashingVectorizer(analyzer="char", ngram_range=(3, 3), n_features=2 ** 20,
                         alternate_sign=False, binary=True, norm="l2", dtype=np.float32)


def vectors(texts) -> sp.csr_matrix:
    return _VEC.transform(texts).tocsr()


def record_texts(rec_path, ids: np.ndarray, filters=None) -> np.ndarray:
    """"core name + normalised address" for ``ids`` ("" if unknown)."""
    r = pd.read_parquet(rec_path, columns=["entity_id", "name_core", "addr_norm"],
                        filters=filters)
    t = pd.Series((r.name_core + " " + r.addr_norm).to_numpy(), index=r.entity_id.to_numpy())
    return t.reindex(ids).fillna("").to_numpy()


def rowdot(A: sp.csr_matrix, ia: np.ndarray, B: sp.csr_matrix, ib: np.ndarray,
           chunk: int = 1_000_000) -> np.ndarray:
    """cosine(A[ia[k]], B[ib[k]]) for every k (rows are l2-normalised)."""
    out = np.empty(len(ia), dtype=np.float32)
    for a in range(0, len(ia), chunk):
        b = a + chunk
        out[a:b] = np.asarray(A[ia[a:b]].multiply(B[ib[a:b]]).sum(axis=1)).ravel()
    return out


def copy_features(q: np.ndarray, s: np.ndarray, p: np.ndarray, Qv: sp.csr_matrix,
                  Sv: sp.csr_matrix, min_p: float = 0.01, cap: int = 20) -> dict[str, np.ndarray]:
    """Features of COLS for pairs (q[i], s[i], p1 p[i]); q / s index rows of Qv / Sv."""
    n = len(q)
    out = {c: np.zeros(n, dtype=np.float32) for c in COLS}
    if n == 0:
        return out
    out["cp_cos_qs"] = rowdot(Qv, q, Sv, s)
    # each record's own best S1 (highest p1)
    o = np.lexsort((-p, q))
    first = np.r_[True, q[o][1:] != q[o][:-1]]
    top_s = pd.Series(s[o][first], index=q[o][first])

    act = np.flatnonzero(p >= min_p)
    if len(act) == 0:
        out["cp_sib_gap"] = -out["cp_cos_qs"]
        return out
    act = act[np.lexsort((-p[act], s[act]))]          # grouped by s, highest p1 first
    sa = s[act]
    starts = np.r_[0, np.flatnonzero(sa[1:] != sa[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(act)])
    g_start = np.repeat(starts, sizes)                 # per active position
    n_anch = np.repeat(np.minimum(sizes, cap), sizes)
    # every active position i paired with each anchor j of its group (j != i)
    i_pos = np.repeat(np.arange(len(act)), n_anch)
    offs = np.arange(len(i_pos)) - np.repeat(np.cumsum(n_anch) - n_anch, n_anch)
    j_pos = np.repeat(g_start, n_anch) + offs
    keep = i_pos != j_pos
    i_pos, j_pos = i_pos[keep], j_pos[keep]
    gap = -out["cp_cos_qs"].copy()
    if len(i_pos):
        d = rowdot(Qv, q[act[i_pos]], Qv, q[act[j_pos]])
        r = np.lexsort((-d, i_pos))                    # best sibling first per i
        best = np.r_[True, i_pos[r][1:] != i_pos[r][:-1]]
        bi, bj, bd = act[i_pos[r][best]], act[j_pos[r][best]], d[r][best]
        out["cp_sib_max"][bi] = bd
        out["cp_sib_p"][bi] = p[bj]
        out["cp_sib_top_s"][bi] = (top_s.reindex(q[bj]).to_numpy() == s[bi]).astype(np.float32)
        out["cp_sib_n08"][:] = np.bincount(act[i_pos], weights=(d >= 0.8), minlength=n)
        gap[bi] = bd - out["cp_cos_qs"][bi]
    out["cp_sib_gap"] = gap
    return out
