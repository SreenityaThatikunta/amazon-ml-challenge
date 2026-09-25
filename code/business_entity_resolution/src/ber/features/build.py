"""Pair + context features for candidate pairs (tasks T3.1–T3.3).

Input: candidate pairs joined with both records' normalised fields, columns
prefixed ``q_`` (the S2/S3 query) and ``s_`` (the S1 candidate). String
similarities use ``rapidfuzz.process.cpdist`` (vectorised, multithreaded).

Groups (see eda/FINDINGS.md):
  * name: exact/compact equality, edit distances split into substitutions vs
    insertions/deletions vs transpositions (distractors change letters,
    true-pair noise mostly drops/doubles/swaps them), fuzzy ratios, alias parts,
    initials, legal-form agreement/conflict, gibberish (vocabulary OOV share),
    name ambiguity (how many S1 share the core name)
  * address: equality, fuzzy ratios, region agreement, house-number relation
    (equal/padded/truncated vs shifted), number-set overlap, PO box/PMB
  * retrieval: TF-IDF cosines and ranks from stage 1
  * context: each score relative to the query's other candidates (margin over
    the best other candidate, rank, gap to best), candidate counts per query
    and per S1
No country feature: every feature means the same thing in any country.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Indel, JaroWinkler, Levenshtein, OSA

from .numbers import number_features

CONTEXT_COLS = ["cos_combo", "cos_name", "cos_addr", "n_tsr", "n_jw", "a_tsr"]


def _cp(a: pd.Series, b: pd.Series, scorer) -> np.ndarray:
    return process.cpdist(a.tolist(), b.tolist(), scorer=scorer, workers=-1)


def initials(core: pd.Series) -> pd.Series:
    return core.map(lambda s: "".join(t[0] for t in s.split()) if s else "")


def name_features(d: pd.DataFrame) -> pd.DataFrame:
    qc, sc = d.q_name_core, d.s_name_core
    qx, sx = d.q_name_compact, d.s_name_compact
    f = pd.DataFrame(index=d.index)
    f["n_core_eq"] = (qc == sc).astype(np.int8)
    f["n_compact_eq"] = (qx == sx).astype(np.int8)
    lev, osa, indel = _cp(qx, sx, Levenshtein.distance), _cp(qx, sx, OSA.distance), \
        _cp(qx, sx, Indel.distance)
    f["n_lev"] = lev
    f["n_sub"] = indel - lev          # substitutions on the optimal path
    f["n_transp"] = lev - osa         # adjacent swaps
    f["n_len_q"], f["n_len_s"] = qx.str.len(), sx.str.len()
    f["n_len_diff"] = f.n_len_q - f.n_len_s
    f["n_ratio"] = _cp(qc, sc, fuzz.ratio)
    f["n_tsr"] = _cp(qc, sc, fuzz.token_set_ratio)
    f["n_tsort"] = _cp(qc, sc, fuzz.token_sort_ratio)
    f["n_partial"] = _cp(qc, sc, fuzz.partial_ratio)
    f["n_jw"] = _cp(qx, sx, JaroWinkler.similarity)
    alias = d.q_name_has_alias.to_numpy()
    alt = np.maximum(_cp(d.q_name_alt_core, sc, fuzz.ratio), _cp(d.q_name_pre_core, sc, fuzz.ratio))
    f["n_alias_best"] = np.where(alias, alt, -1)
    f["n_first_eq"] = (qc.str.split().str[0] == sc.str.split().str[0]).astype(np.int8)
    f["n_last_eq"] = (qc.str.split().str[-1] == sc.str.split().str[-1]).astype(np.int8)
    f["n_q_is_initials"] = ((qx == d.s_initials) & (qx.str.len() >= 2)).astype(np.int8)
    f["n_s_is_initials"] = ((sx == d.q_initials) & (sx.str.len() >= 2)).astype(np.int8)
    ql, sl = d.q_name_legal, d.s_name_legal
    f["l_eq"] = ((ql == sl) & (ql != "")).astype(np.int8)
    f["l_conflict"] = [int(bool(a) and bool(b) and not set(a.split()) & set(b.split()))
                       for a, b in zip(ql, sl)]
    f["l_q_empty"], f["l_s_empty"] = (ql == "").astype(np.int8), (sl == "").astype(np.int8)
    f["q_is_domain"] = d.q_name_is_domain.astype(np.int8)
    f["q_has_alias"] = d.q_name_has_alias.astype(np.int8)
    f["q_indic"] = d.q_name_indic.astype(np.int8)
    f["q_indic_oov"] = d.q_name_n_indic_oov
    f["q_vocab_oov"] = d.q_vocab_oov
    f["s_name_freq"] = d.s_name_freq
    return f


def address_features(d: pd.DataFrame) -> pd.DataFrame:
    qa, sa = d.q_addr_norm, d.s_addr_norm
    f = pd.DataFrame(index=d.index)
    f["a_q_empty"], f["a_s_empty"] = (qa == "").astype(np.int8), (sa == "").astype(np.int8)
    f["a_bag_eq"] = ((d.q_addr_bag == d.s_addr_bag) & (qa != "")).astype(np.int8)
    f["a_tsr"] = _cp(qa, sa, fuzz.token_set_ratio)
    f["a_ratio"] = _cp(d.q_addr_bag, d.s_addr_bag, fuzz.ratio)
    f["a_partial"] = _cp(qa, sa, fuzz.partial_ratio)
    qr, sr = d.q_region, d.s_region
    f["r_eq"] = ((qr == sr) & (qr != "")).astype(np.int8)
    f["r_conflict"] = ((qr != sr) & (qr != "") & (sr != "")).astype(np.int8)
    f["r_q_missing"] = (qr == "").astype(np.int8)
    nums = [number_features(a, b) for a, b in zip(d.q_addr_numbers, d.s_addr_numbers)]
    nf = pd.DataFrame(nums, index=d.index,
                      columns=["hn_rel", "hn_logdiff", "num_common", "num_q_only",
                               "num_s_only", "num_jacc"])
    f = pd.concat([f, nf], axis=1)
    f["hn_rel"] = f.hn_rel.astype(np.int8)
    f["box_q"], f["box_s"] = d.q_addr_has_box.astype(np.int8), d.s_addr_has_box.astype(np.int8)
    return f


def _best_other(q: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For rows grouped by query id ``q``: best value in the group, best value
    among the *other* rows of the group, and rank (0 = best) of each row."""
    order = np.lexsort((-v, q))
    qs, vs = q[order], v[order]
    starts = np.r_[0, np.flatnonzero(qs[1:] != qs[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(qs)])
    first = np.repeat(vs[starts], counts)
    second_idx = np.minimum(starts + 1, len(vs) - 1)
    second = np.where(counts > 1, vs[second_idx], np.nan)
    second = np.repeat(second, counts)
    rank = np.arange(len(qs)) - np.repeat(starts, counts)
    other = np.where(rank == 0, second, first)
    out_best, out_other, out_rank = (np.empty_like(first), np.empty_like(other),
                                     np.empty(len(v), dtype=np.int16))
    out_best[order], out_other[order], out_rank[order] = first, other, rank
    return out_best, out_other, out_rank


def context_features(d: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    """Scores relative to the same query's other candidates. ``d`` must be sorted
    or at least contain every candidate of each query it holds."""
    q = pd.factorize(d.q_id)[0]
    out = pd.DataFrame(index=d.index)
    for col in CONTEXT_COLS:
        v = (d[col] if col in d else f[col]).to_numpy(dtype=np.float64)
        best, other, rank = _best_other(q, v)
        out[f"{col}_margin"] = v - np.nan_to_num(other, nan=0.0)   # >0: beats all others
        out[f"{col}_gap"] = best - v
        out[f"{col}_rank"] = rank
    out["q_n_cands"] = np.bincount(q)[q]
    return out


def build_features(d: pd.DataFrame) -> pd.DataFrame:
    """All features for a frame of candidate pairs with q_/s_ record fields,
    stage-1 retrieval columns, and per-S1 context (s_n_cands, s_n_top1)."""
    retr = d[["cos_name", "cos_addr", "cos_combo", "rank_name", "rank_addr", "rank_combo",
              "s_n_cands", "s_n_top1"]].copy()
    retr["q_source"] = d.q_source.astype(np.int8)
    nf = name_features(d)
    af = address_features(d)
    f = pd.concat([retr, nf, af], axis=1)
    return pd.concat([f, context_features(d, f)], axis=1)
