"""``ber features`` (task T3.3): features for every candidate pair.

Train split: all val-side queries plus a ``train_query_frac`` sample of
train-side queries (the model needs far fewer pairs than exist), labelled
``y = (s1_id == owner)``. Test split: every query.

Outputs ``features_<split>/`` (parquet parts) with q_id, s1_id, [y, side] and
the feature columns of :mod:`ber.features.build`.
"""

from __future__ import annotations

import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from ..artifacts import StageRun
from ..config import Config
from ..eval.split import stable_unit
from ..log import timed
from .build import addr_words, build_features, house_number, initials, name_city

REC_COLS = ["entity_id", "source", "country", "name_core", "name_compact", "name_legal",
            "name_pre_core", "name_alt_core", "name_is_domain", "name_has_alias", "name_indic",
            "name_n_indic_oov", "addr_norm", "addr_bag", "region", "addr_numbers",
            "addr_has_box"]


def record_extras(recs: pd.DataFrame) -> pd.DataFrame:
    """Per-record extras: initials, address words / house number / city words in
    the name (lookalike features), share of name tokens unknown to the S1 name
    vocabulary of the country (gibberish brand names score high), and for S1 how
    many S1 of the country share the exact core name (ambiguity)."""
    recs = recs.copy()
    recs["initials"] = initials(recs.name_core)
    recs["addr_words"] = [addr_words(a) for a in recs.addr_norm]
    recs["house"] = [house_number(x) for x in recs.addr_numbers]
    recs["name_city"] = [name_city(n, a) for n, a in zip(recs.name_core, recs.addr_norm)]
    s1 = recs[recs.source == 1]
    vocab = {c: Counter(t for n in g.name_core for t in set(n.split()))
             for c, g in s1.groupby("country")}
    recs["vocab_oov"] = [
        (sum(vocab.get(c, {}).get(t, 0) == 0 for t in n.split()) / len(n.split())) if n else 1.0
        for n, c in zip(recs.name_core, recs.country)]
    freq = s1.groupby(["country", "name_core"]).size().rename("name_freq")
    recs = recs.join(freq, on=["country", "name_core"])
    recs["name_freq"] = recs.name_freq.fillna(0).astype(np.int32)
    return recs


def s1_context(cand_dir) -> pd.DataFrame:
    """Per S1: number of queries listing it (s_n_cands) and number of queries
    whose best combo candidate it is (s_n_top1), over the full candidate set."""
    n_c, n_t = Counter(), Counter()
    for f in sorted(cand_dir.glob("*.parquet")):
        p = pd.read_parquet(f, columns=["q_id", "s1_id", "cos_combo"])
        p["s1_id"] = p.s1_id.astype(str)
        n_c.update(p.s1_id.value_counts().to_dict())
        best = p.loc[p.groupby("q_id", observed=True).cos_combo.idxmax(), "s1_id"]
        n_t.update(best.value_counts().to_dict())
    idx = list(n_c)
    return pd.DataFrame({"s_n_cands": [n_c[i] for i in idx],
                         "s_n_top1": [n_t.get(i, 0) for i in idx]}, index=idx)


def _query_chunks(part: pd.DataFrame, max_rows: int):
    """Split a candidate part (rows grouped by query) at query boundaries."""
    q = part.q_id.to_numpy()
    cuts = np.r_[0, np.flatnonzero(q[1:] != q[:-1]) + 1, len(q)]
    start = 0
    for i in range(1, len(cuts)):
        if cuts[i] - start >= max_rows or i == len(cuts) - 1:
            yield part.iloc[start:cuts[i]]
            start = cuts[i]


def join_records(chunk: pd.DataFrame, rec: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    """Attach q_/s_ record fields and per-S1 context to candidate pairs."""
    qf = rec.reindex(chunk.q_id.to_numpy()).add_prefix("q_").reset_index(drop=True)
    sf = rec.reindex(chunk.s1_id.to_numpy()).add_prefix("s_").reset_index(drop=True)
    c = ctx.reindex(chunk.s1_id.to_numpy()).fillna(0).reset_index(drop=True)
    return pd.concat([chunk.reset_index(drop=True), qf, sf, c], axis=1)


def compact(df: pd.DataFrame) -> pd.DataFrame:
    """float64 -> float32 and int64/uint32 -> int32 before writing (the models cast
    every feature to float32 anyway); with zstd this halves the features on disk
    (29 GB -> ~13 GB at full scale)."""
    for c in df.columns:
        t = df[c].dtype
        if t == np.float64:
            df[c] = df[c].astype(np.float32)
        elif t in (np.int64, np.uint32):
            df[c] = df[c].astype(np.int32)
    return df


def _sub_chunks(chunk: pd.DataFrame, n: int) -> list[pd.DataFrame]:
    """Split a chunk into ~n pieces at query boundaries (context features need
    every candidate of a query in the same piece)."""
    return list(_query_chunks(chunk, max(1, -(-len(chunk) // n))))


def run_features(cfg: Config) -> None:
    fc = cfg.features
    workers = fc.workers or max(1, min(4, (os.cpu_count() or 4) - 2))
    pool = ProcessPoolExecutor(workers) if workers > 1 else None
    try:
        _run_features(cfg, pool, workers)
    finally:
        if pool is not None:
            pool.shutdown()


def _features(pool, d: pd.DataFrame, workers: int) -> pd.DataFrame:
    """build_features, split over worker processes (the pure-Python lookalike
    features dominate the stage's time)."""
    if pool is None or len(d) < 50_000:
        return build_features(d)
    pieces = _sub_chunks(d, workers)
    return pd.concat(list(pool.map(build_features, pieces)))


def _run_features(cfg: Config, pool, workers: int) -> None:
    fc = cfg.features
    with StageRun(cfg, "features") as st:
        st.metric("workers", workers)
        splits = ["train"] + (["test"] if cfg.run_test else [])
        for split in splits:
            extra = ["side", "owner"] if split == "train" else []
            with timed(st.log, f"{split}: load records + extras"):
                recs = pd.read_parquet(st.input_path("normalize", f"records_{split}"),
                                       columns=REC_COLS + extra)
                recs = record_extras(recs)
                if split == "train":
                    q = recs[recs.source != 1]
                    u = stable_unit(q.entity_id, cfg.seed + 1)
                    keep = (q.side == "val").to_numpy() | (u < fc.train_query_frac)
                    selected = set(q.entity_id[keep])
                    st.metric("train_queries_selected", {"train": int((keep & (q.side == "train")
                                                                       ).sum()),
                                                         "val": int((q.side == "val").sum())})
                rec = recs.set_index("entity_id")
            cand_dir = st.input_path("candidates", f"candidates_{split}")
            with timed(st.log, f"{split}: per-S1 candidate context"):
                ctx = s1_context(cand_dir)
            out_dir = st.path(f"features_{split}")
            out_dir.mkdir(parents=True, exist_ok=True)
            for old in out_dir.glob("*.parquet"):
                old.unlink()
            n_parts = n_rows = n_pos = 0
            with timed(st.log, f"{split}: features"):
                for f in sorted(cand_dir.glob("*.parquet")):
                    part = pd.read_parquet(f).astype({"q_id": str, "s1_id": str})
                    if split == "train":
                        part = part[part.q_id.isin(selected)]
                    if part.empty:
                        continue
                    for chunk in _query_chunks(part, fc.chunk_rows):
                        d = join_records(chunk, rec, ctx)
                        feats = _features(pool, d, workers)
                        keys = d[["q_id", "s1_id"]].copy()
                        if split == "train":
                            keys["y"] = (d.s1_id == d.q_owner).astype(np.int8)
                            keys["side"] = d.q_side.to_numpy()
                            n_pos += int(keys.y.sum())
                        out = compact(pd.concat([keys, feats], axis=1))
                        out.to_parquet(out_dir / f"part-{n_parts:05d}.parquet", index=False,
                                       compression="zstd")
                        n_parts += 1
                        n_rows += len(out)
                    st.log.debug("features %s: %d rows so far", f.name, n_rows)
            st.register(f"features_{split}", out_dir, rows=n_rows, parts=n_parts)
            st.metric(f"{split}_pairs", n_rows)
            if split == "train":
                st.metric("train_positive_rate", round(n_pos / max(n_rows, 1), 4))
