"""``ber features`` (task T3.3): features for every candidate pair.

Train split: all val-side queries plus a ``train_query_frac`` sample of
train-side queries (the model needs far fewer pairs than exist), labelled
``y = (s1_id == owner)``. Test split: every query.

Outputs ``features_<split>/`` (parquet parts) with q_id, s1_id, [y, side] and
the feature columns of :mod:`ber.features.build`.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from ..artifacts import StageError, StageRun, load_manifest
from ..config import Config
from ..eval.split import stable_unit
from ..log import timed
from .build import KeyIndex, build_features, initials

REC_COLS = ["entity_id", "source", "country", "name_core", "name_compact", "name_legal",
            "name_pre_core", "name_alt_core", "name_is_domain", "name_has_alias", "name_indic",
            "name_n_indic_oov", "addr_norm", "addr_bag", "region", "addr_numbers",
            "addr_has_box"]


def learn_fillers(frames: list[pd.DataFrame], min_rate: float, min_ratio: float) -> list[str]:
    """Filler words the noise appends to S2/S3 names ("Riverside", "Services",
    "Participations", "Smt" ...), learned without labels: tokens whose share of
    S2/S3 core names (per country and split) is >= ``min_ratio`` times their share
    of S1 core names, and >= ``min_rate``. A copy of an S1 name keeps its tokens at
    the S1 rate, so only tokens the noise injects stand out. The union over every
    (split, country) is used everywhere, so train and test features agree."""
    out: set[str] = set()
    for df in frames:
        for _, g in df.groupby("country"):
            s1, q = g.name_core[g.source == 1], g.name_core[g.source != 1]
            if len(s1) == 0 or len(q) == 0:
                continue
            cs = Counter(t for n in s1 for t in set(n.split()))
            cq = Counter(t for n in q for t in set(n.split()))
            for t, c in cq.items():
                rate = c / len(q)
                if rate >= min_rate and rate >= min_ratio * (cs[t] + 1) / len(s1):
                    out.add(t)
    return sorted(out)


def name_key(core: pd.Series, fillers: frozenset[str]) -> pd.Series:
    """Sorted unique core tokens without filler words (order- and filler-free
    form of the name); the plain sorted tokens if nothing else is left."""
    def key(n: str) -> str:
        toks = sorted(set(n.split()))
        kept = [t for t in toks if t not in fillers]
        return " ".join(kept or toks)
    return core.map(key)


def token_idf(keys: pd.Series) -> dict[str, float]:
    """Smoothed IDF of name-key tokens over S1 records."""
    df = Counter(t for n in keys for t in n.split())
    n = len(keys)
    return {t: float(np.log((n + 1) / (c + 1)) + 1.0) for t, c in df.items()}


def record_extras(recs: pd.DataFrame, fillers: list[str] | None = None) -> pd.DataFrame:
    """Per-record extras: initials, share of name tokens unknown to the S1 name
    vocabulary of the country (gibberish brand names score high), for S1 how
    many S1 of the country share the exact core name (ambiguity), and (with
    ``fillers``) the filler-free name key."""
    recs = recs.copy()
    recs["initials"] = initials(recs.name_core)
    if fillers is not None:
        recs["name_key"] = name_key(recs.name_core, frozenset(fillers))
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


def _has_output(st: StageRun, name: str) -> bool:
    try:
        return name in load_manifest(st.cfg, "normalize")["outputs"]
    except StageError:
        return False


def run_features(cfg: Config) -> None:
    fc = cfg.features
    with StageRun(cfg, "features") as st:
        splits = (["train"] if cfg.run_train else []) + (["test"] if cfg.run_test else [])
        fillers = None
        if fc.name_key:
            with timed(st.log, "learn filler words"):
                # both splits whenever test records exist (also in a train-only pass),
                # so the train and test passes learn the same list
                frames = [pd.read_parquet(st.input_path("normalize", f"records_{sp}"),
                                          columns=["source", "country", "name_core"])
                          for sp in ("train", "test") if _has_output(st, f"records_{sp}")]
                fillers = learn_fillers(frames, fc.filler_min_rate, fc.filler_min_ratio)
                del frames
            st.write_json(fillers, "fillers")
            st.metric("fillers", " ".join(fillers))
        for split in splits:
            extra = ["side", "owner"] if split == "train" else []
            with timed(st.log, f"{split}: load records + extras"):
                recs = pd.read_parquet(st.input_path("normalize", f"records_{split}"),
                                       columns=REC_COLS + extra)
                recs = record_extras(recs, fillers)
                kidx = None
                if fillers is not None:
                    kidx = KeyIndex(recs.name_key, token_idf(recs.name_key[recs.source == 1]))
                    recs["kpos"] = np.arange(len(recs), dtype=np.int32)
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
                        feats = build_features(d, kidx)
                        keys = d[["q_id", "s1_id"]].copy()
                        if split == "train":
                            keys["y"] = (d.s1_id == d.q_owner).astype(np.int8)
                            keys["side"] = d.q_side.to_numpy()
                            n_pos += int(keys.y.sum())
                        out = pd.concat([keys, feats], axis=1)
                        out.to_parquet(out_dir / f"part-{n_parts:05d}.parquet", index=False)
                        n_parts += 1
                        n_rows += len(out)
                    st.log.debug("features %s: %d rows so far", f.name, n_rows)
            st.register(f"features_{split}", out_dir, rows=n_rows, parts=n_parts)
            st.metric(f"{split}_pairs", n_rows)
            if split == "train":
                st.metric("train_positive_rate", round(n_pos / max(n_rows, 1), 4))
