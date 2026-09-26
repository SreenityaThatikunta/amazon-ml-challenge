"""``ber normalize`` (task T1.4): learn the Indic dictionary, then normalise records.

Outputs in ``artifacts/<run>/normalize/``:
  * ``translit_dict.parquet``: learned Indic -> Latin dictionary (train-side pairs only),
    used for train/val records so val stays honest
  * ``translit_dict_all.parquet`` (``normalize.test_dict_all_pairs``): the same learned
    from **all** train pairs, used for test records. Whole val regions are held out,
    so the train-side dictionary has no Gurmukhi at all (Punjab is a val region);
    test Punjab records (12.5k Gurmukhi names, 33% with unknown words) need it
  * ``records_train/``: parquet parts, one row per train record (S1, S2, S3) with
    split side/dev/owner, raw fields and normalised fields
  * ``records_test/``: same for test (skipped for the dev slice)

Records are processed one source file at a time, in chunks across worker
processes, and every chunk is written straight to disk, so memory stays flat.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from ..artifacts import StageRun
from ..config import Config
from ..eval.reshape import dropped_s1, orphan_queries
from ..eval.split import slice_mask
from ..io import read_source, write_parquet
from ..log import timed
from .address import AddressNormalizer
from .text import INDIC_RE, NameNormalizer
from .translit import Transliterator, build_vocab, dictionary_coverage, learn_dictionary

NAME_FIELDS = ["norm", "core", "legal", "pre_core", "alt_core", "compact", "is_domain",
               "has_alias", "indic", "n_indic_oov"]
ADDR_FIELDS = ["norm", "bag", "region", "numbers", "has_box", "n_placeholders"]

# --------------------------------------------------------------- workers
_NAME: NameNormalizer | None = None
_ADDR: AddressNormalizer | None = None


def _init_worker(dictionary: dict[str, str], vocab: dict[str, str] | None = None) -> None:
    global _NAME, _ADDR
    tr = Transliterator(dictionary, vocab)
    _NAME, _ADDR = NameNormalizer(tr), AddressNormalizer(tr)


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a frame with entity_id, business_name, business_address, country
    (+ any passthrough columns). Runs in a worker (or in-process for tests)."""
    assert _NAME is not None and _ADDR is not None, "call _init_worker first"
    names = [asdict(_NAME(x)) for x in df.business_name]
    addrs = [asdict(_ADDR(a, c)) for a, c in zip(df.business_address, df.country)]
    out = df.rename(columns={"business_name": "name_raw", "business_address": "addr_raw"})
    out = out.reset_index(drop=True)
    n = pd.DataFrame(names, columns=NAME_FIELDS).add_prefix("name_")
    a = pd.DataFrame(addrs, columns=ADDR_FIELDS).add_prefix("addr_")
    a = a.rename(columns={"addr_region": "region"})
    return pd.concat([out, n, a], axis=1)


# ----------------------------------------------------------------- stage


def _workers(cfg: Config) -> int:
    return cfg.normalize.workers or max(1, (os.cpu_count() or 4) - 2)


def _learn(st: StageRun, cfg: Config,
           split_q: pd.DataFrame) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Learn the dictionary on train-side pairs, build the S1 skeleton vocabulary
    (train + test S1: fully known at inference), report coverage on val pairs."""
    with timed(st.log, "learn Indic->Latin dictionary + S1 vocabulary"):
        s1 = read_source(cfg, "train", 1).set_index("entity_id").business_name
        test_s1 = (list(read_source(cfg, "test", 1).business_name)
                   if (cfg.data_dir / "test" / "test_source1.tsv").exists() else [])
        vocab = build_vocab(list(s1) + test_s1)
        q = pd.concat([read_source(cfg, "train", n)[["entity_id", "business_name"]]
                       for n in (2, 3)], ignore_index=True)
        q = q[q.business_name.str.contains(INDIC_RE.pattern, regex=True)]
        q = q.merge(split_q[["entity_id", "owner", "side"]], on="entity_id")
        q = q[q.owner != ""]
        q["s1_name"] = s1.reindex(q.owner).to_numpy()
        tr, va = q[q.side == "train"], q[q.side == "val"]
        d = learn_dictionary(tr.business_name, tr.s1_name,
                             cfg.normalize.dict_min_count, cfg.normalize.dict_min_share)
        st.write_parquet(d, "translit_dict")
        lut = dict(zip(d.indic, d.latin))
        st.metric("translit.train_pairs", len(tr))
        st.metric("translit.dictionary_size", len(lut))
        st.metric("translit.vocab_skeletons", len(vocab))
        st.metric("translit.val_coverage", {k: round(v, 4) if isinstance(v, float) else v
                                            for k, v in dictionary_coverage(
                                                lut, vocab, va.business_name,
                                                va.s1_name).items()})
        lut_all = lut
        if cfg.normalize.test_dict_all_pairs:
            d_all = learn_dictionary(q.business_name, q.s1_name,
                                     cfg.normalize.dict_min_count, cfg.normalize.dict_min_share)
            st.write_parquet(d_all, "translit_dict_all")
            lut_all = dict(zip(d_all.indic, d_all.latin))
            st.metric("translit.dictionary_size_all_pairs", len(lut_all))
    return lut, vocab, lut_all


def _normalize_split(st: StageRun, cfg: Config, split: str, lut: dict[str, str],
                     vocab: dict[str, str], sides: dict[int, pd.DataFrame] | None,
                     dropped: set[str] = frozenset()) -> None:
    out_dir = st.path(f"records_{split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.parquet"):
        old.unlink()
    total, part = 0, 0
    with ProcessPoolExecutor(_workers(cfg), initializer=_init_worker,
                             initargs=(lut, vocab)) as pool:
        for n in (1, 2, 3):
            with timed(st.log, f"normalize {split} source{n}"):
                df = read_source(cfg, split, n)
                df.insert(1, "source", n)
                if sides is not None:  # train: attach split info, apply slice
                    if n == 1 and dropped:  # reshape: these S1 are not in the pool
                        df = df[~df.entity_id.isin(dropped)]
                    df = df.merge(sides[n], on="entity_id", how="left", validate="1:1")
                    df = df[slice_mask(df, cfg)]
                size = cfg.normalize.chunk_size
                chunks = [df.iloc[i:i + size] for i in range(0, len(df), size)]
                for res in pool.map(normalize_frame, chunks):
                    write_parquet(res, out_dir / f"part-{part:05d}.parquet")
                    part += 1
                    total += len(res)
                st.log.info("%s source%d: %d records", split, n, len(df))
    st.register(f"records_{split}", out_dir, rows=total, parts=part)
    st.metric(f"records_{split}", total)


def run_normalize(cfg: Config) -> None:
    with StageRun(cfg, "normalize") as st:
        s1_split = st.read_input("split", "s1", run=cfg.data.split_run)
        q_split = st.read_input("split", "queries", run=cfg.data.split_run)
        lut, vocab, lut_all = _learn(st, cfg, q_split)

        rs = cfg.reshape
        dropped = dropped_s1(s1_split, rs.drop_s1, rs.sides, rs.seed)
        if rs.drop_s1:
            by_c = s1_split[s1_split.entity_id.isin(dropped)].country.value_counts()
            st.metric("reshape_dropped_s1", {str(k): int(v) for k, v in by_c.items()})
        q_split = orphan_queries(q_split, dropped)
        st.metric("reshape_orphaned_queries", int((q_split.owner_dropped != "").sum()))
        cols = ["entity_id", "side", "dev", "geo"]
        s1_kept = s1_split[~s1_split.entity_id.isin(dropped)]
        sides = {1: s1_kept[cols].assign(owner="", owner_dropped=""),
                 2: q_split.loc[q_split.entity_id.str.startswith("S2-"),
                                cols + ["owner", "owner_dropped"]],
                 3: q_split.loc[q_split.entity_id.str.startswith("S3-"),
                                cols + ["owner", "owner_dropped"]]}
        _normalize_split(st, cfg, "train", lut, vocab, sides, dropped)
        if cfg.run_test:
            _normalize_split(st, cfg, "test", lut_all, vocab, None)
        else:
            st.log.info("test split skipped (slice=%s, with_test=%s)", cfg.data.slice,
                        cfg.data.with_test)


def read_records(path: Path, columns: list[str] | None = None,
                 filters: list | None = None) -> pd.DataFrame:
    """Read a ``records_<split>`` directory (all parts)."""
    return pd.read_parquet(path, columns=columns, filters=filters)
