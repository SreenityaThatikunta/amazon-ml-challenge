"""Candidate generation (blocking).

Every record is turned into a bag of *blocking keys* (all country-scoped):

  N  : single core-name token            (legal forms / honorifics removed)
  NN : unordered pair of core-name tokens (robust to word-order transpositions)
  P  : first 5 chars of the space-less core name (domains, glued words)
  C  : the whole space-less core name       (exact name / domain match)
  CS : the sorted core-name tokens          (exact name up to word order)
  A  : single alphabetic address token    (state codes excluded)
  AA : consecutive address-token bigram   (street name + type, city bigrams)
  H  : house/plot number + following address token
  #  : any number with >= 3 digits
  NX : core-name token x address number    (disambiguates very common names;
       on the Source-1 side numbers are also indexed with one digit dropped,
       to survive truncated house numbers such as 3391 -> 391)

A key's weight is its IDF in Source 1 (per country); keys shared by more than
`max_df` Source-1 records are dropped as stop-keys.  For each Source-2/3
record we sum the weights of the keys it shares with each Source-1 record
(a sparse inverted-index join) and keep the top-K Source-1 records.  Because
every Source-2/3 record belongs to at most one Source-1 entity, ranking from
the Source-2/3 side bounds the candidate set at K per record while keeping
recall high.
"""
from __future__ import annotations

import math
import time

import polars as pl

KINDS = ["N", "NN", "P", "C", "CS", "A", "AA", "H", "#", "NX"]
NAME_KINDS = {0, 1, 2, 3, 4}
BLOCK_COLS = ["ctry", "n_toks", "n_cat", "a_words", "a_full", "a_nums"]


def _explode_pos(df: pl.DataFrame, col: str, max_len: int) -> pl.DataFrame:
    return (
        df.select("idx", "c", pl.col(col).list.head(max_len).alias("v"))
        .with_columns(pl.int_ranges(0, pl.col("v").list.len(), dtype=pl.UInt8).alias("pos"))
        .explode("v", "pos")
        .drop_nulls("v")
    )


def _hash(c: pl.Expr, kind: str, v: pl.Expr) -> pl.Expr:
    return (c + pl.lit(kind + ":") + v).hash(seed=7)


def _num_variants(e: pl.Expr) -> pl.Expr:
    """Numbers plus their one-digit-truncated forms (1058 -> 105, 058) – used on the index side."""
    return pl.concat_list(
        [e, e.list.eval(pl.element().filter(pl.element().str.len_chars() >= 3).str.replace(r"\d$", "")),
         e.list.eval(pl.element().filter(pl.element().str.len_chars() >= 3).str.replace(r"^\d", "").str.strip_chars_start("0"))]
    ).list.eval(pl.element().filter(pl.element() != "")).list.unique()


def _keys(df: pl.DataFrame, index_side: bool = False) -> pl.DataFrame:
    """Return long frame (idx:u32, key:u64, kind:u8) for a normalised source frame."""
    base = df.select(
        "idx",
        (pl.col("ctry") + pl.lit("|")).alias("c"),
        pl.col("n_toks").list.eval(pl.element().filter(pl.element().str.len_chars() >= 2)).list.unique(maintain_order=True).alias("nt"),
        pl.col("n_cat"),
        pl.col("a_words").list.eval(
            pl.element().filter((pl.element().str.len_chars() >= 2) & ~pl.element().str.starts_with("state_"))
        ).alias("aw"),
        pl.col("a_full"),
        pl.col("a_nums").list.eval(pl.element().filter(pl.element().str.len_chars() >= 3)).list.unique().alias("an"),
        (_num_variants(pl.col("a_nums").list.head(3)) if index_side else pl.col("a_nums").list.head(3).list.unique()).alias("nx"),
    )
    out = []

    def add(t: pl.DataFrame, kind: str, v: pl.Expr):
        out.append(t.select("idx", _hash(pl.col("c"), kind, v).alias("key"), pl.lit(KINDS.index(kind), dtype=pl.UInt8).alias("kind")))

    nt = _explode_pos(base, "nt", 6)
    add(nt, "N", pl.col("v"))
    pairs = nt.join(nt.select("idx", pl.col("v").alias("v2"), pl.col("pos").alias("pos2")), on="idx")
    pairs = pairs.filter(pl.col("pos") < pl.col("pos2"))
    add(pairs, "NN", pl.min_horizontal("v", "v2") + pl.lit(" ") + pl.max_horizontal("v", "v2"))
    p = base.filter(pl.col("n_cat").str.len_chars() >= 4)
    add(p, "P", pl.col("n_cat").str.slice(0, 5))
    add(base.filter(pl.col("n_cat").str.len_chars() >= 3), "C", pl.col("n_cat"))
    add(base.filter(pl.col("nt").list.len() >= 2), "CS", pl.col("nt").list.sort().list.join(" "))
    nx = nt.filter(pl.col("pos") < 4).join(
        base.select("idx", pl.col("nx").alias("num")).explode("num").drop_nulls("num"), on="idx"
    )
    add(nx, "NX", pl.col("v") + pl.lit("|") + pl.col("num"))
    aw = _explode_pos(base, "aw", 12)
    add(aw.filter(pl.col("v").str.len_chars() >= 3), "A", pl.col("v"))
    nxt = aw.filter(pl.col("pos") >= 1).select("idx", pl.col("v").alias("v2"), (pl.col("pos") - 1).cast(pl.UInt8).alias("pos"))
    big = aw.join(nxt, on=["idx", "pos"])
    add(big, "AA", pl.col("v") + pl.lit(" ") + pl.col("v2"))
    h = base.select("idx", "c", pl.col("a_full").str.extract_all(r"\b[1-9]\d* \w+").list.unique().alias("v")).explode("v").drop_nulls("v")
    add(h, "H", pl.col("v"))
    n = base.select("idx", "c", pl.col("an").alias("v")).explode("v").drop_nulls("v")
    add(n, "#", pl.col("v"))
    return pl.concat(out).unique(["idx", "key"])


def build_s1_index(s1: pl.DataFrame, max_df: int = 100) -> pl.DataFrame:
    """Inverted index over Source 1: (key, s1_idx, w, isname)."""
    parts = []
    step = 500_000
    s1 = s1.select(BLOCK_COLS).with_columns(pl.int_range(pl.len(), dtype=pl.UInt32).alias("idx"))
    for i in range(0, s1.height, step):
        parts.append(_keys(s1.slice(i, step), index_side=True))
    k = pl.concat(parts)
    df = k.group_by("key").agg(pl.len().cast(pl.UInt32).alias("df"))
    n_total = max(int(s1.group_by("ctry").len()["len"].max()), 1)
    k = k.join(df.filter(pl.col("df") <= max_df), on="key", how="inner")
    k = k.with_columns(
        (pl.lit(math.log(n_total)) - pl.col("df").cast(pl.Float32).log()).cast(pl.Float32).alias("w"),
        pl.col("kind").is_in(list(NAME_KINDS)).alias("isname"),
    )
    return k.select("key", pl.col("idx").alias("s1_idx"), "w", "isname")


def generate_candidates(
    index: pl.DataFrame,
    other: pl.DataFrame,
    top_k: int = 10,
    min_rel: float = 0.4,
    chunk: int = 100_000,
    verbose: bool = True,
) -> pl.DataFrame:
    """Return (o_idx, s1_idx, bscore, bname, baddr, bnkeys, brank).

    Keeps, for every Source-2/3 record, the top_k Source-1 records by blocking
    score whose score is at least `min_rel` x the record's best score.
    """
    t0 = time.time()
    other = other.select(BLOCK_COLS).with_columns(pl.int_range(pl.len(), dtype=pl.UInt32).alias("idx"))
    res = []
    for i in range(0, other.height, chunk):
        o = other.slice(i, chunk)
        ok = _keys(o).select("idx", "key")
        j = ok.join(index, on="key", how="inner")
        g = j.group_by("idx", "s1_idx").agg(
            pl.col("w").sum().alias("bscore"),
            pl.col("w").filter(pl.col("isname")).sum().alias("bname"),
            pl.col("w").filter(~pl.col("isname")).sum().alias("baddr"),
            pl.len().cast(pl.UInt16).alias("bnkeys"),
        )
        g = g.with_columns(
            pl.col("bscore").rank("ordinal", descending=True).over("idx").cast(pl.UInt16).alias("brank"),
            (pl.col("bscore") / pl.col("bscore").max().over("idx")).alias("_rel"),
        ).filter((pl.col("brank") <= top_k) & (pl.col("_rel") >= min_rel)).drop("_rel")
        res.append(g.rename({"idx": "o_idx"}))
        if verbose and (i // chunk) % 10 == 0:
            print(f"    chunk {i//chunk}: joined {j.height:,} -> kept {g.height:,} ({time.time()-t0:.0f}s)", flush=True)
    return pl.concat(res)
