"""Pairwise feature engineering for (Source-1, Source-2/3) candidate pairs."""
from __future__ import annotations

import math

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

REC_COLS = ["n_full", "n_core", "n_cat", "n_alt", "n_toks", "n_isdom", "n_hasdba", "a_full", "a_nums", "a_words", "ctry", "is_indic", "addr_empty"]


def enrich(df: pl.DataFrame) -> pl.DataFrame:
    """Per-record derived columns used by the pairwise features."""
    return df.with_columns(
        pl.col("business_name").str.contains(r"[ऀ-ൿ]").alias("is_indic"),
        (pl.col("a_full").str.len_chars() == 0).alias("addr_empty"),
        pl.col("a_words").list.eval(pl.element().filter(~pl.element().str.starts_with("state_"))).alias("a_words"),
        pl.col("a_words").list.eval(pl.element().filter(pl.element().str.starts_with("state_"))).list.first().alias("a_state"),
        pl.col("n_toks").list.eval(pl.element().str.slice(0, 1)).list.join("").alias("n_init"),
    )


def build_stats(s1: pl.DataFrame) -> dict:
    """IDF tables and name-ambiguity counts computed on Source 1 of the same split."""
    n = s1.height
    tok = (
        s1.select("ctry", pl.col("n_toks").list.unique().alias("t")).explode("t").drop_nulls("t")
        .group_by("ctry", "t").len()
        .with_columns((math.log(n) - pl.col("len").cast(pl.Float64).log()).cast(pl.Float32).alias("idf"))
        .select("ctry", "t", "idf")
    )
    aw = (
        s1.select("ctry", pl.col("a_words").list.unique().alias("t")).explode("t").drop_nulls("t")
        .group_by("ctry", "t").len()
        .with_columns((math.log(n) - pl.col("len").cast(pl.Float64).log()).cast(pl.Float32).alias("idf"))
        .select("ctry", "t", "idf")
    )
    amb = s1.group_by("ctry", "n_cat").len().rename({"len": "amb"})
    return {"tok_idf": tok, "aw_idf": aw, "name_amb": amb, "max_idf": float(math.log(n))}


def _cp(a, b, scorer, **kw):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw)


def _idf_overlap(pairs: pl.DataFrame, la: str, lb: str, table: pl.DataFrame, max_idf: float, prefix: str) -> pl.DataFrame:
    """IDF-weighted overlap statistics between two token-list columns."""
    t = table.rename({"t": "_t", "idf": "_idf"})

    def side_sum(col):
        e = pairs.select("pid", "ctry", pl.col(col).list.unique().alias("_t")).explode("_t").drop_nulls("_t")
        e = e.join(t, on=["ctry", "_t"], how="left").with_columns(pl.col("_idf").fill_null(max_idf))
        return e

    ea = side_sum(la)
    eb = side_sum(lb)
    sa = ea.group_by("pid").agg(pl.col("_idf").sum().alias("sa"))
    sb = eb.group_by("pid").agg(pl.col("_idf").sum().alias("sb"))
    sh = ea.join(eb.select("pid", "_t"), on=["pid", "_t"], how="inner").group_by("pid").agg(
        pl.col("_idf").sum().alias("sh"), pl.len().alias("nsh"), pl.col("_idf").max().alias("shmax")
    )
    # most informative token that is NOT shared, on each side
    ua = ea.join(eb.select("pid", "_t"), on=["pid", "_t"], how="anti").group_by("pid").agg(pl.col("_idf").max().alias("ua"))
    ub = eb.join(ea.select("pid", "_t"), on=["pid", "_t"], how="anti").group_by("pid").agg(pl.col("_idf").max().alias("ub"))
    r = (
        pairs.select("pid")
        .join(sa, on="pid", how="left").join(sb, on="pid", how="left").join(sh, on="pid", how="left")
        .join(ua, on="pid", how="left").join(ub, on="pid", how="left")
        .fill_null(0)
    )
    eps = 1e-6
    return r.select(
        "pid",
        (pl.col("sh") / (pl.col("sa") + eps)).cast(pl.Float32).alias(f"{prefix}_wrec"),
        (pl.col("sh") / (pl.col("sb") + eps)).cast(pl.Float32).alias(f"{prefix}_wprec"),
        (pl.col("sh") / (pl.col("sa") + pl.col("sb") - pl.col("sh") + eps)).cast(pl.Float32).alias(f"{prefix}_wjac"),
        pl.col("sh").cast(pl.Float32).alias(f"{prefix}_shidf"),
        pl.col("nsh").cast(pl.Float32).alias(f"{prefix}_nsh"),
        pl.col("shmax").cast(pl.Float32).alias(f"{prefix}_shmax"),
        pl.col("ua").cast(pl.Float32).alias(f"{prefix}_unmatched_a"),
        pl.col("ub").cast(pl.Float32).alias(f"{prefix}_unmatched_b"),
    )


def pair_features(pairs: pl.DataFrame, A: pl.DataFrame, B: pl.DataFrame, stats: dict) -> pl.DataFrame:
    """pairs: (s1_idx, o_idx, blocking cols...). A/B enriched record frames indexed by position."""
    cols = REC_COLS + ["a_state", "n_init"]
    a = A.select(cols).gather(pairs["s1_idx"])
    b = B.select(cols).gather(pairs["o_idx"])
    P = pl.concat(
        [
            pairs.with_row_index("pid"),
            a.rename({c: c + "_a" for c in cols}),
            b.rename({c: c + "_b" for c in cols}),
        ],
        how="horizontal",
    ).with_columns(pl.col("ctry_a").alias("ctry"))

    L = lambda c: P[c].to_list()  # noqa: E731
    nfa, nfb = L("n_full_a"), L("n_full_b")
    nca, ncb = L("n_core_a"), L("n_core_b")
    cta, ctb = L("n_cat_a"), L("n_cat_b")
    ada, adb = L("a_full_a"), L("a_full_b")
    alt_b, alt_a = L("n_alt_b"), L("n_alt_a")
    feats = {
        "n_full_ratio": _cp(nfa, nfb, fuzz.ratio),
        "n_core_ratio": _cp(nca, ncb, fuzz.ratio),
        "n_core_tsort": _cp(nca, ncb, fuzz.token_sort_ratio),
        "n_core_tset": _cp(nca, ncb, fuzz.token_set_ratio),
        "n_core_partial": _cp(nca, ncb, fuzz.partial_ratio),
        "n_core_jw": _cp(nca, ncb, JaroWinkler.normalized_similarity),
        "n_cat_ratio": _cp(cta, ctb, fuzz.ratio),
        "n_cat_partial": _cp(cta, ctb, fuzz.partial_ratio),
        "n_altb_ratio": _cp(nfa, alt_b, fuzz.ratio),
        "n_alta_ratio": _cp(alt_a, nfb, fuzz.ratio),
        "a_ratio": _cp(ada, adb, fuzz.ratio),
        "a_tsort": _cp(ada, adb, fuzz.token_sort_ratio),
        "a_tset": _cp(ada, adb, fuzz.token_set_ratio),
        "a_partial": _cp(ada, adb, fuzz.partial_ratio),
    }
    del nfa, nfb, nca, ncb, ada, adb, alt_a, alt_b
    # numbers as strings
    numa = P["a_nums_a"].list.join(" ").to_list()
    numb = P["a_nums_b"].list.join(" ").to_list()
    feats["num_ratio"] = _cp(numa, numb, fuzz.ratio)
    feats["num_tset"] = _cp(numa, numb, fuzz.token_set_ratio)
    del numa, numb
    F = pl.DataFrame(feats)

    na0 = pl.col("a_nums_a").list.first()
    nb0 = pl.col("a_nums_b").list.first()
    X = P.select(
        "pid",
        # name structure
        pl.col("n_core_a").str.len_chars().cast(pl.Float32).alias("n_len_a"),
        pl.col("n_core_b").str.len_chars().cast(pl.Float32).alias("n_len_b"),
        pl.col("n_toks_a").list.len().cast(pl.Float32).alias("n_ntok_a"),
        pl.col("n_toks_b").list.len().cast(pl.Float32).alias("n_ntok_b"),
        (pl.col("n_core_a") == pl.col("n_core_b")).cast(pl.Float32).alias("n_core_eq"),
        (pl.col("n_cat_a") == pl.col("n_cat_b")).cast(pl.Float32).alias("n_cat_eq"),
        (pl.col("n_toks_a").list.sort() == pl.col("n_toks_b").list.sort()).cast(pl.Float32).alias("n_toks_eq"),
        ((pl.col("n_init_a") == pl.col("n_cat_b")) & (pl.col("n_cat_b").str.len_chars() >= 2)).cast(pl.Float32).alias("n_init_ab"),
        ((pl.col("n_cat_b").str.len_chars() > 0) & pl.col("n_cat_a").str.contains(pl.col("n_cat_b"), literal=True)).cast(pl.Float32).alias("n_cat_b_in_a"),
        ((pl.col("n_cat_a").str.len_chars() > 0) & pl.col("n_cat_b").str.contains(pl.col("n_cat_a"), literal=True)).cast(pl.Float32).alias("n_cat_a_in_b"),
        pl.col("n_isdom_b").cast(pl.Float32).alias("isdom_b"),
        pl.col("n_hasdba_b").cast(pl.Float32).alias("hasdba_b"),
        pl.col("n_hasdba_a").cast(pl.Float32).alias("hasdba_a"),
        pl.col("is_indic_b").cast(pl.Float32).alias("indic_b"),
        # address structure
        pl.col("addr_empty_b").cast(pl.Float32).alias("addr_empty_b"),
        pl.col("addr_empty_a").cast(pl.Float32).alias("addr_empty_a"),
        pl.col("a_full_a").str.len_chars().cast(pl.Float32).alias("a_len_a"),
        pl.col("a_full_b").str.len_chars().cast(pl.Float32).alias("a_len_b"),
        pl.when(pl.col("a_state_a").is_null() | pl.col("a_state_b").is_null()).then(pl.lit(0.5))
        .otherwise((pl.col("a_state_a") == pl.col("a_state_b")).cast(pl.Float64)).cast(pl.Float32).alias("state_eq"),
        # numbers
        pl.col("a_nums_a").list.len().cast(pl.Float32).alias("num_n_a"),
        pl.col("a_nums_b").list.len().cast(pl.Float32).alias("num_n_b"),
        pl.col("a_nums_a").list.set_intersection("a_nums_b").list.len().cast(pl.Float32).alias("num_nsh"),
        pl.col("a_nums_a").list.set_intersection("a_nums_b").list.eval(pl.element().str.len_chars()).list.max().fill_null(0).cast(pl.Float32).alias("num_sh_maxlen"),
        pl.when(na0.is_null() | nb0.is_null()).then(pl.lit(-1.0)).otherwise((na0 == nb0).cast(pl.Float64)).cast(pl.Float32).alias("num_first_eq"),
        pl.when(na0.is_null() | nb0.is_null()).then(pl.lit(-1.0)).otherwise(
            (na0.str.starts_with(nb0) | nb0.str.starts_with(na0) | na0.str.ends_with(nb0) | nb0.str.ends_with(na0)).cast(pl.Float64)
        ).cast(pl.Float32).alias("num_first_affix"),
        pl.when(na0.is_null() | nb0.is_null()).then(pl.lit(-1.0)).otherwise(
            (na0.str.slice(0, 9).cast(pl.Float64, strict=False) - nb0.str.slice(0, 9).cast(pl.Float64, strict=False)).abs()
            / (pl.max_horizontal(na0.str.slice(0, 9).cast(pl.Float64, strict=False), nb0.str.slice(0, 9).cast(pl.Float64, strict=False)) + 1)
        ).cast(pl.Float32).alias("num_first_reldiff"),
        pl.col("a_words_a").list.len().cast(pl.Float32).alias("aw_n_a"),
        pl.col("a_words_b").list.len().cast(pl.Float32).alias("aw_n_b"),
    )
    # IDF-weighted token overlaps
    ov_n = _idf_overlap(P.select("pid", "ctry", "n_toks_a", "n_toks_b"), "n_toks_a", "n_toks_b", stats["tok_idf"], stats["max_idf"], "nt")
    ov_a = _idf_overlap(P.select("pid", "ctry", "a_words_a", "a_words_b"), "a_words_a", "a_words_b", stats["aw_idf"], stats["max_idf"], "aw")
    # name ambiguity: how many Source-1 entities share this exact core name
    amb = stats["name_amb"]
    am = P.select("pid", "ctry", pl.col("n_cat_a").alias("n_cat")).join(amb, on=["ctry", "n_cat"], how="left").select("pid", pl.col("amb").fill_null(0).cast(pl.Float32).alias("amb_a"))
    bm = P.select("pid", "ctry", pl.col("n_cat_b").alias("n_cat")).join(amb, on=["ctry", "n_cat"], how="left").select("pid", pl.col("amb").fill_null(0).cast(pl.Float32).alias("amb_b"))
    out = pl.concat([pairs, F, X.drop("pid")], how="horizontal")
    for t in (ov_n, ov_a, am, bm):
        out = pl.concat([out, t.sort("pid").drop("pid")], how="horizontal")
    return out


def _margin(col: str, grp: str) -> pl.Expr:
    """value minus the best value among the OTHER members of the group (value itself if alone)."""
    x = pl.col(col)
    m1 = x.max().over(grp)
    m2 = x.top_k(2).min().over(grp)
    n = pl.len().over(grp)
    other = pl.when(n == 1).then(pl.lit(0.0)).when(x == m1).then(m2).otherwise(m1)
    return (x - other).cast(pl.Float32)


MARGIN_COLS = ["bscore", "n_core_tset", "n_full_ratio", "a_tset", "a_ratio", "aw_wjac", "nt_wjac", "num_tset"]


def group_features(df: pl.DataFrame) -> pl.DataFrame:
    """Competition features: how this pair compares to the other candidates of the same record."""
    df = df.with_columns((pl.col("n_core_tset") + pl.col("a_tset")).alias("sum_na"))
    return df.with_columns(
        [(pl.col("bscore") / pl.col("bscore").max().over("o_idx")).cast(pl.Float32).alias("b_rel"),
         pl.len().over("o_idx").cast(pl.Float32).alias("o_ncand")]
        + [_margin(c, "o_idx").alias(f"mg_{c}") for c in MARGIN_COLS + ["sum_na"]]
    )


def s1_group_features(df: pl.DataFrame) -> pl.DataFrame:
    """Features over all candidates proposed for the same Source-1 entity (needs the full frame)."""
    return df.with_columns(
        pl.len().over("s1_idx").cast(pl.Float32).alias("s1_ncand"),
        _margin("sum_na", "s1_idx").alias("s1mg_sum_na"),
    )
