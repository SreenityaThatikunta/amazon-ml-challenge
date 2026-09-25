"""Two-stage LightGBM matcher + decision rule."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl

ID_COLS = {"s1_idx", "o_idx", "src", "y", "s1", "mid", "brank_dummy"}

LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.08,
    num_leaves=127,
    min_data_in_leaf=100,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    max_bin=255,
    num_threads=0,
    verbose=-1,
    seed=0,
)


def feature_columns(df: pl.DataFrame, extra_exclude: set | None = None) -> list[str]:
    ex = ID_COLS | (extra_exclude or set())
    return [c for c, t in df.schema.items() if c not in ex and t.is_numeric()]


def to_numpy(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()


def fit(df: pl.DataFrame, cols: list[str], rounds: int = 400, valid: pl.DataFrame | None = None, params: dict | None = None):
    prm = dict(LGB_PARAMS, **(params or {}))
    dtr = lgb.Dataset(to_numpy(df, cols), label=df["y"].to_numpy(), feature_name=cols, free_raw_data=True)
    vs, cb = [], []
    if valid is not None:
        vs = [lgb.Dataset(to_numpy(valid, cols), label=valid["y"].to_numpy(), reference=dtr)]
        cb = [lgb.log_evaluation(100)]
    return lgb.train(prm, dtr, num_boost_round=rounds, valid_sets=vs, callbacks=cb)


def predict(booster, df: pl.DataFrame, cols: list[str], chunk: int = 2_000_000) -> np.ndarray:
    out = []
    for i in range(0, df.height, chunk):
        out.append(booster.predict(to_numpy(df.slice(i, chunk), cols)))
    return np.concatenate(out) if out else np.zeros(0)


def _other_max(x: str, grp: list[str] | str) -> pl.Expr:
    c = pl.col(x)
    m1 = c.max().over(grp)
    m2 = c.top_k(2).min().over(grp)
    n = pl.len().over(grp)
    return pl.when(n == 1).then(pl.lit(0.0)).when(c == m1).then(m2).otherwise(m1)


def stage2_features(df: pl.DataFrame, pcol: str = "p1") -> pl.DataFrame:
    """Context features derived from stage-1 probabilities.

    o-side: does another Source-1 candidate explain this record better?
    s1-side: how many records does this entity attract, how strongly?
    """
    og = ["src", "o_idx"]
    df = df.with_columns(
        _other_max(pcol, og).cast(pl.Float32).alias("p1_o_other"),
        pl.col(pcol).rank("ordinal", descending=True).over(og).cast(pl.Float32).alias("p1_o_rank"),
        pl.col(pcol).sum().over(og).cast(pl.Float32).alias("p1_o_sum"),
    )
    # a record "belongs" to its best candidate; entity-level stats use those best links only
    best = pl.col("p1_o_rank") == 1
    df = df.with_columns(
        (pl.col(pcol) - pl.col("p1_o_other")).cast(pl.Float32).alias("p1_o_margin"),
        (best & (pl.col(pcol) > 0.5)).cast(pl.Float32).alias("_strong"),
        pl.when(best).then(pl.col(pcol)).otherwise(0.0).alias("_pbest"),
    )
    df = df.with_columns(
        (pl.col("_strong").sum().over("s1_idx") - pl.col("_strong")).cast(pl.Float32).alias("p1_s1_nstrong_other"),
        (pl.col("_pbest").sum().over("s1_idx") - pl.col("_pbest")).cast(pl.Float32).alias("p1_s1_psum_other"),
        _other_max("_pbest", "s1_idx").cast(pl.Float32).alias("p1_s1_pmax_other"),
        pl.len().over("s1_idx").cast(pl.Float32).alias("p1_s1_ncand"),
    ).drop("_strong", "_pbest")
    return df


def decide(df: pl.DataFrame, pcol: str, thr: float) -> pl.DataFrame:
    """Each Source-2/3 record is linked to its single best Source-1 candidate if p >= thr."""
    best = df.filter(pl.col(pcol) == pl.col(pcol).max().over(["src", "o_idx"]))
    best = best.unique(["src", "o_idx"], keep="first")
    return best.filter(pl.col(pcol) >= thr).select("s1_idx", "src", "o_idx", pcol)
