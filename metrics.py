"""Macro F0.5 exactly as defined by the challenge."""
from __future__ import annotations

import polars as pl


def macro_f05(pred: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series) -> dict:
    """pred/truth: long frames (s1, mid). s1_ids: all Source-1 ids being evaluated."""
    base = pl.DataFrame({"s1": s1_ids})
    tp = pred.join(truth, on=["s1", "mid"], how="inner").group_by("s1").len().rename({"len": "tp"})
    npred = pred.group_by("s1").len().rename({"len": "np"})
    ntrue = truth.group_by("s1").len().rename({"len": "nt"})
    d = (
        base.join(tp, on="s1", how="left")
        .join(npred, on="s1", how="left")
        .join(ntrue, on="s1", how="left")
        .fill_null(0)
    )
    p = pl.when(pl.col("np") > 0).then(pl.col("tp") / pl.col("np")).otherwise(0.0)
    r = pl.when(pl.col("nt") > 0).then(pl.col("tp") / pl.col("nt")).otherwise(0.0)
    f = pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0).when(pl.col("tp") == 0).then(0.0).otherwise(
        1.25 * p * r / (0.25 * p + r)
    )
    d = d.with_columns(p.alias("P"), r.alias("R"), f.alias("F"))
    single = d.filter(pl.col("nt") == 0)
    multi = d.filter(pl.col("nt") > 0)
    return {
        "F05": d["F"].mean(),
        "F05_singletons": single["F"].mean() if single.height else None,
        "F05_nonsingle": multi["F"].mean() if multi.height else None,
        "micro_P": d["tp"].sum() / max(d["np"].sum(), 1),
        "micro_R": d["tp"].sum() / max(d["nt"].sum(), 1),
        "n": d.height,
        "_per_entity": d,
    }
