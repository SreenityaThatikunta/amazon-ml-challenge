"""``ber decide`` (tasks T3.5/T4.1): pair scores -> final matches, scored on val.

1. one owner per query: each S2/S3 record keeps only its highest-scoring S1
   (EDA: every S2/S3 record matches at most one S1); optionally exact duplicate
   records inside a source are tied to one owner;
2. per S1 either
   * ``threshold``: keep assigned queries with p >= t, or
   * ``expected_f``: keep the prefix maximising expected F0.5 (see
     :mod:`ber.decide.expected_f`), after dropping p < ``min_p``;
3. val: macro F0.5 over every val S1 (singletons included), threshold swept;
   test: decisions written for ``ber submit``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..artifacts import StageRun
from ..config import Config
from ..eval.metrics import per_entity, summarize
from ..eval.split import slice_mask
from ..log import timed
from .expected_f import select


def one_owner(scores: pd.DataFrame) -> pd.DataFrame:
    """Keep each query's best S1 (ties: first)."""
    s = scores.sort_values(["q_id", "p"], ascending=[True, False])
    return s.drop_duplicates("q_id", keep="first")


def dup_keys(recs: pd.DataFrame) -> pd.Series:
    """Key shared by exact duplicate records (same source, country, raw name and
    address), indexed by entity_id."""
    key = recs.source.astype(str) + "|" + recs.country + "|" + recs.name_raw + "|" + recs.addr_raw
    return pd.Series(pd.util.hash_pandas_object(key, index=False).to_numpy(),
                     index=recs.entity_id.to_numpy())


def tie_duplicates(assigned: pd.DataFrame, keys: pd.Series) -> pd.DataFrame:
    """Exact duplicates within a source (~20k per source, EDA 01) must share an
    owner: every member takes the owner and p of the group's best-scoring member."""
    a = assigned.copy()
    a["k"] = keys.reindex(a.q_id.to_numpy()).to_numpy()
    grp = a.groupby("k")
    size = grp.q_id.transform("size")
    if (size > 1).any():
        best = a.loc[grp.p.idxmax(), ["k", "s1_id", "p"]].set_index("k")
        m = size > 1
        a.loc[m, "s1_id"] = best.s1_id.reindex(a.k[m]).to_numpy()
        a.loc[m, "p"] = best.p.reindex(a.k[m]).to_numpy()
    return a.drop(columns="k")


def decide(assigned: pd.DataFrame, method: str, threshold: float, min_p: float,
           empty_prior: float) -> pd.DataFrame:
    if method == "threshold":
        return assigned[assigned.p >= threshold]
    if method == "expected_f":
        return select(assigned[assigned.p >= min_p], empty_prior)
    raise ValueError(f"unknown decide.method {method!r}")


def _as_lists(kept: pd.DataFrame) -> dict[str, list[str]]:
    return kept.groupby("s1_id").q_id.apply(list).to_dict()


def evaluate(kept: pd.DataFrame, truth: dict[str, list[str]], s1_meta: pd.DataFrame) -> pd.DataFrame:
    """Per-entity F0.5 for every val S1, with country / has-Indic-match columns."""
    df = per_entity(truth, _as_lists(kept), keys=s1_meta.index)
    return df.join(s1_meta, on="s1_id")


def run_decide(cfg: Config) -> None:
    dc = cfg.decide
    with StageRun(cfg, "decide") as st:
        recs = pd.read_parquet(st.input_path("normalize", "records_train"),
                               columns=["entity_id", "source", "country", "side", "dev",
                                        "owner", "name_indic", "name_raw", "addr_raw"])
        s1 = recs[(recs.source == 1) & slice_mask(recs, cfg, "val")]
        s1_meta = s1.set_index("entity_id")[["country"]]
        q = recs[(recs.source != 1) & recs.owner.isin(s1_meta.index)]
        truth = q.groupby("owner").entity_id.apply(list).to_dict()
        indic_s1 = set(q.owner[q.name_indic])
        s1_meta["has_indic_match"] = s1_meta.index.isin(indic_s1)
        s1_meta["segment"] = np.where(s1_meta.index.isin(set(truth)), "has_matches", "singleton")
        st.metric("val_s1", {"n": len(s1_meta), "singletons": int((s1_meta.segment ==
                                                                    "singleton").sum())})

        scores = st.read_input(dc.scores_stage, "scores_val").astype({"q_id": str, "s1_id": str})
        assigned = one_owner(scores)
        st.metric("val_one_owner_precision_at_1", round(float(assigned.y.mean()), 5))
        if dc.tie_duplicates:
            q_keys = dup_keys(recs[recs.source != 1])
            before = assigned.s1_id.copy()
            assigned = tie_duplicates(assigned, q_keys)
            st.metric("val_duplicates_retied", int((assigned.s1_id != before).sum()))

        # sweep: threshold method over t, expected_f over min_p
        sweep = {}
        for t in dc.sweep:
            f_thr = evaluate(decide(assigned, "threshold", t, 0, 1), truth, s1_meta).f05.mean()
            f_exp = evaluate(decide(assigned, "expected_f", 0, t, dc.empty_prior), truth,
                             s1_meta).f05.mean()
            sweep[t] = {"threshold": round(float(f_thr), 5), "expected_f_min_p": round(float(f_exp), 5)}
        st.metric("val_sweep", sweep)
        best_t = max(dc.sweep, key=lambda t: sweep[t]["threshold"])
        best_m = max(dc.sweep, key=lambda t: sweep[t]["expected_f_min_p"])
        chosen = {"method": dc.method,
                  "threshold": best_t if dc.threshold is None else dc.threshold,
                  "min_p": best_m if dc.min_p is None else dc.min_p}
        st.metric("chosen", chosen)
        st.write_json(chosen, "decision_params")

        kept = decide(assigned, dc.method, chosen["threshold"], chosen["min_p"], dc.empty_prior)
        ent = evaluate(kept, truth, s1_meta)
        st.write_parquet(ent, "val_per_entity")
        st.write_parquet(kept.reset_index(drop=True), "decisions_val")
        st.metric("val_f05", round(float(ent.f05.mean()), 5))
        for by in ("country", "segment", "has_indic_match"):
            s = summarize(ent, by).round(5)
            st.metric(f"val_by_{by}", {str(k): v for k, v in s.to_dict("index").items()})
        st.metric("val_overall", summarize(ent).round(5).to_dict("index")["ALL"])

        if cfg.run_test:
            with timed(st.log, "decide test"):
                sc = pd.read_parquet(st.input_path(dc.scores_stage, "scores_test"))
                a_t = one_owner(sc.astype({"q_id": str, "s1_id": str}))
                if dc.tie_duplicates:
                    rt = pd.read_parquet(st.input_path("normalize", "records_test"),
                                         columns=["entity_id", "source", "country", "name_raw",
                                                  "addr_raw"])
                    a_t = tie_duplicates(a_t, dup_keys(rt[rt.source != 1]))
                kept_t = decide(a_t, dc.method, chosen["threshold"], chosen["min_p"],
                                dc.empty_prior)
                st.write_parquet(kept_t.reset_index(drop=True), "decisions_test")
                st.metric("test_matches", len(kept_t))
