"""``ber llm-export`` (tasks T3.6 + T5.3): route uncertain queries, build prompts.

Routing: a query goes to the LLM when its best scorer probability lies in
``[route_low, route_high]`` (dev: ~10% of queries in [0.05, 0.95], where the
scorer's top pick is right only ~59% of the time). Everything else is settled
by the scorer + decision layer.

Exports ``llm_<split>.parquet`` with q_id, prompt, candidate ids (in prompt
order), their scorer probabilities, and for train/val the label (1..N, 0 = none):
  * ``train``: routed train-side queries (out-of-fold scores) plus an
    ``easy_frac`` sample of the rest, so the model also sees clear cases
  * ``val``:   routed val-side queries (for blending/evaluation)
  * ``test``:  routed test queries
Candidates are shown in a seeded shuffled order, so the model can't learn that
the first candidate is usually right.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..artifacts import StageRun
from ..config import Config
from ..eval.split import stable_unit
from ..log import timed
from .prompt import PROMPT_VERSION, build_prompt

HINT_COLS = ["n_core_eq", "q_is_domain", "n_q_is_initials", "n_s_is_initials", "n_lev",
             "n_sub", "n_transp", "n_tsr", "l_eq", "l_conflict", "hn_rel", "num_q_only",
             "box_q", "box_s", "r_eq", "r_conflict", "a_tsr"]
REC_COLS = ["entity_id", "name_raw", "addr_raw", "country", "source"]


def route(scores: pd.DataFrame, low: float, high: float) -> pd.Series:
    """Per query: True if its best probability is inside [low, high]."""
    best = scores.groupby("q_id").p.max()
    return (best >= low) & (best <= high)


def top_candidates(scores: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Top-n candidates per query by p, then shuffled per query (seeded)."""
    s = scores.sort_values(["q_id", "p"], ascending=[True, False])
    s = s.groupby("q_id", sort=False).head(n).copy()
    s["_u"] = stable_unit(s.q_id + "|" + s.s1_id, seed)
    return s.sort_values(["q_id", "_u"]).drop(columns="_u")


def export(scores: pd.DataFrame, feats: pd.DataFrame, recs: pd.DataFrame, n: int, seed: int,
           owners: pd.Series | None) -> pd.DataFrame:
    """Build one prompt per query in ``scores`` (already restricted to routed queries)."""
    top = top_candidates(scores, n, seed)
    top = top.merge(feats, on=["q_id", "s1_id"], how="left")
    r = recs.set_index("entity_id")
    top = top.join(r[["name_raw", "addr_raw"]], on="s1_id")
    rows = []
    for qid, g in top.groupby("q_id", sort=False):
        q = r.loc[qid]
        cand_ids = g.s1_id.tolist()
        label = None
        if owners is not None:
            o = owners.get(qid, "")
            label = cand_ids.index(o) + 1 if o in cand_ids else 0
        rows.append({"q_id": qid, "prompt": build_prompt(q, g), "cands": cand_ids,
                     "gbm_p": g.p.astype(float).round(4).tolist(), "label": label})
    return pd.DataFrame(rows)


def run_llm_export(cfg: Config) -> None:
    lc = cfg.llm
    with StageRun(cfg, "llm-export") as st:
        st.metric("prompt_version", PROMPT_VERSION)
        feats_dir = st.input_path("features", "features_train")
        feats = pd.read_parquet(feats_dir, columns=["q_id", "s1_id"] + HINT_COLS)
        feats = feats.astype({"q_id": str, "s1_id": str})
        recs = pd.read_parquet(st.input_path("normalize", "records_train"),
                               columns=REC_COLS + ["owner"])
        owners = recs.set_index("entity_id").owner

        jobs = [("train", st.read_input("scorer", "scores_train_oof")),
                ("val", st.read_input("scorer", "scores_val"))]
        for split, scores in jobs:
            scores = scores.astype({"q_id": str, "s1_id": str})
            routed = route(scores, lc.route_low, lc.route_high)
            keep = routed
            if split == "train":
                u = pd.Series(stable_unit(routed.index.to_series(), cfg.seed + 3),
                              index=routed.index)
                keep = routed | (u < lc.easy_frac)
            with timed(st.log, f"export {split}: {int(keep.sum()):,} prompts "
                               f"({routed.mean():.1%} routed)"):
                out = export(scores[scores.q_id.isin(keep[keep].index)], feats, recs,
                             lc.n_candidates, cfg.seed, owners)
                out["routed"] = routed.reindex(out.q_id).to_numpy()
            st.write_parquet(out, f"llm_{split}")
            st.metric(f"{split}_prompts", {
                "n": len(out), "routed_share": round(float(routed.mean()), 4),
                "label_none": round(float((out.label == 0).mean()), 4),
                "chars_p50": int(out.prompt.str.len().median()),
                "chars_p95": int(out.prompt.str.len().quantile(0.95))})

        if cfg.run_test:
            with timed(st.log, "export test"):
                sc = pd.read_parquet(st.input_path("scorer", "scores_test")).astype(
                    {"q_id": str, "s1_id": str})
                routed = route(sc, lc.route_low, lc.route_high)
                sc = sc[sc.q_id.isin(routed[routed].index)]
                ft = pd.read_parquet(st.input_path("features", "features_test"),
                                     columns=["q_id", "s1_id"] + HINT_COLS,
                                     filters=[("q_id", "in", list(routed[routed].index))])
                rt = pd.read_parquet(st.input_path("normalize", "records_test"),
                                     columns=REC_COLS)
                out = export(sc, ft.astype({"q_id": str, "s1_id": str}), rt,
                             lc.n_candidates, cfg.seed, None)
                st.write_parquet(out, "llm_test")
                st.metric("test_prompts", {"n": len(out),
                                           "routed_share": round(float(routed.mean()), 4)})
