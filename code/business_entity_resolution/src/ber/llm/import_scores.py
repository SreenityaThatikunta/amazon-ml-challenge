"""``ber llm-import`` (tasks T5.5/T5.6): bring Kaggle LLM scores back and blend.

Input: ``llm_scores_<split>.parquet`` files from ``kaggle/infer.py`` (q_id, cands,
probs = [p_none, p_cand1, ..., p_candN]) placed in ``llm.scores_dir``.

For every routed query, each shown candidate's pair probability becomes
``w * p_llm + (1 - w) * p_scorer`` (candidates the LLM did not see keep
``(1 - w) * p_scorer``, so the LLM can also veto them). Non-routed queries keep
the scorer's probability. The best ``w`` is chosen on val by macro F0.5 via the
decision layer, and the blended val/test scores are written with the scorer's
schema so ``ber decide`` can consume them (``decide.scores_stage = llm-import``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..artifacts import StageRun
from ..config import Config
from ..log import timed


def llm_pair_probs(llm: pd.DataFrame) -> pd.DataFrame:
    """Explode Kaggle output into (q_id, s1_id, p_llm) rows."""
    rows = []
    for qid, cands, probs in zip(llm.q_id, llm.cands, llm.probs):
        for s1, p in zip(cands, list(probs)[1:]):
            rows.append((qid, s1, float(p)))
    return pd.DataFrame(rows, columns=["q_id", "s1_id", "p_llm"])


def blend(scores: pd.DataFrame, llm: pd.DataFrame, w: float) -> pd.DataFrame:
    """Blend LLM probabilities into scorer scores for the queries the LLM saw."""
    pp = llm_pair_probs(llm)
    out = scores.merge(pp, on=["q_id", "s1_id"], how="left")
    seen = out.q_id.isin(set(llm.q_id))
    p_llm = out.p_llm.fillna(0.0)
    out["p"] = np.where(seen, w * p_llm + (1 - w) * out.p, out.p).astype(np.float32)
    return out.drop(columns="p_llm")


def run_llm_import(cfg: Config) -> None:
    from ..decide.stage import decide as decide_fn
    from ..decide.stage import evaluate, one_owner
    from ..eval.split import slice_mask

    lc = cfg.llm
    src = cfg.resolve(lc.scores_dir)
    with StageRun(cfg, "llm-import") as st:
        scores_val = st.read_input("scorer", "scores_val").astype({"q_id": str, "s1_id": str})
        llm_val = pd.read_parquet(src / "llm_scores_val.parquet")
        st.metric("val_llm_queries", len(llm_val))

        recs = pd.read_parquet(st.input_path("normalize", "records_train"),
                               columns=["entity_id", "source", "country", "side", "dev", "owner"])
        s1 = recs[(recs.source == 1) & slice_mask(recs, cfg, "val")]
        s1_meta = s1.set_index("entity_id")[["country"]]
        q = recs[(recs.source != 1) & recs.owner.isin(s1_meta.index)]
        truth = q.groupby("owner").entity_id.apply(list).to_dict()

        dc = cfg.decide
        sweep = {}
        for w in lc.blend_sweep:
            b = one_owner(blend(scores_val, llm_val, w))
            best = max((evaluate(decide_fn(b, "expected_f", 0, m, dc.empty_prior), truth,
                                 s1_meta).f05.mean(), m) for m in dc.sweep)
            sweep[w] = {"f05": round(float(best[0]), 5), "min_p": best[1]}
        st.metric("val_blend_sweep", sweep)
        w = lc.blend_weight if lc.blend_weight is not None else \
            max(sweep, key=lambda k: sweep[k]["f05"])
        st.metric("blend_weight", w)
        st.write_parquet(blend(scores_val, llm_val, w), "scores_val")

        if cfg.run_test and (src / "llm_scores_test.parquet").exists():
            with timed(st.log, "blend test"):
                llm_test = pd.read_parquet(src / "llm_scores_test.parquet")
                sc = pd.read_parquet(st.input_path("scorer", "scores_test")).astype(
                    {"q_id": str, "s1_id": str})
                out_dir = st.path("scores_test")
                out_dir.mkdir(parents=True, exist_ok=True)
                blend(sc, llm_test, w).to_parquet(out_dir / "part-00000.parquet", index=False)
                st.register("scores_test", out_dir, rows=len(sc))
