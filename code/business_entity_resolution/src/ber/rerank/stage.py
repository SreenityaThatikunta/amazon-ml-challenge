"""``ber rerank``: stage-2 re-ranker with group context from the stage-1 scores.

The stage-1 scorer judges each (query, S1) pair on its own, plus how the pair
compares with the *query's* other candidates. It never sees the *S1 side* of the
assignment: how many other S2/S3 records already claim the same S1, how strong
those claims are, and whether they come from the same source. That is exactly
what separates a true match from a lookalike distractor sitting next to it
(EDA §3: distractors copy an S1's name/address with a shifted house number or a
one-letter edit, so both records score high against the same S1), and a
singleton S1 from one that attracts a stray record.

Stage 2 is a small LightGBM on:
  * p1 (stage-1 probability) and its logit;
  * query context: rank of the pair in the query, best / sum of the query's
    other p1, number of strong candidates;
  * S1 context: rank of the query among all records listing the S1, best / sum
    / count of the other records' p1, the same restricted to the query's own
    source, and "claims" (records whose best S1 is this one);
  * a few stage-1 features (source, number relation, name/address equality...),
    so stage 2 can condition the context on the kind of pair.

Training data must hold *complete* S1 groups, so it is the val side (every val
query was featurised and scored by a stage-1 model that never saw it). The val
side is cross-fitted (``folds`` folds grouped by the query's top S1): every val
pair gets an out-of-fold p2, so the val F0.5 computed by ``ber decide`` on p2 is
honest. Test pairs get the mean of the fold models. Pairs with p1 < ``min_p1``
are dropped first (they are never selected; this keeps the test pass in memory).

Outputs ``scores_val.parquet`` and ``scores_test/`` with the scorer's schema
(``p`` = p2), so ``ber decide --set decide.scores_stage=rerank`` consumes them.
"""

from __future__ import annotations

import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from ..artifacts import StageRun
from ..config import Config
from ..eval.split import stable_unit
from ..features.build import _best_other
from ..log import timed

HI = 0.5  # "strong" candidate / claim


def _group_stats(key: np.ndarray, v: np.ndarray) -> dict[str, np.ndarray]:
    """Per row: rank of v in its key group (0 = best), best v among the *other*
    rows of the group (-1 if none), sum and count>HI of the other rows, group size."""
    best, other, rank = _best_other(key, v.astype(np.float64))
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    tot = np.bincount(inv, weights=v)
    hi = np.bincount(inv, weights=(v > HI).astype(np.float64))
    return {"rank": rank, "best_other": np.nan_to_num(other, nan=-1.0),
            "sum_other": tot[inv] - v, "n_hi_other": hi[inv] - (v > HI),
            "size": cnt[inv]}


def group_features(q: np.ndarray, s: np.ndarray, src: np.ndarray,
                   p: np.ndarray) -> pd.DataFrame:
    """Stage-2 context features for pairs given as integer codes (q, s), the
    query's source (2/3) and the stage-1 probability p."""
    p = p.astype(np.float64)
    f = {"p1": p, "p1_logit": np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1))}
    g = _group_stats(q, p)
    f["q_rank"], f["q_best_other"], f["q_sum_other"] = g["rank"], g["best_other"], g["sum_other"]
    f["q_n_hi_other"], f["q_n"] = g["n_hi_other"], g["size"]
    f["q_margin"] = p - np.maximum(g["best_other"], 0)

    g = _group_stats(s, p)
    f["s_rank"], f["s_best_other"], f["s_sum_other"] = g["rank"], g["best_other"], g["sum_other"]
    f["s_n_hi_other"], f["s_n"] = g["n_hi_other"], g["size"]
    f["s_margin"] = p - np.maximum(g["best_other"], 0)

    ss = s.astype(np.int64) * 4 + src.astype(np.int64)
    g = _group_stats(ss, p)
    f["ss_rank"], f["ss_best_other"] = g["rank"], g["best_other"]
    f["ss_sum_other"], f["ss_n_hi_other"] = g["sum_other"], g["n_hi_other"]

    # claims: records whose best candidate is this S1 (what one-owner assignment keeps)
    claim = np.where(f["q_rank"] == 0, p, 0.0)
    g = _group_stats(s, claim)
    f["c_rank"], f["c_best_other"] = g["rank"], g["best_other"]
    f["c_sum_other"], f["c_n_hi_other"] = g["sum_other"], g["n_hi_other"]
    g = _group_stats(ss, claim)
    f["cs_best_other"], f["cs_n_hi_other"] = g["best_other"], g["n_hi_other"]
    f["c_expected"] = f["c_sum_other"] + claim  # expected true matches claimed by this S1
    out = pd.DataFrame(f)
    for c in out.columns:
        out[c] = out[c].astype(np.float32)
    return out


def _codes(values: pd.Series, dtype: pd.CategoricalDtype) -> np.ndarray:
    c = pd.Categorical(values.astype(str), dtype=dtype).codes
    if (c < 0).any():
        raise ValueError("pair ids missing from the records")
    return c.astype(np.int32)


def _frame(keys: pd.DataFrame, raw: pd.DataFrame, q_cat, s_cat) -> tuple[pd.DataFrame, np.ndarray]:
    """Stage-2 feature frame for pairs (keys: q_id, s1_id, p; raw: stage-1 features)."""
    q = _codes(keys.q_id, q_cat)
    s = _codes(keys.s1_id, s_cat)
    src = raw.q_source.to_numpy()
    X = group_features(q, s, src, keys.p.to_numpy())
    for c in raw.columns:
        X[c] = raw[c].to_numpy(dtype=np.float32)
    return X, q


def _params(cfg: Config) -> dict:
    rc = cfg.rerank
    return {"objective": "binary", "learning_rate": rc.learning_rate,
            "num_leaves": rc.num_leaves, "min_data_in_leaf": rc.min_data_in_leaf,
            "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1,
            "lambda_l2": 1.0, "seed": cfg.seed, "verbose": -1,
            "num_threads": cfg.scorer.n_threads or max(1, (os.cpu_count() or 4) - 2)}


def _read_val(st: StageRun, cfg: Config, raw_cols: list[str]):
    """Val pairs with p1 >= min_p1: keys (q_id, s1_id, y, p) and raw stage-1 features,
    row-aligned. scores_val was written by streaming the features_train parts
    in order (val side only), so the two line up row by row; this is checked."""
    sv = st.read_input("scorer", "scores_val")
    parts = []
    for f in sorted(st.input_path("features", "features_train").glob("*.parquet")):
        d = pd.read_parquet(f, columns=["q_id", "s1_id", "side"] + raw_cols)
        parts.append(d[d.side == "val"].drop(columns="side"))
    raw = pd.concat(parts, ignore_index=True)
    if len(raw) != len(sv) or not (raw.q_id.astype(str).to_numpy()
                                   == sv.q_id.astype(str).to_numpy()).all():
        raise RuntimeError("scores_val and features_train val rows are not aligned")
    keep = (sv.p >= cfg.rerank.min_p1).to_numpy()
    return (sv[keep].reset_index(drop=True),
            raw.loc[keep, raw_cols].reset_index(drop=True))


def run_rerank(cfg: Config) -> None:
    rc = cfg.rerank
    raw_cols = list(rc.raw_features)
    with StageRun(cfg, "rerank") as st:
        recs = pd.read_parquet(st.input_path("normalize", "records_train"),
                               columns=["entity_id", "source"])
        if cfg.run_train:
            q_cat = pd.CategoricalDtype(recs.entity_id[recs.source != 1].to_numpy())
            s_cat = pd.CategoricalDtype(recs.entity_id[recs.source == 1].to_numpy())
            with timed(st.log, "load val pairs"):
                keys, raw = _read_val(st, cfg, raw_cols)
                X, q = _frame(keys, raw, q_cat, s_cat)
                y = keys.y.to_numpy()
            st.metric("val_pairs", {"kept": len(keys), "positives": int(y.sum())})

            # folds grouped by the query's top S1 (keeps most S1 groups in one fold)
            top = keys.assign(q=q).sort_values(["q", "p"], ascending=[True, False]
                                               ).drop_duplicates("q")
            grp = pd.Series(stable_unit(top.s1_id.astype(str), cfg.seed + 7),
                            index=top.q.to_numpy())
            u = grp.reindex(q).to_numpy()
            fold = np.minimum((u * rc.folds).astype(int), rc.folds - 1)
            oof = np.zeros(len(X), dtype=np.float64)
            imp = np.zeros(X.shape[1])
            with timed(st.log, f"cross-fit {rc.folds} folds"):
                for k in range(rc.folds):
                    tr, te = fold != k, fold == k
                    # early stopping on 10% of the training groups (never the held-out fold)
                    es = tr & (u * 10 % 1 < 0.1)
                    fit = tr & ~es
                    dtr = lgb.Dataset(X[fit], y[fit], free_raw_data=True)
                    des = lgb.Dataset(X[es], y[es], reference=dtr)
                    m = lgb.train(_params(cfg), dtr, num_boost_round=rc.num_boost_round,
                                  valid_sets=[des], valid_names=["es"],
                                  callbacks=[lgb.early_stopping(rc.early_stopping, verbose=False)])
                    oof[te] = m.predict(X[te], num_iteration=m.best_iteration)
                    imp += m.feature_importance("gain")
                    m.save_model(str(st.path(f"model_fold{k}.txt")))
                    st.log.info("fold %d: best iteration %d, held-out AUC %.6f", k,
                                m.best_iteration, roc_auc_score(y[te], oof[te]))
            st.register("models", st.dir)
            st.metric("val_auc_p1", round(float(roc_auc_score(y, keys.p)), 6))
            st.metric("val_auc_p2_oof", round(float(roc_auc_score(y, oof)), 6))
            st.metric("val_logloss_p1", round(float(log_loss(y, keys.p.clip(1e-7, 1 - 1e-7))), 6))
            st.metric("val_logloss_p2_oof", round(float(log_loss(y, oof.clip(1e-7, 1 - 1e-7))), 6))
            imp_df = pd.DataFrame({"feature": X.columns, "gain": imp}).sort_values(
                "gain", ascending=False)
            st.write_parquet(imp_df, "importance")
            st.log.info("stage-2 features by gain:\n%s", imp_df.to_string(index=False))
            st.write_parquet(keys[["q_id", "s1_id", "y"]].assign(
                p=oof.astype(np.float32), p1=keys.p.to_numpy()), "scores_val")
            del X, raw, keys

        if cfg.run_test:
            _score_test(st, cfg, raw_cols)


def _score_test(st: StageRun, cfg: Config, raw_cols: list[str]) -> None:
    """All test pairs with p1 >= min_p1 (the S1 context needs every part at once),
    scored by the mean of the fold models."""
    rc = cfg.rerank
    rt = pd.read_parquet(st.input_path("normalize", "records_test"),
                         columns=["entity_id", "source"])
    q_cat = pd.CategoricalDtype(rt.entity_id[rt.source != 1].to_numpy())
    s_cat = pd.CategoricalDtype(rt.entity_id[rt.source == 1].to_numpy())
    models = [lgb.Booster(model_file=str(st.path(f"model_fold{k}.txt")))
              for k in range(rc.folds)]
    with timed(st.log, "load test pairs"):
        sc_dir = st.input_path("scorer", "scores_test")
        ft_dir = st.input_path("features", "features_test")
        ks, rs = [], []
        for f in sorted(sc_dir.glob("*.parquet")):
            sc = pd.read_parquet(f)
            ft = pd.read_parquet(ft_dir / f.name, columns=["q_id"] + raw_cols)
            if len(ft) != len(sc) or not (ft.q_id.astype(str).to_numpy()
                                          == sc.q_id.astype(str).to_numpy()).all():
                raise RuntimeError(f"scores/features part {f.name} not aligned")
            keep = (sc.p >= rc.min_p1).to_numpy()
            sc = sc[keep]
            ks.append(pd.DataFrame({"q": _codes(sc.q_id, q_cat), "s": _codes(sc.s1_id, s_cat),
                                    "p": sc.p.to_numpy()}))
            rs.append(ft.loc[keep, raw_cols].reset_index(drop=True))
        k = pd.concat(ks, ignore_index=True)
        raw = pd.concat(rs, ignore_index=True)
    st.metric("test_pairs_kept", len(k))
    with timed(st.log, "stage-2 test scores"):
        X = group_features(k.q.to_numpy(), k.s.to_numpy(), raw.q_source.to_numpy(),
                           k.p.to_numpy())
        for c in raw_cols:
            X[c] = raw[c].to_numpy(dtype=np.float32)
        del raw
        p2 = np.mean([m.predict(X, num_iteration=m.best_iteration) for m in models], axis=0)
    out_dir = st.path("scores_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.parquet"):
        old.unlink()
    out = pd.DataFrame({"q_id": q_cat.categories.to_numpy()[k.q.to_numpy()],
                        "s1_id": s_cat.categories.to_numpy()[k.s.to_numpy()],
                        "p": p2.astype(np.float32), "p1": k.p.to_numpy()})
    out.to_parquet(out_dir / "part-00000.parquet", index=False)
    st.register("scores_test", out_dir, rows=len(out))
