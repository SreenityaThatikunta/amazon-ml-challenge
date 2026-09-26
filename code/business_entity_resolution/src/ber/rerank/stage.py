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
# stage-1 features copied to the stage-2 score files: they define the segments of the
# prior-shift correction in ber decide (lookalike type: house-number relation, legal
# conflict, exact name key, address similarity)
SEGMENT_COLS = ["hn_rel", "l_conflict", "k_eq", "a_tsr", "n_core_eq"]


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
    f = _F32(f)
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
    return pd.DataFrame(dict(f), copy=False)


class _F32(dict):
    """dict that stores every value as float32 (keeps the test pass in memory)."""

    def __init__(self, d):
        super().__init__()
        for k, v in d.items():
            self[k] = v

    def __setitem__(self, k, v):
        super().__setitem__(k, np.asarray(v, dtype=np.float32))


# Features whose scale depends on the size / density of the data set they were computed
# on (adversarial validation val vs test: AUC 0.986-0.9997; test US has half the S1 of
# train US and 23 % more S2/S3 records per S1). With ``rerank.domain_norm`` they are
# divided by their mean over the same country and split, so a model trained on val reads
# test on the same scale.
COUNT_COLS = ["s_n", "s_sum_other", "s_n_hi_other", "ss_sum_other", "ss_n_hi_other",
              "c_sum_other", "c_n_hi_other", "c_expected", "cs_n_hi_other", "q_n",
              "q_sum_other", "q_n_hi_other", "s_n_cands", "s_n_top1", "s_name_freq",
              "q_n_cands"]
IDF_COLS = ["k_idf_unmatched_q", "k_idf_unmatched_s", "a_idf_unmatched_q", "a_idf_unmatched_s"]


def domain_normalize(X: pd.DataFrame, groups: np.ndarray | None = None) -> pd.DataFrame:
    """Count-like columns / their mean, IDF-like columns / their 99th percentile of
    non-zero values, within each group (country) of ``X`` (in place; returns X)."""
    groups = np.zeros(len(X), dtype=np.int8) if groups is None else groups
    for c in COUNT_COLS + IDF_COLS:
        if c not in X:
            continue
        v = X[c].to_numpy(dtype=np.float32, copy=True)
        for g in np.unique(groups):
            m = groups == g
            if c in COUNT_COLS:
                ref = float(v[m].mean()) or 1.0
            else:
                nz = v[m][v[m] > 0]
                ref = float(np.quantile(nz, 0.99)) if len(nz) else 1.0
            v[m] = v[m] / ref
        X[c] = v
    return X


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
    in order (val side only), so the two line up row by row (checked part by
    part); only the kept rows of each part are held in memory."""
    sv = st.read_input("scorer", "scores_val")
    # candidate-set size vs recall for other filter levels (the organisers rank smaller
    # candidate sets higher): pairs per val query and share of true pairs kept
    n_q = max(sv.q_id.nunique(), 1)
    st.metric("filter_tradeoff", {t: {"pairs_per_query": round(float((sv.p >= t).sum()) / n_q, 3),
                                      "true_pairs_kept": round(float(sv.y[sv.p >= t].sum())
                                                               / max(float(sv.y.sum()), 1), 5)}
                                  for t in (0.0, 0.0003, 0.001, 0.003, 0.01, 0.03)})
    keep_all = (sv.p >= cfg.rerank.min_p1).to_numpy()
    sv_q = sv.q_id.astype(str).to_numpy()
    parts, off = [], 0
    for f in sorted(st.input_path("features", "features_train").glob("*.parquet")):
        d = pd.read_parquet(f, columns=["q_id", "side"] + raw_cols)
        d = d[(d.side == "val").to_numpy()]
        n = len(d)
        if n == 0:
            continue
        if off + n > len(sv) or not (d.q_id.astype(str).to_numpy() == sv_q[off:off + n]).all():
            raise RuntimeError(f"scores_val and features_train val rows are not aligned ({f.name})")
        parts.append(d.loc[keep_all[off:off + n], raw_cols])
        off += n
    if off != len(sv):
        raise RuntimeError("scores_val has rows beyond the val features")
    raw = pd.concat(parts, ignore_index=True)
    return sv[keep_all].reset_index(drop=True), raw


def _raw_columns(st: StageRun, wanted: list[str]) -> list[str]:
    """Stage-1 features passed to stage 2; ["all"] = every stage-1 feature."""
    if list(wanted) != ["all"]:
        return list(wanted)
    import pyarrow.parquet as pq
    first = sorted(st.input_path("features", "features_train").glob("*.parquet"))[0]
    return [c for c in pq.ParquetFile(first).schema_arrow.names
            if c not in ("q_id", "s1_id", "y", "side")]


def run_rerank(cfg: Config) -> None:
    rc = cfg.rerank
    with StageRun(cfg, "rerank") as st:
        raw_cols = _raw_columns(st, rc.raw_features)
        recs = pd.read_parquet(st.input_path("normalize", "records_train"),
                               columns=["entity_id", "source"])
        if cfg.run_train:
            q_cat = pd.CategoricalDtype(recs.entity_id[recs.source != 1].to_numpy())
            s_cat = pd.CategoricalDtype(recs.entity_id[recs.source == 1].to_numpy())
            with timed(st.log, "load val pairs"):
                keys, raw = _read_val(st, cfg, raw_cols)
                X, q = _frame(keys, raw, q_cat, s_cat)
                y = keys.y.to_numpy()
                if rc.domain_norm:
                    cty = pd.read_parquet(st.input_path("normalize", "records_train"),
                                          columns=["entity_id", "country"],
                                          filters=[("side", "==", "val")]
                                          ).set_index("entity_id").country
                    groups = pd.factorize(cty.reindex(keys.q_id.astype(str)).fillna(""))[0]
                    X = domain_normalize(X, groups)
            st.metric("val_pairs", {"kept": len(keys), "positives": int(y.sum())})

            # folds grouped by the query's top S1 (keeps most S1 groups in one fold)
            top = keys.assign(q=q).sort_values(["q", "p"], ascending=[True, False]
                                               ).drop_duplicates("q")
            grp = pd.Series(stable_unit(top.s1_id.astype(str), cfg.seed + 7),
                            index=top.q.to_numpy())
            u = grp.reindex(q).to_numpy()
            fold = np.minimum((u * rc.folds).astype(int), rc.folds - 1)
            if rc.fold_by == "region":  # robustness check: whole regions held out per fold
                reg = pd.read_parquet(st.input_path("normalize", "records_train"),
                                      columns=["entity_id", "region"],
                                      filters=[("source", "==", 1), ("side", "==", "val")]
                                      ).set_index("entity_id").region
                top_reg = pd.Series(reg.reindex(top.s1_id.astype(str)).fillna("").to_numpy(),
                                    index=top.q.to_numpy())
                sizes = top_reg.value_counts()
                # largest first, each region to the currently smallest fold (balanced sizes)
                load, fold_of = [0] * rc.folds, {}
                for r, n in sizes.items():
                    k = int(np.argmin(load))
                    fold_of[r] = k
                    load[k] += int(n)
                fold = np.array([fold_of[r] for r in top_reg.reindex(q).to_numpy()])
                st.metric("fold_regions", {k: sorted(r for r, f in fold_of.items() if f == k)
                                           for k in range(rc.folds)})
            oof = np.zeros(len(X), dtype=np.float64)
            imp = np.zeros(X.shape[1])
            w = None
            if rc.weights_file:  # e.g. adversarial-validation weights (val rows made test-shaped)
                wf = pd.read_parquet(rc.weights_file)
                w = (keys[["q_id", "s1_id"]].astype(str)
                     .merge(wf.astype({"q_id": str, "s1_id": str}), on=["q_id", "s1_id"], how="left")
                     .w.fillna(1.0).to_numpy(dtype=np.float64))
                st.metric("train_weights", {"file": rc.weights_file,
                                            "q": np.quantile(w, [0.01, 0.5, 0.99]).round(3).tolist()})
            if rc.model_from_run:  # e.g. v3's models on val_ts: fold k model on fold k pairs
                with timed(st.log, f"apply {rc.model_from_run}'s {rc.folds} fold models"):
                    for k in range(rc.folds):
                        te = fold == k
                        src = st.input_path("rerank", "importance", run=rc.model_from_run).parent
                        m = lgb.Booster(model_file=str(src / f"model_fold{k}.txt"))
                        oof[te] = m.predict(X[te][m.feature_name()],
                                            num_iteration=m.best_iteration)
                        imp += m.feature_importance("gain")
            with timed(st.log, f"cross-fit {rc.folds} folds"):
                for k in range(rc.folds if not rc.model_from_run else 0):
                    tr, te = fold != k, fold == k
                    # early stopping on 10% of the training groups (never the held-out fold)
                    es = tr & (u * 10 % 1 < 0.1)
                    fit = tr & ~es
                    dtr = lgb.Dataset(X[fit], y[fit], free_raw_data=True,
                                      weight=None if w is None else w[fit])
                    des = lgb.Dataset(X[es], y[es], reference=dtr,
                                      weight=None if w is None else w[es])
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
            out = keys[["q_id", "s1_id", "y"]].assign(p=oof.astype(np.float32),
                                                      p1=keys.p.to_numpy())
            for c in SEGMENT_COLS:  # kept for the prior-shift correction in decide
                if c in X:
                    out[c] = X[c].to_numpy()
            st.write_parquet(out, "scores_val")
            del out
            del X, raw, keys

        if cfg.run_test:
            _score_test(st, cfg, raw_cols)


def _score_test(st: StageRun, cfg: Config, raw_cols: list[str]) -> None:
    """Test pairs with p1 >= min_p1, one country at a time (S1 and query groups
    never cross countries, and the stage-1 score parts are written country by
    country), scored by the mean of the fold models."""
    rc = cfg.rerank
    rt = pd.read_parquet(st.input_path("normalize", "records_test"),
                         columns=["entity_id", "source", "country"])
    q_cat = pd.CategoricalDtype(rt.entity_id[rt.source != 1].to_numpy())
    s_cat = pd.CategoricalDtype(rt.entity_id[rt.source == 1].to_numpy())
    s1_country = rt[rt.source == 1].set_index("entity_id").country
    models = [lgb.Booster(model_file=str(st.path(f"model_fold{k}.txt")))
              for k in range(rc.folds)]
    sc_dir = st.input_path("scorer", "scores_test")
    ft_dir = st.input_path("features", "features_test")
    by_country: dict[str, list] = {}
    for f in sorted(sc_dir.glob("*.parquet")):
        first = pd.read_parquet(f, columns=["s1_id"]).s1_id.iloc[:1].astype(str)
        if len(first):
            by_country.setdefault(s1_country[first.iloc[0]], []).append(f)
    out_dir = st.path("scores_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.parquet"):
        old.unlink()
    n_total = 0
    for country, files in sorted(by_country.items()):
        with timed(st.log, f"test {country}: {len(files)} parts"):
            ks, rs = [], []
            for f in files:
                sc = pd.read_parquet(f)
                ft = pd.read_parquet(ft_dir / f.name, columns=["q_id"] + raw_cols)
                if len(ft) != len(sc) or not (ft.q_id.astype(str).to_numpy()
                                              == sc.q_id.astype(str).to_numpy()).all():
                    raise RuntimeError(f"scores/features part {f.name} not aligned")
                keep = (sc.p >= rc.min_p1).to_numpy()
                sc = sc[keep]
                ks.append(pd.DataFrame({"q": _codes(sc.q_id, q_cat),
                                        "s": _codes(sc.s1_id, s_cat), "p": sc.p.to_numpy()}))
                rs.append(ft.loc[keep, raw_cols].reset_index(drop=True))
                del sc, ft
            k = pd.concat(ks, ignore_index=True)
            raw = pd.concat(rs, ignore_index=True)
            del ks, rs
            X = group_features(k.q.to_numpy(), k.s.to_numpy(), raw.q_source.to_numpy(),
                               k.p.to_numpy())
            for c in raw_cols:
                X[c] = raw[c].to_numpy(dtype=np.float32)
            del raw
            if rc.domain_norm:  # this frame is one country of the test split
                X = domain_normalize(X)
            p2 = np.zeros(len(X), dtype=np.float64)
            step = 2_000_000
            for a in range(0, len(X), step):
                xa = X.iloc[a:a + step]
                p2[a:a + step] = np.mean([m.predict(xa, num_iteration=m.best_iteration)
                                          for m in models], axis=0)
            seg = {c: X[c].to_numpy() for c in SEGMENT_COLS if c in X}
            del X
            out = pd.DataFrame({"q_id": q_cat.categories.to_numpy()[k.q.to_numpy()],
                                "s1_id": s_cat.categories.to_numpy()[k.s.to_numpy()],
                                "p": p2.astype(np.float32), "p1": k.p.to_numpy(), **seg})
            out.to_parquet(out_dir / f"part-{country}.parquet", index=False)
            n_total += len(out)
            st.metric(f"test_pairs_kept.{country}", len(out))
            del out, k, p2
    st.register("scores_test", out_dir, rows=n_total)
