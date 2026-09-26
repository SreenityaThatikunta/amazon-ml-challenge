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
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
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


def domain_normalize(M: np.ndarray, cols: list[str], groups: np.ndarray | None = None) -> np.ndarray:
    """Count-like columns / their mean, IDF-like columns / their 99th percentile of
    non-zero values, within each group (country) of the rows of the float32 matrix
    ``M`` whose columns are ``cols`` (in place; returns M)."""
    idx = {c: j for j, c in enumerate(cols)}
    groups = np.zeros(len(M), dtype=np.int8) if groups is None else groups
    ug = np.unique(groups)
    for c in COUNT_COLS + IDF_COLS:
        j = idx.get(c)
        if j is None:
            continue
        v = np.ascontiguousarray(M[:, j])  # contiguous float32 copy of the column
        for g in ug:
            m = groups == g
            if c in COUNT_COLS:
                ref = float(v[m].mean()) or 1.0
            else:
                nz = v[m][v[m] > 0]
                ref = float(np.quantile(nz, 0.99)) if len(nz) else 1.0
            v[m] = v[m] / ref
        M[:, j] = v
    return M


def _codes(values: pd.Series, dtype: pd.CategoricalDtype) -> np.ndarray:
    c = pd.Categorical(values.astype(str), dtype=dtype).codes
    if (c < 0).any():
        raise ValueError("pair ids missing from the records")
    return c.astype(np.int32)


def _arrow_codes(ids: pa.ChunkedArray) -> np.ndarray:
    """Integer codes of string ids (Arrow dictionary encoding: no Python strings).
    Only the grouping matters to the stage-2 features, not the code values."""
    return pc.dictionary_encode(ids.combine_chunks()).indices.to_numpy().astype(np.int64)


def group_columns() -> list[str]:
    """Names (in order) of the columns :func:`group_features` returns."""
    one = np.zeros(1, dtype=np.int64)
    return list(group_features(one, one, one + 2, np.zeros(1)).columns)


def _fill_group(M: np.ndarray, q: np.ndarray, s: np.ndarray, src: np.ndarray,
                p: np.ndarray) -> None:
    """Write the group features into the first columns of M (in place)."""
    G = group_features(q, s, src, p)
    for j, c in enumerate(G.columns):
        M[:, j] = G[c].to_numpy()


def _fill_raw(M: np.ndarray, row: int, col0: int, table: pa.Table, raw_cols: list[str],
              keep: np.ndarray) -> int:
    """Write the kept rows of an Arrow table's stage-1 feature columns into
    M[row:, col0:] (numpy casts every type to float32); returns the rows written."""
    n = int(keep.sum())
    for j, c in enumerate(raw_cols):
        M[row:row + n, col0 + j] = table.column(c).to_numpy()[keep]
    return n


def _params(cfg: Config) -> dict:
    rc = cfg.rerank
    return {"objective": "binary", "learning_rate": rc.learning_rate,
            "num_leaves": rc.num_leaves, "min_data_in_leaf": rc.min_data_in_leaf,
            "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1,
            "lambda_l2": 1.0, "seed": cfg.seed, "verbose": -1,
            "num_threads": cfg.scorer.n_threads or max(1, (os.cpu_count() or 4) - 2)}


# (stage 2) where each side's pairs come from: (scores stage, output), (features stage, output)
def _read_pairs(st: StageRun, cfg: Config, raw_cols: list[str], sides: list[str]):
    """Pairs with p1 >= min_p1 of the given sides (val, and hold if
    ``rerank.hold_frac`` > 0) as (keys: q_id, s1_id, y, p, side; M: float32 matrix
    with the stage-1 features in the columns after the group features, which are
    still empty; cols). Only the kept rows are materialised, straight into M.

    val: ``scores_val`` was written by streaming the val rows of the features_train
    parts in order, so the two line up row by row (checked part by part on the
    Arrow ids). hold: ``scores_hold/`` has one part per features_hold part, under
    the same file name (checked the same way)."""
    min_p1 = cfg.rerank.min_p1
    sv = hold_parts = None
    n_keep = 0
    if "val" in sides:
        sv = pq.read_table(st.input_path("scorer", "scores_val"), columns=["q_id", "s1_id", "y", "p"])
        n_keep += int((sv.column("p").to_numpy() >= min_p1).sum())
    if "hold" in sides:
        hold_parts = sorted(st.input_path("scorer", "scores_hold").glob("*.parquet"))
        n_keep += sum(int((pq.read_table(f, columns=["p"]).column("p").to_numpy() >= min_p1).sum())
                      for f in hold_parts)
    gcols = group_columns()
    cols = gcols + raw_cols
    M = np.empty((n_keep, len(cols)), dtype=np.float32)
    row, keys = 0, []
    if sv is not None:
        p, y = sv.column("p").to_numpy(), sv.column("y").to_numpy()
        # candidate-set size vs recall for other filter levels (the organisers rank
        # smaller candidate sets higher): pairs per val query, share of true pairs kept
        n_q = max(pc.count_distinct(sv.column("q_id")).as_py(), 1)
        n_pos = max(float(y.sum()), 1)
        st.metric("filter_tradeoff", {
            t: {"pairs_per_query": round(float((p >= t).sum()) / n_q, 3),
                "true_pairs_kept": round(float(y[p >= t].sum()) / n_pos, 5)}
            for t in (0.0, 0.0003, 0.001, 0.003, 0.01, 0.03)})
        keep_all = p >= min_p1
        sv_q = sv.column("q_id")
        off = 0
        for f in sorted(st.input_path("features", "features_train").glob("*.parquet")):
            d = pq.read_table(f, columns=["q_id", "side"] + raw_cols)
            d = d.filter(pc.equal(d.column("side"), "val"))
            n = d.num_rows
            if n == 0:
                continue
            if off + n > len(p) or not pc.all(pc.equal(d.column("q_id"),
                                                       sv_q.slice(off, n))).as_py():
                raise RuntimeError(f"scores_val and features_train val rows are not aligned ({f.name})")
            row += _fill_raw(M, row, len(gcols), d, raw_cols, keep_all[off:off + n])
            off += n
            del d
        if off != len(p):
            raise RuntimeError("scores_val has rows beyond the val features")
        km = pa.array(keep_all)
        keys.append(pd.DataFrame({"q_id": pc.filter(sv_q, km).to_pandas(),
                                  "s1_id": pc.filter(sv.column("s1_id"), km).to_pandas(),
                                  "y": y[keep_all], "p": p[keep_all], "side": "val"}))
        del sv, sv_q, p, y
    if hold_parts is not None:
        ft_dir = st.input_path("features_hold", "features_hold")
        for f in hold_parts:
            sc = pq.read_table(f, columns=["q_id", "s1_id", "y", "p"])
            ft = pq.read_table(ft_dir / f.name, columns=["q_id"] + raw_cols)
            if ft.num_rows != sc.num_rows or not pc.all(
                    pc.equal(ft.column("q_id"), sc.column("q_id"))).as_py():
                raise RuntimeError(f"scores_hold/features_hold part {f.name} not aligned")
            p = sc.column("p").to_numpy()
            keep = p >= min_p1
            row += _fill_raw(M, row, len(gcols), ft, raw_cols, keep)
            km = pa.array(keep)
            keys.append(pd.DataFrame({"q_id": pc.filter(sc.column("q_id"), km).to_pandas(),
                                      "s1_id": pc.filter(sc.column("s1_id"), km).to_pandas(),
                                      "y": sc.column("y").to_numpy()[keep], "p": p[keep],
                                      "side": "hold"}))
            del sc, ft
    return pd.concat(keys, ignore_index=True), M, cols


def _query_countries(st: StageRun, cfg: Config, sides: list[str]) -> pd.Series:
    """Country of every val (and hold) query, indexed by entity_id."""
    out = [pd.read_parquet(st.input_path("normalize", "records_train"),
                           columns=["entity_id", "country"],
                           filters=[("side", "==", "val")]).set_index("entity_id").country]
    if "hold" in sides:
        out.append(st.read_input("features_hold", "hold_queries",
                                 columns=["entity_id", "country"]).set_index("entity_id").country)
    return pd.concat(out)


def _raw_columns(st: StageRun, wanted: list[str]) -> list[str]:
    """Stage-1 features passed to stage 2; ["all"] = every stage-1 feature."""
    if list(wanted) != ["all"]:
        return list(wanted)
    first = sorted(st.input_path("features", "features_train").glob("*.parquet"))[0]
    return [c for c in pq.ParquetFile(first).schema_arrow.names
            if c not in ("q_id", "s1_id", "y", "side")]


def run_rerank(cfg: Config) -> None:
    rc = cfg.rerank
    with StageRun(cfg, "rerank") as st:
        raw_cols = _raw_columns(st, rc.raw_features)
        if cfg.run_train:
            # every side is loaded whenever it exists (group features and domain scales
            # then see the same pairs whatever stage 2 trains on); train_sides picks the
            # training rows
            sides = ["val", "hold"] if rc.hold_frac > 0 else ["val"]
            if not set(rc.train_sides) <= set(sides) or not rc.train_sides:
                raise ValueError(f"rerank.train_sides {rc.train_sides} must be a non-empty "
                                 f"subset of the loaded sides {sides} (hold needs hold_frac > 0)")
            with timed(st.log, f"load pairs ({'+'.join(sides)})"):
                keys, M, cols = _read_pairs(st, cfg, raw_cols, sides)
                q = pd.factorize(keys.q_id)[0]
                s = pd.factorize(keys.s1_id)[0]
                _fill_group(M, q, s, M[:, cols.index("q_source")], keys.p.to_numpy())
                y = keys.y.to_numpy()
                if rc.domain_norm:
                    cty = _query_countries(st, cfg, sides)
                    groups = pd.factorize(cty.reindex(keys.q_id.astype(str)).fillna(""))[0]
                    domain_normalize(M, cols, groups)
            is_val = (keys.side == "val").to_numpy()
            trainable = keys.side.isin(rc.train_sides).to_numpy()
            st.metric("pairs", {sd: {"kept": int((keys.side == sd).sum()),
                                     "positives": int(y[(keys.side == sd).to_numpy()].sum())}
                                for sd in sides})

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
                                      filters=[("source", "==", 1)]).set_index("entity_id").region
                top_reg = pd.Series(reg.reindex(top.s1_id.astype(str)).fillna("").to_numpy(),
                                    index=top.q.to_numpy())
                del reg
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
            oof = np.zeros(len(M), dtype=np.float64)
            imp = np.zeros(M.shape[1])
            w = None
            if rc.weights_file:  # e.g. adversarial-validation weights (val rows made test-shaped)
                wf = pd.read_parquet(rc.weights_file)
                w = (keys[["q_id", "s1_id"]].astype(str)
                     .merge(wf.astype({"q_id": str, "s1_id": str}), on=["q_id", "s1_id"], how="left")
                     .w.fillna(1.0).to_numpy(dtype=np.float64))
                st.metric("train_weights", {"file": rc.weights_file,
                                            "q": np.quantile(w, [0.01, 0.5, 0.99]).round(3).tolist()})
            with timed(st.log, f"cross-fit {rc.folds} folds (train on {'+'.join(rc.train_sides)})"):
                for k in range(rc.folds):
                    tr, te = (fold != k) & trainable, fold == k
                    # early stopping on 10% of the training groups (never the held-out fold)
                    es = tr & (u * 10 % 1 < 0.1)
                    fit = tr & ~es
                    dtr = lgb.Dataset(M[fit], y[fit], feature_name=cols, free_raw_data=True,
                                      weight=None if w is None else w[fit])
                    des = lgb.Dataset(M[es], y[es], reference=dtr,
                                      weight=None if w is None else w[es])
                    m = lgb.train(_params(cfg), dtr, num_boost_round=rc.num_boost_round,
                                  valid_sets=[des], valid_names=["es"],
                                  callbacks=[lgb.early_stopping(rc.early_stopping, verbose=False)])
                    del dtr, des
                    oof[te] = m.predict(M[te], num_iteration=m.best_iteration)
                    imp += m.feature_importance("gain")
                    m.save_model(str(st.path(f"model_fold{k}.txt")))
                    tv = te & is_val
                    st.log.info("fold %d: best iteration %d, held-out val AUC %.6f", k,
                                m.best_iteration, roc_auc_score(y[tv], oof[tv]))
            st.register("models", st.dir)
            yv, pv, ov = y[is_val], keys.p.to_numpy()[is_val], oof[is_val]
            st.metric("val_auc_p1", round(float(roc_auc_score(yv, pv)), 6))
            st.metric("val_auc_p2_oof", round(float(roc_auc_score(yv, ov)), 6))
            st.metric("val_logloss_p1", round(float(log_loss(yv, np.clip(pv, 1e-7, 1 - 1e-7))), 6))
            st.metric("val_logloss_p2_oof", round(float(log_loss(yv, np.clip(ov, 1e-7, 1 - 1e-7))), 6))
            if "hold" in sides:
                hm = ~is_val
                st.metric("hold_auc_p1", round(float(roc_auc_score(y[hm], keys.p.to_numpy()[hm])), 6))
                st.metric("hold_auc_p2_oof", round(float(roc_auc_score(y[hm], oof[hm])), 6))
            imp_df = pd.DataFrame({"feature": cols, "gain": imp}).sort_values(
                "gain", ascending=False)
            st.write_parquet(imp_df, "importance")
            st.log.info("stage-2 features by gain:\n%s", imp_df.to_string(index=False))
            out = keys[["q_id", "s1_id", "y"]].assign(p=oof.astype(np.float32),
                                                      p1=keys.p.to_numpy())
            for c in SEGMENT_COLS:  # kept for the prior-shift corrections in decide
                if c in cols:
                    out[c] = M[:, cols.index(c)]
            del M
            st.write_parquet(out[is_val].reset_index(drop=True), "scores_val")
            if "hold" in sides:
                st.write_parquet(out[~is_val].reset_index(drop=True), "scores_hold")
            del out, keys

        if cfg.run_test:
            _score_test(st, cfg, raw_cols)


def _score_test(st: StageRun, cfg: Config, raw_cols: list[str]) -> None:
    """Test pairs with p1 >= min_p1, one country at a time (S1 and query groups
    never cross countries, and the stage-1 score parts are written country by
    country), scored by the mean of the fold models. Kept rows are counted first
    and written straight into one float32 matrix per country; ids stay in Arrow."""
    rc = cfg.rerank
    rt = pd.read_parquet(st.input_path("normalize", "records_test"),
                         columns=["entity_id", "country"], filters=[("source", "==", 1)])
    s1_country = rt.set_index("entity_id").country
    del rt
    models = [lgb.Booster(model_file=str(st.path(f"model_fold{k}.txt")))
              for k in range(rc.folds)]
    gcols = group_columns()
    cols = gcols + raw_cols
    if models[0].feature_name() != cols:
        raise RuntimeError("the stage-2 models were trained on other features than "
                           "rerank.raw_features gives now")
    sc_dir = st.input_path("scorer", "scores_test")
    ft_dir = st.input_path("features", "features_test")
    by_country: dict[str, list] = {}
    for f in sorted(sc_dir.glob("*.parquet")):
        first = pq.read_table(f, columns=["s1_id"]).column("s1_id").slice(0, 1).to_pylist()
        if first:
            by_country.setdefault(s1_country[first[0]], []).append(f)
    out_dir = st.path("scores_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.parquet"):
        old.unlink()
    n_total = 0
    for country, files in sorted(by_country.items()):
        with timed(st.log, f"test {country}: {len(files)} parts"):
            n = sum(int((pq.read_table(f, columns=["p"]).column("p").to_numpy()
                         >= rc.min_p1).sum()) for f in files)
            M = np.empty((n, len(cols)), dtype=np.float32)
            qs, ss, ps = [], [], []
            row = 0
            for f in files:
                sc = pq.read_table(f, columns=["q_id", "s1_id", "p"])
                ft = pq.read_table(ft_dir / f.name, columns=["q_id"] + raw_cols)
                if ft.num_rows != sc.num_rows or not pc.all(
                        pc.equal(ft.column("q_id"), sc.column("q_id"))).as_py():
                    raise RuntimeError(f"scores/features part {f.name} not aligned")
                keep = sc.column("p").to_numpy() >= rc.min_p1
                row += _fill_raw(M, row, len(gcols), ft, raw_cols, keep)
                km = pa.array(keep)
                qs.extend(pc.filter(sc.column("q_id"), km).chunks)
                ss.extend(pc.filter(sc.column("s1_id"), km).chunks)
                ps.append(sc.column("p").to_numpy()[keep])
                del sc, ft
            q_ids = pa.chunked_array(qs, type=pa.large_string())
            s_ids = pa.chunked_array(ss, type=pa.large_string())
            p1 = np.concatenate(ps)
            del qs, ss, ps
            _fill_group(M, _arrow_codes(q_ids), _arrow_codes(s_ids),
                        M[:, cols.index("q_source")], p1)
            if rc.domain_norm:  # this matrix is one country of the test split
                domain_normalize(M, cols)
            p2 = np.zeros(n, dtype=np.float64)
            step = 500_000
            for a in range(0, n, step):
                p2[a:a + step] = np.mean([m.predict(M[a:a + step], num_iteration=m.best_iteration)
                                          for m in models], axis=0)
            seg = {c: M[:, cols.index(c)].copy() for c in SEGMENT_COLS if c in cols}
            del M
            out = pa.table({"q_id": q_ids, "s1_id": s_ids, "p": p2.astype(np.float32),
                            "p1": p1, **seg})
            pq.write_table(out, out_dir / f"part-{country}.parquet")
            n_total += n
            st.metric(f"test_pairs_kept.{country}", n)
            del out, q_ids, s_ids, p1, p2, seg
    st.register("scores_test", out_dir, rows=n_total)
