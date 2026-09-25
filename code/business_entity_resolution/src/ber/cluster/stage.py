"""``ber rescore`` (M8, tasks T8.1–T8.3): second-stage scorer with cluster context.

Stage-1 scores each (query, S1) pair on its own. But an S1's true copies are
noisy versions of *each other*, and the S2/S3 cap (≤ 5 / ≤ 6 per S1) and the
one-owner rule tie pairs together. This stage rescores each query's top-K
stage-1 candidates with features about the S1's confident cluster: the other
queries assigned to it with p ≥ ``conf``, excluding the query itself.

  * pair: stage-1 p, rank and margin within the query, the query's best two p;
  * cluster of the S1 (without the query): size, sum / max of p, same-source
    size (cap), how many assigned members score above this pair;
  * the query vs the cluster members: best name ratio, best address
    token-set ratio, exact duplicate (same core name and address);
  * a few stage-1 features joined back (name equality, house-number relation,
    address similarity, legal form agreement).

Only S1 of countries seen in training are rescored: on the France proxy the
US/India-trained rescore lowered F0.5 (0.9706 -> 0.9662, experiments/README.md
E9), so unseen-country pairs keep their stage-1 p (from the scorer's
unseen-country model).

Training uses the **val side only** (every val query has honest stage-1 scores),
cross-fitted in ``folds`` folds by the query's best S1, so the val scores stay
out-of-sample; the test model is trained on all val pairs. Outputs mirror the
scorer (``scores_val``, ``scores_test/``), so ``decide.scores_stage=rescore``
picks them up.
"""

from __future__ import annotations

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.metrics import log_loss, roc_auc_score

from ..artifacts import StageRun
from ..config import Config
from ..eval.split import stable_unit
from ..log import timed

STAGE1_FEATS = ["n_core_eq", "n_ratio", "hn_rel", "a_tsr", "l_quality", "cos_combo"]  # default
MAX_MEMBERS = 8  # cluster members compared per S1 (highest p first)


# ---------------------------------------------------------------- features


def top_k(scores: pd.DataFrame, k: int) -> pd.DataFrame:
    """Each query's k best candidates, with rank, margin over the best other
    candidate, and the query's best / second-best p."""
    s = scores.sort_values(["q_id", "p"], ascending=[True, False], kind="stable")
    s["rank"] = s.groupby("q_id", sort=False).cumcount().astype(np.int16)
    first = s.groupby("q_id", sort=False).p.transform("first")
    second = s.p.where(s["rank"] == 1).groupby(s.q_id, sort=False).transform("max").fillna(0.0)
    s["q_p1"], s["q_p2"] = first.astype(np.float32), second.astype(np.float32)
    s["p_margin"] = np.where(s["rank"] == 0, s.p - s.q_p2, s.p - s.q_p1).astype(np.float32)
    return s[s["rank"] < k].reset_index(drop=True)


def cluster_features(pairs: pd.DataFrame, conf: float) -> pd.DataFrame:
    """Cluster context per pair. ``pairs``: top-k pairs with q_id, s1_id, p, rank.
    The query's own contribution is removed from its S1's cluster."""
    src = pairs.q_id.str[1].astype(np.int8)            # "S2-..." -> 2
    own = pairs[pairs["rank"] == 0][["q_id", "s1_id", "p"]].copy()
    own["src"] = own.q_id.str[1].astype(np.int8)
    own["conf"] = own.p >= conf
    g = own.groupby("s1_id")
    stats = pd.DataFrame({
        "s_n_assigned": g.size(),
        "s_n_conf": g.conf.sum(),
        "s_sum_conf": own.p.where(own.conf, 0.0).groupby(own.s1_id).sum(),
        "s_n_conf_s2": (own.conf & (own.src == 2)).groupby(own.s1_id).sum(),
        "s_n_conf_s3": (own.conf & (own.src == 3)).groupby(own.s1_id).sum(),
        "s_max_conf": own.p.where(own.conf, 0.0).groupby(own.s1_id).max(),
    })
    f = stats.reindex(pairs.s1_id.to_numpy()).fillna(0).reset_index(drop=True)
    self_ = ((pairs["rank"] == 0) & (pairs.p >= conf)).to_numpy()
    selfp = np.where(self_, pairs.p.to_numpy(), 0.0)
    out = pd.DataFrame(index=pairs.index)
    out["c_n"] = f.s_n_conf.to_numpy() - self_
    out["c_sum"] = f.s_sum_conf.to_numpy() - selfp
    same = np.where(src == 2, f.s_n_conf_s2.to_numpy(), f.s_n_conf_s3.to_numpy())
    out["c_same_src"] = same - self_
    out["c_cap_room"] = np.where(src == 2, 5, 6) - out.c_same_src
    out["s_n_assigned"] = f.s_n_assigned.to_numpy() - (pairs["rank"] == 0).to_numpy()
    # how many queries assigned to this S1 score strictly above this pair: sort
    # members and pairs together by (S1, -p), pairs first on ties, and count the
    # members seen so far within the S1
    codes, uniq = pd.factorize(pd.concat([own.s1_id, pairs.s1_id], ignore_index=True))
    n_own = len(own)
    is_member = np.r_[np.ones(n_own, dtype=np.int32), np.zeros(len(pairs), dtype=np.int32)]
    pp = np.r_[own.p.to_numpy(), pairs.p.to_numpy()]
    order = np.lexsort((is_member, -pp, codes))
    c_sorted, m_sorted = codes[order], is_member[order]
    cum = np.cumsum(m_sorted)
    start = np.r_[0, np.flatnonzero(c_sorted[1:] != c_sorted[:-1]) + 1]
    grp_start_cum = np.repeat(np.r_[0, cum][start], np.diff(np.r_[start, len(order)]))
    before = cum - m_sorted - grp_start_cum
    above = np.empty(len(order), dtype=np.int32)
    above[order] = before
    out["s_n_above"] = above[n_own:].astype(np.int16)
    return out, own[own.conf][["q_id", "s1_id", "p"]]


def member_similarity(pairs: pd.DataFrame, members: pd.DataFrame, rec: pd.DataFrame,
                      chunk: int = 2_000_000) -> pd.DataFrame:
    """Best similarity of each pair's query to the other confident members of
    its S1 (top ``MAX_MEMBERS`` by p). ``rec``: name_core, addr_norm by entity_id."""
    m = members.sort_values(["s1_id", "p"], ascending=[True, False])
    m = m[m.groupby("s1_id").cumcount() < MAX_MEMBERS][["s1_id", "q_id"]]
    m = m.rename(columns={"q_id": "m_id"})
    out = pd.DataFrame({"m_name_max": -1.0, "m_addr_max": -1.0, "m_dup": 0, "m_n": 0},
                       index=pairs.index)
    res = []
    keys = pairs[["q_id", "s1_id"]].reset_index()
    for i in range(0, len(keys), chunk // 4):
        c = keys.iloc[i:i + chunk // 4].merge(m, on="s1_id")
        c = c[c.q_id != c.m_id]
        if c.empty:
            continue
        qn, mn = rec.name_core.reindex(c.q_id).tolist(), rec.name_core.reindex(c.m_id).tolist()
        qa, ma = rec.addr_norm.reindex(c.q_id).tolist(), rec.addr_norm.reindex(c.m_id).tolist()
        r = pd.DataFrame({
            "index": c["index"].to_numpy(),
            "m_name": process.cpdist(qn, mn, scorer=fuzz.ratio, workers=-1),
            "m_addr": process.cpdist(qa, ma, scorer=fuzz.token_set_ratio, workers=-1),
            "m_dup": [int(a == b and c_ == d and a != "") for a, b, c_, d in zip(qn, mn, qa, ma)]})
        res.append(r)
    if not res:
        return out
    r = pd.concat(res).groupby("index").agg(m_name_max=("m_name", "max"),
                                           m_addr_max=("m_addr", "max"),
                                           m_dup=("m_dup", "max"), m_n=("m_name", "size"))
    out.loc[r.index, r.columns] = r.to_numpy()
    return out


def build(pairs: pd.DataFrame, rec: pd.DataFrame, conf: float,
          prev: np.ndarray | None = None, stage1_feats: list[str] = STAGE1_FEATS) -> pd.DataFrame:
    """Rescore features. ``prev``: the previous round's rescored p (rounds > 1);
    clusters, owners and ranks are then rebuilt from it, and it is a feature."""
    base = pairs[["p", "rank", "p_margin", "q_p1", "q_p2"] +
                 [c for c in stage1_feats if c in pairs]].copy()
    base["q_source"] = pairs.q_id.str[1].astype(np.int8)
    cp = pairs
    if prev is not None:
        tmp = pairs[["q_id", "s1_id"]].assign(p=prev, _i=np.arange(len(pairs)))
        cp = top_k(tmp, len(tmp)).sort_values("_i").set_index(pairs.index)
        base["p_prev"], base["p_prev_margin"] = cp.p, cp.p_margin
        base["rank_prev"] = cp["rank"]
    cf, members = cluster_features(cp, conf)
    ms = member_similarity(cp, members, rec)
    return pd.concat([base, cf, ms], axis=1).astype(np.float32)


# ---------------------------------------------------------------- stage


def _params(cfg: Config) -> dict:
    return {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63,
            "min_data_in_leaf": 200, "feature_fraction": 0.9, "bagging_fraction": 0.8,
            "bagging_freq": 1, "lambda_l2": 1.0, "seed": cfg.seed, "verbose": -1,
            "num_threads": max(1, (os.cpu_count() or 4) - 2)}


def _fit(X, y, cfg: Config, rounds: int | None = None):
    """Fit with early stopping on a 10% query-free slice, or a fixed number of rounds."""
    if rounds:
        return lgb.train(_params(cfg), lgb.Dataset(X, y), num_boost_round=rounds)
    u = np.random.default_rng(cfg.seed).random(len(X)) < 0.1
    dtr, dva = lgb.Dataset(X[~u], y[~u]), lgb.Dataset(X[u], y[u])
    return lgb.train(_params(cfg), dtr, num_boost_round=2000, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def load_top_k(st: StageRun, split: str, k: int, stage1_feats: list[str]) -> pd.DataFrame:
    """Top-k pairs per query with stage-1 p and the given stage-1 features. Scores and
    features are read in lockstep: the scorer writes its rows in feature-part
    order (val: the val-side rows of ``features_train``; test: part i of
    ``features_test`` -> part i of ``scores_test``), so no join is needed.
    Test rows keep ``_part`` / ``_row`` (their position in the features parts), so
    stage-1 features can be fetched part by part later instead of held in memory."""
    fdir = st.input_path("features", f"features_{split}")
    cols = ["q_id", "s1_id"] + stage1_feats + (["y", "side"] if split == "train" else [])
    out = []
    if split == "train":
        sv = st.read_input("scorer", "scores_val")
        pos = 0
        for f in sorted(fdir.glob("*.parquet")):
            d = pd.read_parquet(f, columns=cols)
            d = d[d.side == "val"].drop(columns="side").reset_index(drop=True)
            if d.empty:
                continue
            s = sv.iloc[pos:pos + len(d)].reset_index(drop=True)
            pos += len(d)
            assert (s.q_id.astype(str).to_numpy() == d.q_id.astype(str).to_numpy()).all(), \
                "scores_val not aligned with features_train val rows"
            out.append(top_k(d.assign(p=s.p.to_numpy()).astype({"q_id": str, "s1_id": str}), k))
    else:
        sdir = st.input_path("scorer", "scores_test")
        for ff, sf in zip(sorted(fdir.glob("*.parquet")), sorted(sdir.glob("*.parquet"))):
            d = pd.read_parquet(ff, columns=cols)
            s = pd.read_parquet(sf, columns=["q_id", "p"])
            assert len(s) == len(d) and s.q_id.iloc[0] == d.q_id.iloc[0], f"{sf.name} not aligned"
            d = d.assign(p=s.p.to_numpy(), _part=np.int16(len(out)),
                         _row=np.arange(len(d), dtype=np.int32))
            out.append(top_k(d.astype({"q_id": str, "s1_id": str}), k))
    return pd.concat(out, ignore_index=True)


def predict_parts(mdl, pairs: pd.DataFrame, ctx: pd.DataFrame, fparts: list,
                  mask: np.ndarray) -> np.ndarray:
    """Score ``pairs`` (with ``_part`` / ``_row`` positions in ``fparts``) part by
    part: the model's stage-1 features are read from each part and joined to the
    global ``ctx`` rows. Rows outside ``mask`` keep their stage-1 p."""
    order = np.argsort(pairs._part.to_numpy(), kind="stable")
    bounds = np.searchsorted(pairs._part.to_numpy()[order], np.arange(len(fparts) + 1))
    need = [c for c in mdl.feature_name() if c not in ctx.columns]
    rows = pairs._row.to_numpy()
    p = pairs.p.to_numpy(dtype=np.float32).copy()
    for i, ff in enumerate(fparts):
        sel = order[bounds[i]:bounds[i + 1]]
        sel = sel[mask[sel]]
        if not len(sel):
            continue
        f1 = pd.read_parquet(ff, columns=need).iloc[rows[sel]]
        X = pd.concat([ctx.iloc[sel].reset_index(drop=True),
                       f1.reset_index(drop=True).astype(np.float32)], axis=1)
        p[sel] = mdl.predict(X[mdl.feature_name()]).astype(np.float32)
    return p


def _records(st: StageRun, split: str, ids: set[str]) -> pd.DataFrame:
    r = pd.read_parquet(st.input_path("normalize", f"records_{split}"),
                        columns=["entity_id", "source", "name_core", "addr_norm"],
                        filters=[("source", "!=", 1)])
    return r[r.entity_id.isin(ids)].set_index("entity_id")


def run_rescore(cfg: Config) -> None:
    rc = cfg.rescore
    feats1 = rc.stage1_feats or STAGE1_FEATS
    with StageRun(cfg, "rescore") as st:
        if feats1 == ["*"]:  # every stage-1 feature column
            import pyarrow.parquet as pq
            first = next(st.input_path("features", "features_train").glob("*.parquet"))
            feats1 = [c for c in pq.read_schema(first).names
                      if c not in ("q_id", "s1_id", "y", "side")]
        st.metric("stage1_feats", len(feats1))
        with timed(st.log, "val: top-k"):
            pv = load_top_k(st, "train", rc.k, feats1)
            rec = _records(st, "train", set(pv.q_id))
            yv = pv.y.to_numpy()
        st.metric("val_pairs", len(pv))
        st.metric("val_auc_stage1", round(float(roc_auc_score(yv, pv.p)), 6))
        st.metric("val_logloss_stage1", round(float(log_loss(yv, pv.p.clip(1e-7, 1 - 1e-7))), 6))
        if rc.model_from_run:  # apply an existing rescore model chain (e.g. France proxy)
            prev = None
            for r in range(rc.rounds):
                mdl = lgb.Booster(model_file=str(st.input_path("rescore", f"model_r{r}",
                                                               run=rc.model_from_run)))
                prev = mdl.predict(build(pv, rec, rc.conf, prev, feats1)[mdl.feature_name()]
                                   ).astype(np.float32)
            st.metric("val_auc_rescore", round(float(roc_auc_score(yv, prev)), 6))
            st.write_parquet(pv[["q_id", "s1_id", "y"]].assign(p=prev), "scores_val")
            return

        best = pv[pv["rank"] == 0].set_index("q_id").s1_id
        fold = (stable_unit(best.reindex(pv.q_id).reset_index(drop=True), cfg.seed + 6)
                * rc.folds).astype(int)
        models, prev = [], None
        for r in range(rc.rounds):
            with timed(st.log, f"round {r}: features + {rc.folds}-fold cross-fit"):
                Xv = build(pv, rec, rc.conf, prev, feats1)
                p_cf = np.empty(len(pv), dtype=np.float32)
                iters = []
                for k in range(rc.folds):
                    m = fold == k
                    mdl = _fit(Xv[~m], yv[~m], cfg)
                    iters.append(mdl.best_iteration or mdl.current_iteration())
                    p_cf[m] = mdl.predict(Xv[m], num_iteration=mdl.best_iteration)
                final = _fit(Xv, yv, cfg, rounds=int(np.mean(iters)))
            st.metric(f"val_auc_rescore_r{r}", round(float(roc_auc_score(yv, p_cf)), 6))
            st.metric(f"val_logloss_rescore_r{r}",
                      round(float(log_loss(yv, p_cf.clip(1e-7, 1 - 1e-7))), 6))
            final.save_model(str(st.path(f"model_r{r}.txt")))
            st.register(f"model_r{r}", st.path(f"model_r{r}.txt"))
            imp = pd.DataFrame({"feature": list(Xv.columns),
                                "gain": final.feature_importance("gain")}
                               ).sort_values("gain", ascending=False)
            st.log.info("round %d rescore features by gain:\n%s", r, imp.to_string(index=False))
            models.append(final)
            prev = p_cf  # out-of-fold, so round r+1 trains on honest round-r scores
        st.metric("val_auc_rescore", round(float(roc_auc_score(yv, prev)), 6))
        st.metric("val_logloss_rescore", round(float(log_loss(yv, prev.clip(1e-7, 1 - 1e-7))), 6))
        st.write_parquet(imp, "importance")
        st.write_parquet(pv[["q_id", "s1_id", "y"]].assign(p=prev), "scores_val")
        del Xv, pv, rec

        if cfg.run_test:
            with timed(st.log, "test: top-k + features + score"):
                # ~30M top-k pairs: cluster/member features are built globally (few
                # columns); stage-1 features are fetched and scored part by part
                pt = load_top_k(st, "test", rc.k, [])
                rec = _records(st, "test", set(pt.q_id))
                info = json.loads(st.input_path("scorer", "model_info").read_text())
                s1c = pd.read_parquet(st.input_path("normalize", "records_test"),
                                      columns=["entity_id", "country"],
                                      filters=[("source", "==", 1)]).set_index("entity_id").country
                seen = s1c.reindex(pt.s1_id.to_numpy()).isin(info["train_countries"]).to_numpy()
                fparts = sorted(st.input_path("features", "features_test").glob("*.parquet"))
                prev = None
                for mdl in models:  # unseen-country pairs keep their stage-1 p every round
                    ctx = build(pt, rec, rc.conf, prev, [])
                    prev = predict_parts(mdl, pt, ctx, fparts, seen)
                    del ctx
                pt["p"] = prev
                st.metric("test_pairs_rescored", int(seen.sum()))
                st.metric("test_pairs_kept_stage1", int((~seen).sum()))
                out_dir = st.path("scores_test")
                out_dir.mkdir(parents=True, exist_ok=True)
                for old in out_dir.glob("*.parquet"):
                    old.unlink()
                pt[["q_id", "s1_id", "p"]].to_parquet(out_dir / "part-00000.parquet", index=False)
                st.register("scores_test", out_dir, rows=len(pt))
