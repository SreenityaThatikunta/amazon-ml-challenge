"""``ber scorer`` (task T3.4): LightGBM pair scorer.

Trains on train-side candidate pairs, early-stops on val-side pairs, then
scores the val pairs (``scores_val.parquet``) and every test pair
(``scores_test/``). Saves the model, feature list and feature importance.

With ``oof_folds > 1`` it also writes out-of-fold scores for the train-side
pairs (``scores_train_oof.parquet``; folds by query). The LLM stage needs
honest probabilities for its training prompts: in-sample predictions would be
overconfident, so routing would behave differently on train than on test.
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
from ..log import timed

KEY_COLS = ["q_id", "s1_id", "y", "side"]
CATEGORICAL = ["hn_rel", "q_source"]


def feature_columns(df: pd.DataFrame, drop: list[str] | tuple[str, ...] = ()) -> list[str]:
    """Feature columns of a features frame; ``drop`` are feature names or name
    prefixes ending in ``*`` (``scorer.drop_features``, for ablations)."""
    exact = {d for d in drop if not d.endswith("*")}
    prefixes = tuple(d[:-1] for d in drop if d.endswith("*"))
    return [c for c in df.columns if c not in KEY_COLS and c not in exact
            and not (prefixes and c.startswith(prefixes))]


def train_model(tr: pd.DataFrame, va: pd.DataFrame, cfg: Config, log) -> lgb.Booster:
    sc = cfg.scorer
    feats = feature_columns(tr, sc.drop_features)
    params = {
        "objective": "binary", "learning_rate": sc.learning_rate, "num_leaves": sc.num_leaves,
        "min_data_in_leaf": sc.min_data_in_leaf, "feature_fraction": sc.feature_fraction,
        "bagging_fraction": sc.bagging_fraction, "bagging_freq": 1, "lambda_l2": sc.lambda_l2,
        "seed": cfg.seed, "verbose": -1,
        "num_threads": sc.n_threads or max(1, (os.cpu_count() or 4) - 2),
    }
    dtr = lgb.Dataset(tr[feats], tr.y, categorical_feature=CATEGORICAL, free_raw_data=True)
    dva = lgb.Dataset(va[feats], va.y, categorical_feature=CATEGORICAL, reference=dtr)
    return lgb.train(params, dtr, num_boost_round=sc.num_boost_round, valid_sets=[dva],
                     valid_names=["val"],
                     callbacks=[lgb.early_stopping(sc.early_stopping, verbose=False),
                                lgb.log_evaluation(100)])


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """float32 features (halves memory), int32 categoricals, string ids."""
    num = [c for c in df.columns if c not in KEY_COLS and c not in CATEGORICAL]
    df[num] = df[num].astype(np.float32)
    for c in CATEGORICAL:
        df[c] = df[c].astype("int32")
    return df


def load_sampled(path, cfg: Config, log) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stream feature parts; keep a query-level sample of train-side rows and of
    val-side rows (the latter for early stopping). Memory stays bounded."""
    sc = cfg.scorer
    tr_parts, es_parts = [], []
    for f in sorted(path.glob("*.parquet")):
        d = pd.read_parquet(f)
        d["q_id"] = d.q_id.astype(str)
        u = stable_unit(d.q_id, cfg.seed + 4)
        t = d[(d.side == "train").to_numpy() & (u < sc.train_query_frac)]
        e = d[(d.side == "val").to_numpy() & (u < sc.es_query_frac)]
        if len(t):
            tr_parts.append(_prep(t.copy()))
        if len(e):
            es_parts.append(_prep(e.copy()))
    tr = pd.concat(tr_parts, ignore_index=True) if tr_parts else pd.DataFrame()
    es = pd.concat(es_parts, ignore_index=True) if es_parts else pd.DataFrame()
    log.info("sampled %d train rows, %d early-stopping rows", len(tr), len(es))
    return tr, es


def predict_side(model: lgb.Booster, path, side: str, feats: list[str]) -> pd.DataFrame:
    """Scores for every row of one side, streaming the parts."""
    out = []
    for f in sorted(path.glob("*.parquet")):
        d = pd.read_parquet(f)
        d = d[d.side == side]
        if d.empty:
            continue
        d = _prep(d)
        out.append(d[["q_id", "s1_id", "y"]].assign(
            p=model.predict(d[feats], num_iteration=model.best_iteration).astype(np.float32)))
    return pd.concat(out, ignore_index=True)


def score_test(st: StageRun, model: lgb.Booster, feats: list[str]) -> None:
    """Score every test pair, one feature part at a time."""
    out_dir = st.path("scores_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.parquet"):
        old.unlink()
    n = 0
    with timed(st.log, "score test"):
        src = st.input_path("features", "features_test")
        for i, f in enumerate(sorted(src.glob("*.parquet"))):
            t = _prep(pd.read_parquet(f))
            pt = model.predict(t[feats], num_iteration=model.best_iteration)
            t[["q_id", "s1_id"]].assign(p=pt.astype(np.float32)).to_parquet(
                out_dir / f"part-{i:05d}.parquet", index=False)
            n += len(t)
    st.register("scores_test", out_dir, rows=n)


def run_scorer(cfg: Config) -> None:
    with StageRun(cfg, "scorer") as st:
        if not cfg.run_train:  # test-only pass with the model of the train pass
            model = lgb.Booster(model_file=str(st.path("model.txt")))
            score_test(st, model, model.feature_name())
            return
        feat_dir = st.input_path("features", "features_train")
        with timed(st.log, "load sampled train features"):
            tr, va = load_sampled(feat_dir, cfg, st.log)
        st.metric("rows", {"train": len(tr), "val": len(va),
                           "train_pos": int(tr.y.sum()) if len(tr) else 0,
                           "val_pos": int(va.y.sum()) if len(va) else 0})
        feats = feature_columns(tr if len(tr) else va, cfg.scorer.drop_features)
        if cfg.scorer.model_from_run:  # apply an existing model (e.g. France proxy)
            path = st.input_path("scorer", "model", run=cfg.scorer.model_from_run)
            model = lgb.Booster(model_file=str(path))
            feats = feature_columns(pd.read_parquet(next(feat_dir.glob("*.parquet"))).head(1))
            missing = set(model.feature_name()) - set(feats)
            if missing:
                raise ValueError(f"features missing for model: {sorted(missing)}")
            feats = model.feature_name()
            sv = predict_side(model, feat_dir, "val", feats)
            st.metric("val_auc", round(float(roc_auc_score(sv.y, sv.p)), 6))
            st.write_parquet(sv, "scores_val")
            return
        with timed(st.log, "train LightGBM"):
            model = train_model(tr, va, cfg, st.log)
        model.save_model(str(st.path("model.txt")))
        st.register("model", st.path("model.txt"))
        imp = pd.DataFrame({"feature": feats,
                            "gain": model.feature_importance("gain"),
                            "split": model.feature_importance("split")}
                           ).sort_values("gain", ascending=False)
        st.write_parquet(imp, "importance")
        st.log.info("top features by gain:\n%s", imp.head(25).to_string(index=False))

        if cfg.scorer.oof_folds > 1:
            with timed(st.log, f"out-of-fold scores ({cfg.scorer.oof_folds} folds)"):
                fold = (stable_unit(tr.q_id.astype(str), cfg.seed + 2)
                        * cfg.scorer.oof_folds).astype(int)
                oof = np.empty(len(tr), dtype=np.float32)
                for k in range(cfg.scorer.oof_folds):
                    m = fold == k
                    mk = train_model(tr[~m], va, cfg, st.log)
                    oof[m] = mk.predict(tr.loc[m, feats], num_iteration=mk.best_iteration)
                st.write_parquet(tr[["q_id", "s1_id", "y"]].assign(p=oof), "scores_train_oof")
                st.metric("train_oof_auc", round(float(roc_auc_score(tr.y, oof)), 6))

        del va
        with timed(st.log, "score all val pairs"):
            sv = predict_side(model, feat_dir, "val", feats)
        st.metric("best_iteration", model.best_iteration)
        st.metric("val_auc", round(float(roc_auc_score(sv.y, sv.p)), 6))
        st.metric("val_logloss", round(float(log_loss(sv.y, sv.p)), 6))
        st.write_parquet(sv, "scores_val")

        if cfg.run_test:
            score_test(st, model, feats)
