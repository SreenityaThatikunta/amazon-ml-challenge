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


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in KEY_COLS]


def train_model(tr: pd.DataFrame, va: pd.DataFrame, cfg: Config, log) -> lgb.Booster:
    sc = cfg.scorer
    feats = feature_columns(tr)
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


def run_scorer(cfg: Config) -> None:
    with StageRun(cfg, "scorer") as st:
        with timed(st.log, "load train features"):
            df = pd.read_parquet(st.input_path("features", "features_train"))
            for c in CATEGORICAL:
                df[c] = df[c].astype("int32")
        tr, va = df[df.side == "train"], df[df.side == "val"]
        del df
        st.metric("rows", {"train": len(tr), "val": len(va),
                           "train_pos": int(tr.y.sum()), "val_pos": int(va.y.sum())})
        feats = feature_columns(va)
        if cfg.scorer.model_from_run:  # apply an existing model (e.g. France proxy)
            path = st.input_path("scorer", "model", run=cfg.scorer.model_from_run)
            model = lgb.Booster(model_file=str(path))
            missing = set(model.feature_name()) - set(feats)
            if missing:
                raise ValueError(f"features missing for model: {sorted(missing)}")
            feats = model.feature_name()
            p = model.predict(va[feats])
            st.metric("val_auc", round(float(roc_auc_score(va.y, p)), 6))
            st.write_parquet(va[["q_id", "s1_id", "y"]].assign(p=p.astype(np.float32)),
                             "scores_val")
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

        p = model.predict(va[feats], num_iteration=model.best_iteration)
        st.metric("best_iteration", model.best_iteration)
        st.metric("val_auc", round(float(roc_auc_score(va.y, p)), 6))
        st.metric("val_logloss", round(float(log_loss(va.y, p)), 6))
        st.write_parquet(va[["q_id", "s1_id", "y"]].assign(p=p.astype(np.float32)), "scores_val")

        if cfg.run_test:
            out_dir = st.path("scores_test")
            out_dir.mkdir(parents=True, exist_ok=True)
            for old in out_dir.glob("*.parquet"):
                old.unlink()
            n = 0
            with timed(st.log, "score test"):
                src = st.input_path("features", "features_test")
                for i, f in enumerate(sorted(src.glob("*.parquet"))):
                    t = pd.read_parquet(f)
                    for c in CATEGORICAL:
                        t[c] = t[c].astype("int32")
                    pt = model.predict(t[feats], num_iteration=model.best_iteration)
                    t[["q_id", "s1_id"]].assign(p=pt.astype(np.float32)).to_parquet(
                        out_dir / f"part-{i:05d}.parquet", index=False)
                    n += len(t)
            st.register("scores_test", out_dir, rows=n)
