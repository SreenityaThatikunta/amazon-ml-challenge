"""``ber scorer`` (task T3.4): LightGBM pair scorer.

Trains on train-side candidate pairs, early-stops on val-side pairs, then
scores the val pairs (``scores_val.parquet``) and every test pair
(``scores_test/``). Saves the model, feature list and feature importance.

With ``oof_folds > 1`` it also writes out-of-fold scores for the train-side
pairs (``scores_train_oof.parquet``; folds by query). The LLM stage needs
honest probabilities for its training prompts: in-sample predictions would be
overconfident, so routing would behave differently on train than on test.

``scorer.extra_train`` adds every labelled pair of other runs' ``features_train``
to training only (never to early stopping or val scores).

Unseen countries (T7, France): ``scorer.unseen_extra_train`` trains a second
model on a ``unseen_base_frac`` share of the same real pairs plus those runs'
labelled pairs (synthetic French pairs, pseudo-labels). Pairs whose S1 country
never occurs in the training data are scored with it; every other pair with the
main model. Features stay country-agnostic: the country only picks the model.
On dev, synthetic French training lifted the France proxy 0.880 -> 0.971 but
cost US/India 0.0014 (experiments/README.md, E6), hence two models.
"""

from __future__ import annotations

import json
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


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """float32 features (halves memory), int32 categoricals, string ids."""
    num = [c for c in df.columns if c not in KEY_COLS and c not in CATEGORICAL]
    df[num] = df[num].astype(np.float32)
    for c in CATEGORICAL:
        df[c] = df[c].astype("int32")
    return df


def load_sampled(path, cfg: Config, log, extra_runs: list[str] | None = None,
                 train_frac: float | None = None, with_es: bool = True
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stream feature parts; keep a query-level sample of train-side rows and of
    val-side rows (the latter for early stopping), plus every labelled pair of
    ``extra_runs`` (default ``scorer.extra_train``). Memory stays bounded."""
    sc = cfg.scorer
    extra_runs = sc.extra_train if extra_runs is None else extra_runs
    train_frac = sc.train_query_frac if train_frac is None else train_frac
    tr_parts, es_parts = [], []
    for f in sorted(path.glob("*.parquet")):
        d = pd.read_parquet(f)
        d["q_id"] = d.q_id.astype(str)
        u = stable_unit(d.q_id, cfg.seed + 4)
        t = d[(d.side == "train").to_numpy() & (u < train_frac)]
        e = d[(d.side == "val").to_numpy() & (u < sc.es_query_frac)] if with_es else d.iloc[:0]
        if len(t):
            tr_parts.append(_prep(t.copy()))
        if len(e):
            es_parts.append(_prep(e.copy()))
    for run in extra_runs:
        n0 = sum(map(len, tr_parts))
        extra = cfg.resolve(cfg.paths.artifacts_dir) / run / "features" / "features_train"
        if not extra.is_dir():
            raise FileNotFoundError(f"scorer.extra_train: no features_train in run {run!r}")
        for f in sorted(extra.glob("*.parquet")):
            d = pd.read_parquet(f)
            d = d[stable_unit(d.q_id.astype(str), cfg.seed + 5) < sc.extra_train_frac]
            d["q_id"] = run + ":" + d.q_id.astype(str)  # keep query ids distinct across runs
            d["side"] = "train"
            tr_parts.append(_prep(d))
        log.info("extra train rows from run %s: %d", run, sum(map(len, tr_parts)) - n0)
    tr = pd.concat(tr_parts, ignore_index=True) if tr_parts else pd.DataFrame()
    es = pd.concat(es_parts, ignore_index=True) if es_parts else pd.DataFrame()
    log.info("sampled %d train rows, %d early-stopping rows", len(tr), len(es))
    return tr, es


class Scorer:
    """Main model plus an optional unseen-country model; routes each pair by the
    country of its S1 (``s1_country``: entity_id -> country)."""

    def __init__(self, main: lgb.Booster, unseen: lgb.Booster | None,
                 train_countries: list[str], s1_country: pd.Series):
        self.main, self.unseen = main, unseen
        self.train_countries = set(train_countries)
        self.s1_country = s1_country
        self.feats = main.feature_name()

    def unseen_mask(self, s1_ids: pd.Series) -> np.ndarray:
        if self.unseen is None:
            return np.zeros(len(s1_ids), dtype=bool)
        c = self.s1_country.reindex(s1_ids.astype(str).to_numpy())
        return (~c.isin(self.train_countries)).to_numpy()

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        p = np.empty(len(d), dtype=np.float32)
        m = self.unseen_mask(d.s1_id)
        for model, rows in ((self.main, ~m), (self.unseen, m)):
            if rows.any():
                p[rows] = model.predict(d.loc[rows, model.feature_name()],
                                        num_iteration=model.best_iteration)
        return p


def predict_side(scorer: Scorer, path, side: str) -> pd.DataFrame:
    """Scores for every row of one side, streaming the parts."""
    out = []
    for f in sorted(path.glob("*.parquet")):
        d = pd.read_parquet(f)
        d = d[d.side == side]
        if d.empty:
            continue
        d = _prep(d).reset_index(drop=True)
        out.append(d[["q_id", "s1_id", "y"]].assign(p=scorer.predict(d)))
    return pd.concat(out, ignore_index=True)


def s1_countries(st: StageRun, split: str) -> pd.Series:
    r = pd.read_parquet(st.input_path("normalize", f"records_{split}"),
                        columns=["entity_id", "country"], filters=[("source", "==", 1)])
    return r.set_index("entity_id").country


def load_scorer(st: StageRun, run: str, s1_country: pd.Series) -> Scorer:
    """The scorer saved by ``run`` (main + optional unseen-country model)."""
    main = lgb.Booster(model_file=str(st.input_path("scorer", "model", run=run)))
    info = json.loads(st.input_path("scorer", "model_info", run=run).read_text())
    unseen = (lgb.Booster(model_file=str(st.input_path("scorer", "model_unseen", run=run)))
              if info.get("unseen_model") else None)
    return Scorer(main, unseen, info["train_countries"], s1_country)


def run_scorer(cfg: Config) -> None:
    with StageRun(cfg, "scorer") as st:
        feat_dir = st.input_path("features", "features_train")
        with timed(st.log, "load sampled train features"):
            tr, va = load_sampled(feat_dir, cfg, st.log)
        st.metric("rows", {"train": len(tr), "val": len(va),
                           "train_pos": int(tr.y.sum()) if len(tr) else 0,
                           "val_pos": int(va.y.sum()) if len(va) else 0})
        feats = feature_columns(tr if len(tr) else va)
        s1c_train = s1_countries(st, "train")
        if cfg.scorer.model_from_run:  # apply an existing model (e.g. France proxy)
            scorer = load_scorer(st, cfg.scorer.model_from_run, s1c_train)
            have = feature_columns(pd.read_parquet(next(feat_dir.glob("*.parquet"))).head(1))
            for model in (scorer.main, scorer.unseen):
                missing = set(model.feature_name()) - set(have) if model else set()
                if missing:
                    raise ValueError(f"features missing for model: {sorted(missing)}")
            sv = predict_side(scorer, feat_dir, "val")
            st.metric("val_unseen_country_share",
                      round(float(scorer.unseen_mask(sv.s1_id).mean()), 5))
            st.metric("val_auc", round(float(roc_auc_score(sv.y, sv.p)), 6))
            st.write_parquet(sv, "scores_val")
            return
        train_countries = sorted(s1c_train.unique())
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

        unseen = None
        if cfg.scorer.unseen_extra_train:
            del tr
            with timed(st.log, "train unseen-country LightGBM"):
                tr_u, _ = load_sampled(feat_dir, cfg, st.log, cfg.scorer.unseen_extra_train,
                                       cfg.scorer.train_query_frac * cfg.scorer.unseen_base_frac,
                                       with_es=False)
                st.metric("rows_unseen_model", {"train": len(tr_u), "pos": int(tr_u.y.sum())})
                unseen = train_model(tr_u, va, cfg, st.log)
                del tr_u
            unseen.save_model(str(st.path("model_unseen.txt")))
            st.register("model_unseen", st.path("model_unseen.txt"))
            st.metric("best_iteration_unseen", unseen.best_iteration)
        st.write_json({"train_countries": train_countries, "unseen_model": unseen is not None,
                       "unseen_extra_train": cfg.scorer.unseen_extra_train}, "model_info")
        st.metric("train_countries", train_countries)

        del va
        scorer = Scorer(model, unseen, train_countries, s1c_train)
        with timed(st.log, "score all val pairs"):
            sv = predict_side(scorer, feat_dir, "val")
        st.metric("best_iteration", model.best_iteration)
        st.metric("val_auc", round(float(roc_auc_score(sv.y, sv.p)), 6))
        st.metric("val_logloss", round(float(log_loss(sv.y, sv.p)), 6))
        st.write_parquet(sv, "scores_val")

        if cfg.run_test:
            out_dir = st.path("scores_test")
            out_dir.mkdir(parents=True, exist_ok=True)
            for old in out_dir.glob("*.parquet"):
                old.unlink()
            n = n_unseen = 0
            scorer.s1_country = s1_countries(st, "test")
            with timed(st.log, "score test"):
                src = st.input_path("features", "features_test")
                for i, f in enumerate(sorted(src.glob("*.parquet"))):
                    t = _prep(pd.read_parquet(f))
                    t[["q_id", "s1_id"]].assign(p=scorer.predict(t)).to_parquet(
                        out_dir / f"part-{i:05d}.parquet", index=False)
                    n += len(t)
                    n_unseen += int(scorer.unseen_mask(t.s1_id).sum())
            st.register("scores_test", out_dir, rows=n)
            st.metric("test_pairs_unseen_country", n_unseen)
