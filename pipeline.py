"""End-to-end pipeline: raw data -> normalisation -> blocking -> features -> 2-stage model -> output.

Every stage caches its result in the work directory and is skipped when the
cache exists, so an interrupted run can simply be restarted.
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass

import lightgbm as lgb
import numpy as np
import polars as pl

from . import blocking, features, metrics, model, prepare, resources

SRCS = (2, 3)
FEAT_REC_COLS = ["business_name", "n_full", "n_core", "n_cat", "n_alt", "n_toks", "n_isdom", "n_hasdba",
                 "a_full", "a_nums", "a_words", "ctry"]
IDS = ["s1_idx", "src", "o_idx"]
# stage-1 features carried into the (compact) stage-2 re-ranker
KEY_FEATS = ["mg_sum_na", "sum_na", "num_tset", "mg_bscore", "num_n_b", "mg_a_tset", "n_full_ratio",
             "mg_n_full_ratio", "num_first_reldiff", "a_tset", "nt_unmatched_b", "n_cat_b_in_a", "amb_a",
             "amb_b", "addr_empty_b", "n_core_tset", "bscore", "o_ncand", "b_rel", "isdom_b", "indic_b"]
CTX_FEATS = ["p1", "p1_o_other", "p1_o_rank", "p1_o_sum", "p1_o_margin", "p1_s1_nstrong_other",
             "p1_s1_psum_other", "p1_s1_pmax_other", "p1_s1_ncand"]
STAGE2_FEATS = KEY_FEATS + CTX_FEATS


@dataclass
class Config:
    top_k: int = 10            # max Source-1 candidates kept per Source-2/3 record
    min_rel: float = 0.4       # ... and only those scoring >= min_rel x the record's best
    max_df: int = 100          # blocking keys shared by > max_df Source-1 records are ignored
    frac_val: float = 0.03     # share of TRAIN Source-1 entities held out for validation
    frac_train: float = 0.05   # share of TRAIN Source-1 entities whose records train the model
    rounds1: int = 400
    rounds2: int = 300
    block_chunk: int = 100_000
    feat_chunk: int = 60_000
    seed: int = 0


def log(msg: str) -> None:
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


class Paths:
    def __init__(self, work: str):
        self.work = work
        os.makedirs(work, exist_ok=True)

    def __call__(self, name: str) -> str:
        return os.path.join(self.work, name)


# ----------------------------------------------------------------------------
# Stage 1: normalisation
# ----------------------------------------------------------------------------
def stage_prepare(ds: prepare.DataSource, P: Paths) -> dict:
    res_path = P("resources.json")
    if not os.path.exists(res_path):
        log("learning transliteration resources from the training ground truth")
        s1 = prepare.read_tsv(ds.tsv("train", "train_source1.tsv"))
        oth = pl.concat([prepare.read_tsv(ds.tsv("train", f"train_source{k}.tsv")) for k in SRCS])
        gt = prepare.read_ground_truth(ds.tsv("train", "train_ground_truth.tsv"))
        res = resources.learn_resources(s1, oth, gt)
        resources.save(res, res_path)
        log(f"  {len(res['translit_tokens'])} native-script words, {len(res['translit_phrases'])} state names, "
            f"{len(res['name_canon'])} spelling-variant classes")
        del s1, oth, gt
        gc.collect()
    res = resources.load(res_path)
    for split in ("train", "test"):
        for k in (1, 2, 3):
            out = P(f"{split}_s{k}.parquet")
            if not os.path.exists(out):
                prepare.prepare_source(ds.tsv(split, f"{split}_source{k}.tsv"), res, out, label=f"{split} source {k}")
                gc.collect()
    return res


# ----------------------------------------------------------------------------
# Stage 2: blocking
# ----------------------------------------------------------------------------
def stage_block(P: Paths, split: str, cfg: Config) -> None:
    outs = [P(f"{split}_cand_s{k}.parquet") for k in SRCS]
    if all(os.path.exists(o) for o in outs):
        return
    s1 = pl.read_parquet(P(f"{split}_s1.parquet"), columns=blocking.BLOCK_COLS)
    t = time.time()
    index = blocking.build_s1_index(s1, max_df=cfg.max_df)
    log(f"{split}: Source-1 inverted index with {index.height:,} postings ({time.time()-t:.0f}s)")
    del s1
    gc.collect()
    for k, out in zip(SRCS, outs):
        if os.path.exists(out):
            continue
        t = time.time()
        o = pl.read_parquet(P(f"{split}_s{k}.parquet"), columns=blocking.BLOCK_COLS)
        c = blocking.generate_candidates(index, o, top_k=cfg.top_k, min_rel=cfg.min_rel,
                                         chunk=cfg.block_chunk, verbose=False)
        c.write_parquet(out)
        log(f"{split}: source {k}: {o.height:,} records -> {c.height:,} candidate pairs ({time.time()-t:.0f}s)")
        del o, c
        gc.collect()
    del index
    gc.collect()


# ----------------------------------------------------------------------------
# Stage 3: features
# ----------------------------------------------------------------------------
def load_records(P: Paths, split: str, k: int, rows: pl.Series | None = None) -> pl.DataFrame:
    lf = pl.scan_parquet(P(f"{split}_s{k}.parquet")).select(FEAT_REC_COLS).with_row_index("_row")
    if rows is not None:
        lf = lf.filter(pl.col("_row").is_in(rows.implode()))
    return features.enrich(lf.collect()).drop("business_name")


def iter_feature_chunks(P: Paths, split: str, k: int, cand: pl.DataFrame, S1: pl.DataFrame, stats: dict,
                        chunk_rec: int):
    """Yield feature frames for whole groups of Source-2/3 records (all candidates of a record together)."""
    rows = cand["o_idx"].unique().sort()
    O = load_records(P, split, k, rows)
    pos = O.select(pl.col("_row").alias("o_idx"), pl.int_range(pl.len(), dtype=pl.UInt32).alias("_opos"))
    O = O.drop("_row")
    cand = cand.join(pos, on="o_idx", how="inner").sort("o_idx", "brank")
    o_ids = cand["o_idx"].unique().sort()
    for i in range(0, len(o_ids), chunk_rec):
        lo, hi = o_ids[i], o_ids[min(i + chunk_rec, len(o_ids)) - 1]
        c = cand.filter((pl.col("o_idx") >= lo) & (pl.col("o_idx") <= hi))
        f = features.pair_features(c.rename({"o_idx": "_orig", "_opos": "o_idx"}), S1, O, stats)
        f = f.rename({"o_idx": "_opos"}).rename({"_orig": "o_idx"}).drop("_opos")
        yield features.group_features(f).with_columns(pl.lit(k, dtype=pl.UInt8).alias("src"))


def write_parts(frames, out_path: str) -> int:
    parts_dir = out_path + ".parts"
    if os.path.isdir(parts_dir):
        shutil.rmtree(parts_dir)
    os.makedirs(parts_dir)
    files, n = [], 0
    for j, f in enumerate(frames):
        fp = os.path.join(parts_dir, f"part_{j:05d}.parquet")
        f.write_parquet(fp)
        files.append(fp)
        n += f.height
    if files:
        pl.scan_parquet(files).sink_parquet(out_path)
    shutil.rmtree(parts_dir)
    return n


def labels_for(P: Paths, ds: prepare.DataSource) -> pl.DataFrame:
    """(s1_idx, src, o_idx) of every ground-truth pair in the training split."""
    path = P("train_truth.parquet")
    if os.path.exists(path):
        return pl.read_parquet(path)
    s1_ids = pl.read_parquet(P("train_s1.parquet"), columns=["entity_id"]).with_row_index("s1_idx")
    gt = prepare.read_ground_truth(ds.tsv("train", "train_ground_truth.tsv"))
    gt = gt.join(s1_ids.rename({"entity_id": "s1"}), on="s1")
    parts = []
    for k in SRCS:
        o_ids = pl.read_parquet(P(f"train_s{k}.parquet"), columns=["entity_id"]).with_row_index("o_idx")
        parts.append(gt.join(o_ids.rename({"entity_id": "mid"}), on="mid").select(
            pl.col("s1_idx").cast(pl.UInt32), pl.lit(k, dtype=pl.UInt8).alias("src"), pl.col("o_idx").cast(pl.UInt32)))
    t = pl.concat(parts)
    t.write_parquet(path)
    return t


def dev_split(P: Paths, cfg: Config) -> tuple:
    """Entity-level split of TRAIN.  V (validation) and T (training) are disjoint random sets of
    Source-1 entities.  O_V = every record whose candidate list touches V (so V is scored exactly
    as the test set will be);  O_T = records touching T, minus O_V."""
    n1 = pl.read_parquet(P("train_s1.parquet"), columns=["ctry"]).height
    u = np.random.default_rng(cfg.seed).random(n1)
    V = pl.Series("s1_idx", np.nonzero(u < cfg.frac_val)[0].astype(np.uint32))
    T = pl.Series("s1_idx", np.nonzero((u >= cfg.frac_val) & (u < cfg.frac_val + cfg.frac_train))[0].astype(np.uint32))
    sel = {}
    for k in SRCS:
        c = pl.read_parquet(P(f"train_cand_s{k}.parquet"), columns=["o_idx", "s1_idx"])
        v_o = c.filter(pl.col("s1_idx").is_in(V.implode()))["o_idx"].unique()
        t_o = c.filter(pl.col("s1_idx").is_in(T.implode()) & ~pl.col("o_idx").is_in(v_o.implode()))["o_idx"].unique()
        sel[k] = {"val": v_o, "train": t_o}
    return V, T, sel


def add_labels(df: pl.DataFrame, truth: pl.DataFrame) -> pl.DataFrame:
    return df.join(truth.with_columns(pl.lit(1, dtype=pl.UInt8).alias("y")), on=IDS, how="left").with_columns(
        pl.col("y").fill_null(0))


# ----------------------------------------------------------------------------
# Scoring (streamed: full feature matrices are never all in memory at once)
# ----------------------------------------------------------------------------
def score_split(P: Paths, split: str, cands: dict, S1: pl.DataFrame, stats: dict, boosters1: list,
                cols1: list, cfg: Config, out_path: str) -> pl.DataFrame:
    """Features -> stage-1 probability for every candidate pair; keeps a compact frame
    (ids + p1 + KEY_FEATS) on disk for the stage-2 re-ranker."""
    if not os.path.exists(out_path):
        def gen():
            for k in SRCS:
                t = time.time()
                n = 0
                for f in iter_feature_chunks(P, split, k, cands[k], S1, stats, cfg.feat_chunk):
                    X = model.to_numpy(f, cols1)
                    p1 = sum(b.predict(X) for b in boosters1) / len(boosters1)
                    n += f.height
                    yield f.select(IDS + KEY_FEATS).with_columns(pl.Series("p1", p1.astype(np.float32)))
                log(f"{split}: scored source {k}: {n:,} pairs ({time.time()-t:.0f}s)")
        write_parts(gen(), out_path)
    return pl.read_parquet(out_path)


def load_matrix(lf: pl.LazyFrame, cols: list[str], chunk: int = 1_000_000) -> np.ndarray:
    """Materialise feature columns as one float32 matrix without an intermediate full DataFrame."""
    n = lf.select(pl.len()).collect().item()
    X = np.empty((n, len(cols)), dtype=np.float32)
    for off in range(0, n, chunk):
        X[off:off + chunk] = model.to_numpy(lf.slice(off, chunk).select(cols).collect(), cols)
    return X


def stage2_predict(df: pl.DataFrame, booster2, chunk: int = 2_000_000) -> np.ndarray:
    ctx = model.stage2_features(df.select(IDS + ["p1"]), "p1").select(CTX_FEATS)
    out = []
    for off in range(0, df.height, chunk):
        Xk = model.to_numpy(df.slice(off, chunk), KEY_FEATS)
        Xc = model.to_numpy(ctx.slice(off, chunk), CTX_FEATS)
        out.append(booster2.predict(np.hstack([Xk, Xc])))
    return np.concatenate(out).astype(np.float32)


def evaluate_grid(va: pl.DataFrame, V: pl.Series, vt: pl.DataFrame) -> list[dict]:
    grid = [round(float(x), 3) for x in np.arange(0.30, 0.951, 0.025)]
    results = []
    for pcol in ("p1", "p2"):
        for thr in grid:
            d = model.decide(va, pcol, thr).filter(pl.col("s1_idx").is_in(V.implode()))
            pred = d.select(pl.col("s1_idx").alias("s1"), (pl.col("src").cast(pl.Int64) * 10**10 + pl.col("o_idx")).alias("mid"))
            r = metrics.macro_f05(pred, vt, V.rename("s1"))
            results.append({"model": pcol, "thr": thr, "F05": r["F05"], "F05_singletons": r["F05_singletons"],
                            "F05_nonsingle": r["F05_nonsingle"], "micro_P": r["micro_P"], "micro_R": r["micro_R"]})
    return results


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def run(data: str, work: str, out_dir: str, cfg: Config) -> dict:
    t_all = time.time()
    P = Paths(work)
    ds = prepare.DataSource(data)
    report: dict = {"config": asdict(cfg)}

    # 1-2. normalise + block ----------------------------------------------------
    stage_prepare(ds, P)
    stage_block(P, "train", cfg)
    stage_block(P, "test", cfg)

    # 3. training / validation data ---------------------------------------------
    truth = labels_for(P, ds)
    V, T, sel = dev_split(P, cfg)
    S1 = load_records(P, "train", 1).drop("_row")
    stats = features.build_stats(S1)
    train_feat = P("train_T_features.parquet")
    if not os.path.exists(train_feat):
        def gen():
            for k in SRCS:
                c = pl.read_parquet(P(f"train_cand_s{k}.parquet")).filter(pl.col("o_idx").is_in(sel[k]["train"].implode()))
                t = time.time()
                n = 0
                for f in iter_feature_chunks(P, "train", k, c, S1, stats, cfg.feat_chunk):
                    n += f.height
                    yield f
                log(f"train: features for training records, source {k}: {n:,} pairs ({time.time()-t:.0f}s)")
        write_parts(gen(), train_feat)

    # 4. stage-1 model: 2 folds (split by record) -> out-of-fold p1 for stage 2
    lf = pl.scan_parquet(train_feat)
    ids = add_labels(lf.select(IDS).collect(), truth)
    cols1 = model.feature_columns(lf.head(1000).collect().with_columns(pl.lit(0, dtype=pl.UInt8).alias("y")))
    X = load_matrix(lf, cols1)
    Xkey = X[:, [cols1.index(c) for c in KEY_FEATS]].copy()
    ids = ids.with_columns(
        ((pl.col("o_idx").cast(pl.UInt64) * 40503 + pl.col("src").cast(pl.UInt64)) % 2).cast(pl.UInt8).alias("fold"))
    y = ids["y"].to_numpy()
    fold = ids["fold"].to_numpy()
    log(f"stage 1: {len(y):,} training pairs, {len(cols1)} features, positive rate {y.mean():.3f}")
    boosters1, oof = [], np.zeros(len(y), dtype=np.float32)
    for f in (0, 1):
        mpath = P(f"model_stage1_fold{f}.txt")
        if os.path.exists(mpath):
            b = lgb.Booster(model_file=mpath)
        else:
            t = time.time()
            b = lgb.train(model.LGB_PARAMS, lgb.Dataset(X[fold == f], label=y[fold == f], feature_name=cols1),
                          num_boost_round=cfg.rounds1)
            b.save_model(mpath)
            log(f"stage 1 fold {f} trained ({time.time()-t:.0f}s)")
        oof[fold != f] = b.predict(X[fold != f])
        boosters1.append(b)
    del X
    gc.collect()
    imp = sorted(zip(cols1, boosters1[0].feature_importance("gain")), key=lambda z: -z[1])
    report["stage1_features"] = cols1
    report["stage1_top_features"] = [(c, round(float(g))) for c, g in imp[:30]]

    # 5. stage-2 re-ranker on out-of-fold p1 + context ------------------------
    m2path = P("model_stage2.txt")
    if os.path.exists(m2path):
        booster2 = lgb.Booster(model_file=m2path)
    else:
        ctx = pl.concat([ids, pl.DataFrame(Xkey, schema=KEY_FEATS)], how="horizontal").with_columns(pl.Series("p1", oof))
        ctx = model.stage2_features(ctx, "p1")
        ctx = ctx.filter(pl.col("s1_idx").is_in(T.implode()))   # entity groups of T are complete
        t = time.time()
        booster2 = lgb.train(model.LGB_PARAMS, lgb.Dataset(model.to_numpy(ctx, STAGE2_FEATS), label=ctx["y"].to_numpy(),
                                                           feature_name=STAGE2_FEATS), num_boost_round=cfg.rounds2)
        booster2.save_model(m2path)
        log(f"stage 2 trained on {ctx.height:,} pairs ({time.time()-t:.0f}s)")
        del ctx
    imp2 = sorted(zip(STAGE2_FEATS, booster2.feature_importance("gain")), key=lambda z: -z[1])
    report["stage2_top_features"] = [(c, round(float(g))) for c, g in imp2]
    del Xkey, oof, ids
    gc.collect()

    # 6. validation on held-out entities V -> choose model + threshold --------
    val_cands = {k: pl.read_parquet(P(f"train_cand_s{k}.parquet")).filter(pl.col("o_idx").is_in(sel[k]["val"].implode()))
                 for k in SRCS}
    va = score_split(P, "train", val_cands, S1, stats, boosters1, cols1, cfg, P("val_scored.parquet"))
    va = va.with_columns(pl.Series("p2", stage2_predict(va, booster2)))
    vt = truth.filter(pl.col("s1_idx").is_in(V.implode()))
    vt = vt.select(pl.col("s1_idx").alias("s1"), (pl.col("src").cast(pl.Int64) * 10**10 + pl.col("o_idx")).alias("mid"))
    results = evaluate_grid(va, V, vt)
    res_df = pl.DataFrame(results)
    best = res_df.sort("F05", descending=True).row(0, named=True)
    for pcol in ("p1", "p2"):
        b = res_df.filter(pl.col("model") == pcol).sort("F05", descending=True).row(0, named=True)
        log(f"validation ({len(V):,} held-out entities) {pcol}: best F0.5={b['F05']:.5f} at thr={b['thr']} "
            f"(singletons {b['F05_singletons']:.4f}, others {b['F05_nonsingle']:.4f}, P={b['micro_P']:.4f}, R={b['micro_R']:.4f})")
    vc = pl.concat([val_cands[k].select("s1_idx", "o_idx").with_columns(pl.lit(k, dtype=pl.UInt8).alias("src")) for k in SRCS])
    ceil = truth.filter(pl.col("s1_idx").is_in(V.implode())).join(vc, on=IDS, how="semi").height / max(vt.height, 1)
    report.update({"validation_grid": results, "chosen": best, "val_entities": len(V),
                   "val_true_pairs": vt.height, "val_blocking_recall": ceil, "train_entities": len(T),
                   "val_pairs_scored": va.height})
    log(f"chosen: {best['model']} thr={best['thr']}  (blocking recall ceiling on validation: {ceil:.4f})")
    va.select(IDS + ["p1", "p2"]).write_parquet(P("val_predictions.parquet"))
    del va, val_cands, S1, stats
    gc.collect()

    # 7. test -------------------------------------------------------------------
    S1t = load_records(P, "test", 1).drop("_row")
    stats_t = features.build_stats(S1t)
    test_cands = {k: pl.read_parquet(P(f"test_cand_s{k}.parquet")) for k in SRCS}
    te = score_split(P, "test", test_cands, S1t, stats_t, boosters1, cols1, cfg, P("test_scored.parquet"))
    del S1t, stats_t, test_cands
    gc.collect()
    te = te.with_columns(pl.Series("p2", stage2_predict(te, booster2)))
    dec = model.decide(te, best["model"], best["thr"])
    report["test_pairs_scored"] = te.height
    report["test_links"] = dec.height
    write_outputs(P, te.select(IDS), dec, out_dir)
    del te
    gc.collect()
    report["runtime_min"] = round((time.time() - t_all) / 60, 1)

    # 8. validate the output format with the organisers' script ----------------
    run_validator(ds, data, P, out_dir, report)
    with open(os.path.join(out_dir, "run_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=str)
    log(f"done in {report['runtime_min']} min")
    return report


def run_validator(ds: prepare.DataSource, data: str, P: Paths, out_dir: str, report: dict) -> None:
    """Run the organisers' utils/validate_submission.py (with --check-ids) on the outputs."""
    tmp_files = []
    if ds.zip is not None:
        vpath = ds.extract_file("utils/validate_submission.py", P("validate_submission.py"))
        tdir = P("test_tsv")
        os.makedirs(tdir, exist_ok=True)
        for k in (1, 2, 3):
            fp = os.path.join(tdir, f"test_source{k}.tsv")
            if not os.path.exists(fp):
                with open(fp, "wb") as fh:
                    fh.write(ds.tsv("test", f"test_source{k}.tsv"))
            tmp_files.append(fp)
    else:
        vpath = os.path.join(os.path.dirname(os.path.abspath(data)), "utils", "validate_submission.py")
        tdir = os.path.join(data, "test")
    if not vpath or not os.path.exists(vpath):
        log("validator script not found - skipped")
        return
    r = subprocess.run([sys.executable, vpath, "--matching", os.path.join(out_dir, "matching_results.tsv"),
                        "--candidate", os.path.join(out_dir, "candidate_pairs.tsv"), "--test-dir", tdir, "--check-ids"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    msg = (r.stdout + r.stderr).strip()
    log("validator output:\n" + msg)
    report["validator"] = msg[-3000:]
    report["validator_exit_code"] = r.returncode
    for fp in tmp_files:          # free ~1.2 GB of disk
        try:
            os.remove(fp)
        except OSError:
            pass


def write_outputs(P: Paths, cand_ids: pl.DataFrame, dec: pl.DataFrame, out_dir: str) -> None:
    """matching_results.tsv + candidate_pairs.tsv (one row per test Source-1 entity)."""
    os.makedirs(out_dir, exist_ok=True)
    s1 = pl.read_parquet(P("test_s1.parquet"), columns=["entity_id"]).with_row_index("s1_idx")
    s1 = s1.with_columns(pl.col("s1_idx").cast(pl.UInt32))
    oid = pl.concat([pl.read_parquet(P(f"test_s{k}.parquet"), columns=["entity_id"]).with_row_index("o_idx")
                     .with_columns(pl.lit(k, dtype=pl.UInt8).alias("src")) for k in SRCS])
    oid = oid.with_columns(pl.col("o_idx").cast(pl.UInt32))

    def listing(pairs: pl.DataFrame, colname: str) -> pl.DataFrame:
        g = (pairs.select(pl.col("s1_idx").cast(pl.UInt32), pl.col("src").cast(pl.UInt8), pl.col("o_idx").cast(pl.UInt32))
             .unique()
             .join(oid, on=["src", "o_idx"], how="inner")
             .sort("s1_idx", "src", "o_idx")
             .group_by("s1_idx", maintain_order=True).agg(pl.col("entity_id").str.join(",").alias(colname)))
        out = s1.join(g, on="s1_idx", how="left").sort("s1_idx")
        return out.select(pl.col("entity_id").alias("source1_entity_id"), pl.col(colname).fill_null(""))

    m = listing(dec, "matched_entity_ids")
    c = listing(cand_ids, "candidate_entity_ids")
    m.write_csv(os.path.join(out_dir, "matching_results.tsv"), separator="\t", quote_style="never")
    c.write_csv(os.path.join(out_dir, "candidate_pairs.tsv"), separator="\t", quote_style="never")
    log(f"wrote matching_results.tsv ({(m['matched_entity_ids'] != '').sum():,} of {m.height:,} entities matched, "
        f"{dec.height:,} links) and candidate_pairs.tsv")
