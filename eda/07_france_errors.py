"""France error dump and proxy realism check (task T7.1).

France appears only in test, so there are no labels. Two views:

A. Real test (run ``main``): for every France and US query take its top-2
   candidates by scorer p. A query is *ambiguous* when p1 > 0.5 and p2 > 0.2.
   Each (query, candidate) pair gets diagnostic tags built from its features
   and records; tag rates on the runner-up of ambiguous queries show which
   lookalike patterns the scorer fails to rule out. France vs US tells what is
   France-specific.
B. Synthetic proxy (run ``synthfr``, labelled): the same tags on false
   positives / missed pairs, plus the same summary statistics as A, to judge
   whether the proxy looks like the real test data.

Output: eda/reports/07_france_errors.txt (+ examples per tag).

Run from student_resource/:  .venv/bin/python eda/07_france_errors.py [--proxy-only]
(--proxy-only: skip the real test scan; the real columns of the realism table are
the reference numbers below, from the full run on 2026-09-25.)
"""

from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
REPORT = ROOT / "eda" / "reports" / "07_france_errors.txt"
SEED = 42

# house-number relation codes (ber.features.numbers)
MISSING, EQUAL, PADDED, TRUNCATED, SHIFTED, DIFFERENT = range(6)

FEATS = ["q_id", "s1_id", "n_core_eq", "n_ratio", "n_q_is_initials", "l_eq", "l_conflict",
         "l_q_empty", "q_vocab_oov", "s_name_freq", "a_q_empty", "a_tsr", "a_bag_eq", "hn_rel",
         "num_q_only", "num_s_only", "box_q", "box_s", "cos_combo_margin"]
REC = ["entity_id", "name_raw", "addr_raw", "name_core", "name_legal", "name_compact",
       "addr_norm", "addr_numbers", "region", "country"]

TAGS = ["legal_conflict", "street_diff_same_num", "num_shifted", "num_truncated",
        "num_different", "initials_mismatch", "initials_match", "q_gibberish",
        "s1_twin", "city_in_s1_name", "q_addr_empty", "name_exact"]


def log(*a):
    print(*a)
    print(*a, file=OUT)


def _ids(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("q_id", "s1_id"):
        if c in df:
            df[c] = df[c].astype(str)
    return df


def top2(scores: pd.DataFrame) -> pd.DataFrame:
    """Top-2 candidates per query with rank (0/1), p1, p2."""
    s = scores.sort_values(["q_id", "p"], ascending=[True, False])
    s["rank"] = s.groupby("q_id").cumcount()
    s = s[s["rank"] < 2]
    p = s.pivot(index="q_id", columns="rank", values="p")
    s = s.join(p.rename(columns={0: "p1", 1: "p2"}), on="q_id")
    s["p2"] = s.p2.fillna(0.0)
    return s


def street_tokens(addr_norm: str, region: str) -> set[str]:
    """Non-numeric address tokens, without region words (city stays in: it is
    the same for both sides of a candidate pair almost always)."""
    reg = set(region.lower().split())
    return {t for t in addr_norm.split() if not t.isdigit() and t not in reg and len(t) > 1}


def tag_pairs(d: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic tags for pairs with q_/s_ record columns and features."""
    t = pd.DataFrame(index=d.index)
    t["legal_conflict"] = d.l_conflict == 1
    qs = [street_tokens(a, r) for a, r in zip(d.q_addr_norm, d.q_region)]
    ss = [street_tokens(a, r) for a, r in zip(d.s_addr_norm, d.s_region)]
    street_diff = np.array([bool(a) and bool(b) and len(a ^ b) > 0 and not (a <= b or b <= a)
                            for a, b in zip(qs, ss)])
    t["street_diff_same_num"] = (d.hn_rel == EQUAL).to_numpy() & street_diff
    t["num_shifted"] = d.hn_rel == SHIFTED
    t["num_truncated"] = d.hn_rel == TRUNCATED
    t["num_different"] = d.hn_rel == DIFFERENT
    looks_initials = d.q_name_compact.str.fullmatch(r"[a-z]{2,5}") & \
        (d.q_name_core.str.count(" ") == 0) & d.q_name_raw.str.strip().str.fullmatch(r"[A-Z.]{2,8}")
    s_init = d.s_name_core.map(lambda s: "".join(w[0] for w in s.split()) if s else "")
    t["initials_match"] = looks_initials & (d.q_name_compact == s_init)
    t["initials_mismatch"] = looks_initials & (d.q_name_compact != s_init)
    t["q_gibberish"] = d.q_vocab_oov >= 0.99
    t["city_in_s1_name"] = [bool(set(n.split()) & {w for w in a.split() if len(w) >= 4 and
                                                    not w.isdigit()})
                            for n, a in zip(d.s_name_core, d.s_addr_norm)]
    t["q_addr_empty"] = d.a_q_empty == 1
    t["name_exact"] = d.n_core_eq == 1
    return t


def attach(pairs: pd.DataFrame, recs: pd.DataFrame) -> pd.DataFrame:
    r = recs.set_index("entity_id")
    q = r.add_prefix("q_")
    s = r.add_prefix("s_")
    return pairs.join(q, on="q_id").join(s, on="s1_id")


# ------------------------------------------------------------------ loading


def load_real(countries: tuple[str, ...], sample: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top-2 pairs (+features) for queries whose best S1 is in ``countries``,
    streamed part by part; ``sample`` keeps that share of a country's queries
    (stable hash). Returns (pairs, records of the countries)."""
    main = ART / "main"
    recs = pd.concat([pd.read_parquet(f, columns=REC, filters=[("country", "in", countries)])
                      for f in sorted(glob.glob(str(main / "normalize/records_test/*.parquet")))])
    s1r = recs[recs.entity_id.str.startswith("S1-")]
    keep_s1 = set(s1r.entity_id)
    s1_country = s1r.set_index("entity_id").country
    out = []
    sparts = sorted(glob.glob(str(main / "scorer/scores_test/*.parquet")))
    fparts = sorted(glob.glob(str(main / "features/features_test/*.parquet")))
    assert len(sparts) == len(fparts), "scores/features parts differ"
    for i, (sf, ff) in enumerate(zip(sparts, fparts)):
        s = _ids(pd.read_parquet(sf))
        best = s.loc[s.groupby("q_id").p.idxmax()]
        best = best[best.s1_id.isin(keep_s1)]
        c = best.s1_id.map(s1_country)
        h = pd.util.hash_pandas_object(best.q_id, index=False).to_numpy() % 1000 / 1000
        qs = set(best.q_id[h < c.map(sample).fillna(1.0).to_numpy()])
        if not qs:
            continue
        t = top2(s[s.q_id.isin(qs)])
        f = _ids(pd.read_parquet(ff, columns=FEATS))
        out.append(t.merge(f, on=["q_id", "s1_id"], how="left"))
        if i % 20 == 0:
            print(f"  part {i}/{len(sparts)}", file=sys.stderr)
    pairs = pd.concat(out, ignore_index=True)
    return pairs, recs


def load_proxy() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run = ART / "synthfr"
    scores = _ids(pd.read_parquet(run / "scorer/scores_val.parquet"))
    feats = pd.concat([_ids(pd.read_parquet(f, columns=FEATS + ["y"]))
                       for f in sorted(glob.glob(str(run / "features/features_train/*.parquet")))])
    recs = pd.concat([pd.read_parquet(f, columns=REC)
                      for f in sorted(glob.glob(str(run / "normalize/records_train/*.parquet")))])
    dec = _ids(pd.read_parquet(run / "decide/decisions_val.parquet"))
    per = pd.read_parquet(run / "decide/val_per_entity.parquet")
    t = top2(scores).merge(feats.drop(columns="y"), on=["q_id", "s1_id"], how="left")
    return t, recs, dec, per


# ------------------------------------------------------------------ reports


def summary(name: str, t: pd.DataFrame, n_s1: int, n_pred: int, n_empty: int) -> dict:
    q = t[t["rank"] == 0]
    top = q.join(tag_pairs(q))
    return {
        "run": name, "queries": len(q), "S1": n_s1,
        "p1<.05": (q.p1 < 0.05).mean(), "p1 .05-.5": q.p1.between(0.05, 0.5).mean(),
        "p1 .5-.95": q.p1.between(0.5, 0.95, inclusive="right").mean(), "p1>.95": (q.p1 > 0.95).mean(),
        "ambiguous": ((q.p1 > 0.5) & (q.p2 > 0.2)).mean(),
        "pred/S1": n_pred / n_s1, "empty S1": n_empty / n_s1,
        "top1 name_exact": top.name_exact.mean(), "top1 legal_conflict": top.legal_conflict.mean(),
        "top1 num_shifted": top.num_shifted.mean(), "top1 num_truncated": top.num_truncated.mean(),
        "top1 q_gibberish": top.q_gibberish.mean(), "top1 initials": (top.initials_match |
                                                                   top.initials_mismatch).mean(),
        "top1 q_addr_empty": top.q_addr_empty.mean(), "s_name_freq med": q.s_name_freq.median(),
        "s_name_freq p90": q.s_name_freq.quantile(0.9),
    }


def tag_rates(label: str, pairs: pd.DataFrame, base: pd.DataFrame | None = None) -> None:
    tg = tag_pairs(pairs)
    rates = tg.mean()
    if base is not None:
        b = tag_pairs(base).mean()
        df = pd.DataFrame({label: rates, "baseline": b, "lift": rates / b.replace(0, np.nan)})
    else:
        df = rates.to_frame(label)
    log(f"\n{label}  (n={len(pairs)})")
    log(df.round(4).to_string())


def examples(label: str, pairs: pd.DataFrame, tag: str, k: int = 8, other: pd.DataFrame | None = None):
    tg = tag_pairs(pairs)
    sel = pairs[tg[tag]]
    if sel.empty:
        return
    sel = sel.sample(min(k, len(sel)), random_state=SEED)
    log(f"\n--- {label} / {tag}: {tg[tag].sum()} pairs; examples")
    for r in sel.itertuples():
        log(f"Q {r.q_id}: {r.q_name_raw} | {r.q_addr_raw}")
        if other is not None:
            w = other[other.q_id == r.q_id]
            for x in w.itertuples():
                log(f"   winner p={x.p:.3f} {x.s1_id}: {x.s_name_raw} | {x.s_addr_raw}")
        log(f"   {'cand' if other is None else 'runner-up'} p={r.p:.3f} {r.s1_id}: "
            f"{r.s_name_raw} | {r.s_addr_raw}   [hn_rel={r.hn_rel} l_conf={r.l_conflict}]")


def s1_twins(pairs: pd.DataFrame) -> pd.Series:
    """For runner-ups: winner and runner-up S1 share the exact core name."""
    w = pairs[pairs["rank"] == 0].set_index("q_id").s_name_core
    r = pairs[pairs["rank"] == 1]
    return r.s_name_core.to_numpy() == w.reindex(r.q_id).to_numpy()


# realism reference from the full run (real test France, model `main`)
REAL_FRANCE = {"p1<.05": 0.3037, "p1 .05-.5": 0.0557, "p1 .5-.95": 0.0286, "p1>.95": 0.6120,
               "ambiguous": 0.0079, "pred/S1": 3.4838, "empty S1": 0.0485,
               "top1 name_exact": 0.5107, "top1 legal_conflict": 0.0563,
               "top1 num_shifted": 0.2343, "top1 num_truncated": 0.0157,
               "top1 q_gibberish": 0.0558, "top1 initials": 0.0167, "top1 q_addr_empty": 0.0300,
               "s_name_freq med": 1.0, "s_name_freq p90": 38.0}


def main(proxy_only: bool = False) -> None:
    log("# 07 France error dump + proxy realism (T7.1)\n")

    # ---------------- B first (small): proxy
    pt, precs, pdec, pper = load_proxy()
    pt = attach(pt, precs)

    if proxy_only:
        row = summary("proxy France", pt, len(pper), int(pper.n_pred.sum()),
                      int((pper.n_pred == 0).sum()))
        log("## Realism: real test France (reference) vs proxy")
        log(pd.DataFrame({"test France": REAL_FRANCE, "proxy France": row}).drop(
            ["run", "queries", "S1"], errors="ignore").round(4).to_string())
        proxy_errors(pt, precs, pdec, pper)
        return

    # ---------------- A: real test
    rt, rrecs = load_real(("France", "US"), {"US": 0.2})
    log("(US: 20% sample of queries)")
    rt = attach(rt, rrecs)
    s1c = rrecs.set_index("entity_id").country
    rt["country"] = rt.s1_id.map(s1c)
    # a query's country = country of its best candidate
    qc = rt[rt["rank"] == 0].set_index("q_id").country
    rt["country"] = rt.q_id.map(qc)

    sub = pd.read_csv(ROOT / "output/matching_results.tsv", sep="\t", dtype=str,
                      keep_default_na=False)
    sub["n"] = sub.matched_entity_ids.map(lambda x: 0 if not x else x.count(",") + 1)
    sub["country"] = sub.source1_entity_id.map(s1c)

    rows = []
    for c in ("US", "France"):
        s = sub[sub.country == c]
        rows.append(summary(f"test {c}", rt[rt.country == c], len(s), int(s.n.sum()),
                            int((s.n == 0).sum())))
    rows.append(summary("proxy France", pt, len(pper), int(pper.n_pred.sum()),
                        int((pper.n_pred == 0).sum())))
    log("## Realism: real test vs proxy (query = S2/S3 record, top1 = best candidate)")
    log(pd.DataFrame(rows).set_index("run").T.round(4).to_string())

    # ---------------- A: ambiguous runner-ups, France vs US
    log("\n## A. Real test: runner-ups of ambiguous queries (p1 > 0.5, p2 > 0.2)")
    amb_q = rt[(rt["rank"] == 0) & (rt.p1 > 0.5) & (rt.p2 > 0.2)].q_id
    amb = rt[rt.q_id.isin(set(amb_q))]
    for c in ("France", "US"):
        a = amb[amb.country == c]
        ru = a[a["rank"] == 1].copy()
        allru = rt[(rt.country == c) & (rt["rank"] == 1)]
        log(f"\n{c}: {len(ru)} ambiguous queries; runner-up S1 has the same core name as the "
            f"winner: {s1_twins(a).mean():.3f}")
        tag_rates(f"{c} ambiguous runner-ups", ru, allru)
    fr = amb[amb.country == "France"]
    fr_w = fr[fr["rank"] == 0][["q_id", "p", "s1_id", "s_name_raw", "s_addr_raw"]]
    fr_r = fr[fr["rank"] == 1]
    for tag in ["legal_conflict", "street_diff_same_num", "initials_mismatch", "num_shifted",
                "num_truncated", "q_gibberish", "city_in_s1_name", "name_exact"]:
        examples("France ambiguous", fr_r, tag, other=fr_w)

    proxy_errors(pt, precs, pdec, pper)


def proxy_errors(pt, precs, pdec, pper) -> None:
    log("\n## B. Proxy errors (labelled)")
    per = pper
    log(f"proxy F0.5 {per.f05.mean():.5f}; S1 {len(per)}; singletons {(per.n_true == 0).sum()}")
    loss = 1 - per.f05
    fn = per.n_true - per.tp
    fpn = per.n_pred - per.tp
    cat = np.select([(per.n_true == 0) & (per.n_pred > 0), (fpn > 0) & (fn > 0), fpn > 0, fn > 0],
                    ["singleton_given_match", "both", "fp_only", "fn_only"], "ok")
    log("loss share by kind:\n" + (loss.groupby(cat).sum() / loss.sum()).round(3).to_string())
    # FP pairs: query matched to the wrong S1. Is the query a distractor (no owner) or
    # a true copy of another S1?
    truth = pd.read_csv(ART / "main/synth-france/data/train/train_ground_truth.tsv", sep="\t",
                        dtype=str, keep_default_na=False)
    owner = {q: s for s, l in zip(truth.source1_entity_id, truth.matched_entity_ids)
             for q in l.split(",") if q}
    fpp = pdec[pdec.y == 0].copy()
    fpp["kind"] = np.where(fpp.q_id.map(owner).isna(), "distractor", "copy_of_other_S1")
    log("\nFP pairs by query kind:\n" + fpp.kind.value_counts().to_string())
    # the decided owner is the query's top-1 candidate, so it is in pt (rank 0)
    fpp = fpp[["q_id", "s1_id", "kind"]].merge(pt, on=["q_id", "s1_id"], how="inner")
    tag_rates("proxy FP pairs", fpp, pt[pt["rank"] == 0])
    for tag in ["legal_conflict", "num_shifted", "num_truncated", "name_exact", "q_gibberish"]:
        examples("proxy FP", fpp, tag, k=5)
    # missed true pairs
    tp_pairs = {(o, q) for q, o in owner.items()}
    got = set(zip(pdec.s1_id, pdec.q_id))
    missed = [(s, q) for s, q in tp_pairs if (s, q) not in got]
    m = pd.DataFrame(missed, columns=["s1_id", "q_id"])
    allp = _ids(pd.read_parquet(ART / "synthfr/scorer/scores_val.parquet"))
    m = m.merge(allp, on=["q_id", "s1_id"], how="left")
    assigned = pdec.set_index("q_id").s1_id
    m["why"] = np.select([m.p.isna(), m.q_id.isin(assigned.index)],
                         ["not_in_candidates", "assigned_to_other_S1"], "dropped_low_p")
    log(f"\nmissed true pairs: {len(m)} of {len(tp_pairs)}")
    log(m.why.value_counts().to_string())
    log("p of dropped: " + m[m.why == "dropped_low_p"].p.describe().round(3).to_string())


if __name__ == "__main__":
    proxy_only = "--proxy-only" in sys.argv
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    path = REPORT.with_name("07_france_proxy.txt") if proxy_only else REPORT
    with open(path, "w", encoding="utf-8") as OUT:
        main(proxy_only)
