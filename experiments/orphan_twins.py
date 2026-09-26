"""H-test (plan.md §0, 2.1): are test's extra "no-match" records copies of S1 that were
dropped from the test pool? Copies of one missing entity (its S2 and S3 records) would
be near-identical to each other while matching nothing in S1.

For each country: orphans = S2/S3 records whose best stage-1 p1 < 0.05 (records with no
kept pair count as orphans). Val orphans are compared with test orphans from a set of
test regions with about as many orphans (similar density). For each record, the cosine to
its nearest *other* orphan (char 3-gram TF-IDF over core name + address). Control: val
records that do have an owner, same size, where near twins are the S2/S3 copies of one S1.
Usage: .venv/bin/python experiments/orphan_twins.py [run]   (default run v3)"""
import glob
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

RUN = sys.argv[1] if len(sys.argv) > 1 else "v3"
A = f"artifacts/{RUN}"
COLS = ["entity_id", "source", "country", "region", "name_core", "addr_norm"]
rng = np.random.default_rng(0)


def load(split, country, extra=()):
    fs = sorted(glob.glob(f"{A}/normalize/records_{split}/*.parquet"))
    return pd.concat([pd.read_parquet(f, columns=COLS + list(extra),
                                      filters=[("country", "==", country)]) for f in fs],
                     ignore_index=True)


def best_p(files):
    out = []
    for f in files:
        s = pd.read_parquet(f, columns=["q_id", "p"])
        out.append(s.groupby("q_id").p.max())
    return pd.concat(out).groupby(level=0).max()


def nn_cos(df):
    text = (df.name_core + " | " + df.addr_norm).to_numpy()
    X = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=1, dtype=np.float32).fit_transform(text)
    C = sp_matmul_topn(X, X.T.tocsr(), top_n=2, threshold=0.3, sort=True, n_threads=8).tocsr()
    best = np.zeros(len(df), dtype=np.float32)
    for i in range(len(df)):
        a, b = C.indptr[i], C.indptr[i + 1]
        for j, v in zip(C.indices[a:b], C.data[a:b]):
            if j != i:
                best[i] = max(best[i], v)
    return best


def summary(name, cos, n):
    return {"set": name, "n": n, "twin>=0.9": round(float((cos >= 0.9).mean()), 4),
            "twin>=0.8": round(float((cos >= 0.8).mean()), 4),
            "twin>=0.7": round(float((cos >= 0.7).mean()), 4)}


pv = best_p([f"{A}/scorer/scores_val.parquet"])
pt = best_p(sorted(glob.glob(f"{A}/scorer/scores_test/*.parquet")))
rows = []
for country in ("US", "India"):
    tr = load("train", country, ["owner", "side"])
    vq = tr[(tr.side == "val") & (tr.source != 1)].copy()
    vq["p1"] = vq.entity_id.map(pv).fillna(0.0)
    v_orph = vq[vq.p1 < 0.05]
    v_unm = vq[vq.owner == ""]
    v_own = vq[vq.owner != ""]
    v_own = v_own[v_own.owner.isin(rng.choice(v_own.owner.unique(),
                                              size=min(v_own.owner.nunique(), len(v_orph) // 3),
                                              replace=False))]
    te = load("test", country)
    tq = te[te.source != 1].copy()
    tq["p1"] = tq.entity_id.map(pt).fillna(0.0)
    t_orph_all = tq[tq.p1 < 0.05]
    # test regions (random order) until about as many orphans as val
    counts = t_orph_all.region.value_counts()
    regs, tot = [], 0
    for r in rng.permutation(counts.index.to_numpy()):
        if tot >= len(v_orph):
            break
        regs.append(r); tot += counts[r]
    t_orph = t_orph_all[t_orph_all.region.isin(regs)]
    for name, df in ((f"{country} val orphans (p1<.05)", v_orph),
                     (f"{country} val truly unmatched", v_unm),
                     (f"{country} test orphans (p1<.05), {len(regs)} regions", t_orph),
                     (f"{country} control: val matched copies", v_own)):
        rows.append(summary(name, nn_cos(df.reset_index(drop=True)), len(df)))
        print(rows[-1], flush=True)
    print(f"{country}: orphans per query val {len(v_orph) / len(vq):.3f}, "
          f"test {len(t_orph_all) / len(tq):.3f}", flush=True)
print(pd.DataFrame(rows).to_string(index=False))
