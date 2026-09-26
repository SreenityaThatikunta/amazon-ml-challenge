"""E31: one-word swaps at the same address. Pairs whose core names (legal forms removed)
differ in exactly one word on each side (the two words are not typos of each other) and
whose normalised addresses are equal. On labelled val: how often is such a pair a true
match, by how common the S1's swapped word is among S1 names (generic vs distinctive), and
what does the model (v3 stage 2) say? On test: how many such pairs each country has among
its kept pairs (France names are "<word> <generic word> <legal form>").
Usage: .venv/bin/python experiments/word_swap.py"""
import glob

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

COLS = ["entity_id", "country", "name_core", "addr_norm", "source"]


def swap_pairs(sc, rec, df_s1):
    d = sc.join(rec.add_prefix("q_"), on="q_id").join(rec.add_prefix("s_"), on="s1_id")
    d = d[(d.q_addr_norm == d.s_addr_norm) & (d.q_addr_norm != "")]
    qt = [set(n.split()) for n in d.q_name_core]
    st = [set(n.split()) for n in d.s_name_core]
    one = np.array([len(a - b) == 1 and len(b - a) == 1 and len(a & b) >= 1 for a, b in zip(qt, st)])
    d = d[one].copy()
    qa = [next(iter(a - b)) for a, b in zip(np.array(qt, dtype=object)[one], np.array(st, dtype=object)[one])]
    sa = [next(iter(b - a)) for a, b in zip(np.array(qt, dtype=object)[one], np.array(st, dtype=object)[one])]
    d["q_word"], d["s_word"] = qa, sa
    d["typo"] = [Levenshtein.normalized_similarity(a, b) >= 0.6 for a, b in zip(qa, sa)]
    d = d[~d.typo]
    d["s_word_df"] = [df_s1.get(w, 0) for w in d.s_word]
    return d


def word_df(rec):
    s1 = rec[rec.source == 1]
    out = {}
    for c, g in s1.groupby("country"):
        vc = pd.Series([t for n in g.name_core for t in set(n.split())]).value_counts()
        out[c] = vc.to_dict()
    return out


# val (labels): v3 stage-2 scores on plain val
rv = pd.read_parquet("artifacts/v3/normalize/records_train", columns=COLS + ["side"],
                     filters=[("side", "==", "val")]).set_index("entity_id")
sv = pd.read_parquet("artifacts/v3/rerank/scores_val.parquet").astype({"q_id": str, "s1_id": str})
dfv = word_df(rv.reset_index())
rows = []
for c in ("US", "India"):
    rc = rv[rv.country == c]
    s = sv[sv.q_id.isin(rc.index)]
    d = swap_pairs(s, rc[["name_core", "addr_norm"]], dfv[c])
    d["generic"] = pd.cut(d.s_word_df, [0, 10, 100, 1000, 10 ** 9], labels=["<=10", "11-100", "101-1000", ">1000"])
    g = d.groupby("generic", observed=True).agg(n=("y", "size"), true_rate=("y", "mean"), mean_p2=("p", "mean"))
    print(f"\n{c} val: one-word swaps at the same address, by how common the S1's word is among S1 names")
    print(g.round(3).to_string())
    print("examples:", d.sample(min(8, len(d)), random_state=0)[["q_name_core", "s_name_core", "y", "p"]].to_string(index=False))

# test: counts per country among kept pairs (v3)
rt = pd.read_parquet("artifacts/v3/normalize/records_test", columns=COLS).set_index("entity_id")
dft = word_df(rt.reset_index())
st_ = pd.concat(pd.read_parquet(f, columns=["q_id", "s1_id", "p"])
                for f in sorted(glob.glob("artifacts/v3/rerank/scores_test/*.parquet"))).astype({"q_id": str, "s1_id": str})
print("\ntest: one-word swaps at the same address among kept pairs (v3)")
for c in ("US", "India", "France"):
    rc = rt[rt.country == c]
    s = st_[st_.q_id.isin(rc.index)]
    d = swap_pairs(s, rc[["name_core", "addr_norm"]], dft[c])
    nq = int((rc.source != 1).sum())
    print(f"{c}: {len(d)} pairs = {len(d) / nq:.4f} per record; p2 mean {d.p.mean():.3f}; "
          f"p2 in [0.2,0.8] {((d.p >= .2) & (d.p <= .8)).mean():.3f}; S1 word df median {d.s_word_df.median():.0f}")
