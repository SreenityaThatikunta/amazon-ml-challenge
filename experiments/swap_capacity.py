"""E31: are French one-word swaps true copies or extra distractor records? Label-free test via
copy counts: a true copy of an S1 takes one of its usual copy slots, a distractor is an extra
record. For S1 with a same-address one-word-swap record vs S1 without, the number of *other*
records confidently linked to the S1 (p2 >= 0.9, best S1 of the record, not a swap pair).
Checked on US/India val with labels first (true swap vs false swap), then applied to test.
Usage: .venv/bin/python experiments/swap_capacity.py"""
import glob

import numpy as np
import pandas as pd

src = open("experiments/word_swap.py").read().split("# val (labels)")[0]
ws = {}
exec(src, ws)


def confident_other_links(sc, swaps):
    best = sc.sort_values(["q_id", "p"], ascending=[True, False]).drop_duplicates("q_id")
    best = best[best.p >= 0.9]
    sw = set(zip(swaps.q_id, swaps.s1_id))
    best = best[[k not in sw for k in zip(best.q_id, best.s1_id)]]
    return best.groupby("s1_id").size()


def report(name, sc, rec, dfw, labels=False):
    sw = ws["swap_pairs"](sc, rec[["name_core", "addr_norm"]], dfw)
    sw = sw[sw.s_word_df > 100]                         # generic swapped word
    n_other = confident_other_links(sc, sw)
    s1 = rec.index[rec.source == 1]
    cnt = n_other.reindex(s1).fillna(0)
    has = cnt.index.isin(set(sw.s1_id))
    out = {"S1 without swap": round(float(cnt[~has].mean()), 3), "S1 with swap": round(float(cnt[has].mean()), 3),
           "n with swap": int(has.sum())}
    if labels:
        t = set(sw.s1_id[sw.y == 1]); f = set(sw.s1_id[sw.y == 0])
        out["S1 with TRUE swap"] = round(float(cnt[cnt.index.isin(t - f)].mean()), 3)
        out["S1 with FALSE swap"] = round(float(cnt[cnt.index.isin(f - t)].mean()), 3)
    print(name, out)


COLS = ws["COLS"]
rv = pd.read_parquet("artifacts/v3/normalize/records_train", columns=COLS + ["side"],
                     filters=[("side", "==", "val")]).set_index("entity_id")
sv = pd.read_parquet("artifacts/v3/rerank/scores_val.parquet").astype({"q_id": str, "s1_id": str})
dfv = ws["word_df"](rv.reset_index())
for c in ("US", "India"):
    rc = rv[rv.country == c]
    report(f"{c} val", sv[sv.q_id.isin(rc.index)], rc, dfv[c], labels=True)
rt = pd.read_parquet("artifacts/v3/normalize/records_test", columns=COLS).set_index("entity_id")
dft = ws["word_df"](rt.reset_index())
st_ = pd.concat(pd.read_parquet(f, columns=["q_id", "s1_id", "p"])
                for f in sorted(glob.glob("artifacts/v3/rerank/scores_test/*.parquet"))).astype({"q_id": str, "s1_id": str})
for c in ("US", "India", "France"):
    rc = rt[rt.country == c]
    report(f"{c} test", st_[st_.q_id.isin(rc.index)], rc, dft[c])
