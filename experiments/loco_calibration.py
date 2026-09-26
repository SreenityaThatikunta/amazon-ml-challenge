"""E30: is the leave-one-country-out (LOCO) recall loss a calibration problem?
For each run: reliability of p2 per country (mean p vs true rate by p bin) and the
F0.5 per country with (a) the run's decision rule, (b) the best global logit shift
p' = sigmoid(logit(p) + b) chosen per country with labels (oracle: upper bound of what a
label-free France calibration could recover).
Usage: .venv/bin/python experiments/loco_calibration.py RUN [RUN ...]  (val_ts runs)"""
import sys

import numpy as np
import pandas as pd

from ber.decide.stage import decide, evaluate, one_owner

REC = "artifacts/v3ts/normalize/records_train"
recs = pd.read_parquet(REC, columns=["entity_id", "source", "country", "side", "owner"],
                       filters=[("side", "==", "val")])
s1 = recs[recs.source == 1].set_index("entity_id")[["country"]]
q = recs[(recs.source != 1) & recs.owner.isin(s1.index)]
truth = q.groupby("owner").entity_id.apply(list).to_dict()
s1["segment"] = np.where(s1.index.isin(set(truth)), "has_matches", "singleton")
s1["has_indic_match"] = False
cty_q = recs.set_index("entity_id").country


def f05(sc, meta, b=0.0, min_p=0.02):
    sc = sc.copy()
    if b:
        lp = np.log(sc.p.clip(1e-7, 1 - 1e-7) / (1 - sc.p.clip(1e-7, 1 - 1e-7)))
        sc["p"] = 1 / (1 + np.exp(-(lp + b)))
    kept = decide(one_owner(sc), "expected_f", 0, min_p, 1.0)
    return float(evaluate(kept, truth, meta).f05.mean())


for run in sys.argv[1:]:
    sv = pd.read_parquet(f"artifacts/{run}/rerank/scores_val.parquet").astype({"q_id": str, "s1_id": str})
    sv["country"] = cty_q.reindex(sv.q_id).to_numpy()
    for c in ("US", "India"):
        sc = sv[sv.country == c]
        meta = s1[s1.country == c]
        bins = pd.cut(sc.p, [0, .05, .2, .5, .8, .95, .99, 1.0], include_lowest=True)
        rel = sc.groupby(bins, observed=True).agg(n=("y", "size"), mean_p=("p", "mean"), rate=("y", "mean"))
        base = f05(sc, meta)
        shifts = {b: f05(sc, meta, b) for b in (-1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 3.0)}
        best = max(shifts, key=shifts.get)
        print(f"\n== {run} {c}: F0.5 {base:.5f}; best logit shift {best:+.1f} -> {shifts[best]:.5f}  "
              f"all: { {k: round(v, 5) for k, v in shifts.items()} }")
        print(rel.round(4).to_string())
