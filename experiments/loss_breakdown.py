"""Where the remaining val loss is (sum of 1 - F0.5 over val S1), for any run whose
decide stage has run: loss kind, and for missed true pairs the reason (never a
candidate / not in the rescored top-k / assigned to another S1 / dropped at low p).
Usage (from student_resource/): .venv/bin/python experiments/loss_breakdown.py <run> [scores_stage]"""
import sys

import numpy as np
import pandas as pd

run = sys.argv[1]
stage = sys.argv[2] if len(sys.argv) > 2 else "rescore"
A = f"artifacts/{run}"
per = pd.read_parquet(f"{A}/decide/val_per_entity.parquet")
loss = 1 - per.f05
fn, fp = per.n_true - per.tp, per.n_pred - per.tp
kind = np.select([(per.n_true == 0) & (per.n_pred > 0), (fp > 0) & (fn > 0), fp > 0, fn > 0],
                 ["singleton_given_match", "both", "fp_only", "fn_only"], "ok")
print(f"{run}: val F0.5 {per.f05.mean():.5f}, loss/S1 {loss.mean():.5f}")
print("loss share by kind:\n" + (loss.groupby(kind).sum() / loss.sum()).round(3).to_string())
print("loss share by country:\n" + (loss.groupby(per.country).sum() / loss.sum()).round(3).to_string())

recs = pd.read_parquet(f"{A}/normalize/records_train", columns=["entity_id", "owner", "side"])
val_s1 = set(per.s1_id)
truth = recs[recs.owner.isin(val_s1)][["entity_id", "owner"]].rename(
    columns={"entity_id": "q_id", "owner": "s1_id"})
dec = pd.read_parquet(f"{A}/decide/decisions_val.parquet").astype({"q_id": str, "s1_id": str})
sc = pd.read_parquet(f"{A}/{stage}/scores_val.parquet").astype({"q_id": str, "s1_id": str})
cand = pd.concat([pd.read_parquet(f, columns=["q_id", "s1_id"]) for f in
                  sorted(__import__("glob").glob(f"{A}/candidates/candidates_train/*.parquet"))]
                 ).astype({"q_id": str, "s1_id": str})
got = set(zip(dec.q_id, dec.s1_id))
m = truth[[k not in got for k in zip(truth.q_id, truth.s1_id)]]
in_cand = set(zip(cand.q_id, cand.s1_id))
in_sc = sc.set_index(["q_id", "s1_id"]).p
best = sc.loc[sc.groupby("q_id").p.idxmax()].set_index("q_id").s1_id
why = []
for q, s in zip(m.q_id, m.s1_id):
    if (q, s) not in in_cand:
        why.append("not_a_candidate")
    elif (q, s) not in in_sc.index:
        why.append("not_in_scored_top_k")
    elif best.get(q) != s:
        why.append("assigned_to_other_S1")
    else:
        why.append("dropped_low_p")
m = m.assign(why=why)
print(f"\nmissed true pairs: {len(m)} of {len(truth)} ({len(m) / len(truth):.4f})")
print(m.why.value_counts().to_string())
d = m[m.why == "dropped_low_p"]
if len(d):
    p = in_sc.reindex(list(zip(d.q_id, d.s1_id)))
    print("p of dropped:", p.describe()[["mean", "25%", "50%", "75%"]].round(3).to_dict())
