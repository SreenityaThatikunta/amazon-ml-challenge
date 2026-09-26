"""E31: records a run is unsure about on test (top p2 in [lo, hi]) for one country, with their
top-2 S1 candidates, raw fields, plus the share of such records per country (label-free
France diagnosis: France has ~3x the ambiguous records of US/India).
Usage: .venv/bin/python experiments/ambiguous_dump.py [run] [country] [n]  (default v3 France 40)"""
import glob
import sys

import pandas as pd

RUN = sys.argv[1] if len(sys.argv) > 1 else "v3"
C = sys.argv[2] if len(sys.argv) > 2 else "France"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 40
LO, HI = 0.2, 0.8
sc = pd.concat(pd.read_parquet(f, columns=["q_id", "s1_id", "p", "p1"])
               for f in sorted(glob.glob(f"artifacts/{RUN}/rerank/scores_test/*.parquet")))
sc = sc.astype({"q_id": str, "s1_id": str}).sort_values(["q_id", "p"], ascending=[True, False])
rec = pd.read_parquet(f"artifacts/{RUN}/normalize/records_test",
                      columns=["entity_id", "source", "country", "name_raw", "addr_raw"]).set_index("entity_id")
top = sc.drop_duplicates("q_id")
top["country"] = rec.country.reindex(top.q_id).to_numpy()
n_q = rec[rec.source != 1].country.value_counts()
share = (top[(top.p >= LO) & (top.p <= HI)].country.value_counts() / n_q).round(4)
print(f"records with top p2 in [{LO}, {HI}] per S2/S3 record:\n{share.to_string()}\n")
amb = top[(top.country == C) & (top.p >= LO) & (top.p <= HI)].sample(N, random_state=1)
for q in amb.q_id:
    c = sc[sc.q_id == q].head(2)
    print(f"Q  {rec.name_raw[q]} | {rec.addr_raw[q]}")
    for _, r in c.iterrows():
        print(f"   p2 {r.p:.2f} (p1 {r.p1:.2f})  {rec.name_raw[r.s1_id]} | {rec.addr_raw[r.s1_id]}")
