"""True val pairs that candidate generation missed (run `main`): counts by country /
source / script / empty address, and random examples. Streams candidate parts.
Usage (from student_resource/): .venv/bin/python experiments/candidate_misses.py [run]"""
import glob
import sys

import pandas as pd

run = sys.argv[1] if len(sys.argv) > 1 else "main"
A = f"artifacts/{run}"
q = pd.read_parquet(f"{A}/split/queries.parquet", columns=["entity_id", "owner", "side", "country"])
q = q[(q.side == "val") & (q.owner != "")]
truth = set(zip(q.entity_id, q.owner))
found = set()
for f in sorted(glob.glob(f"{A}/candidates/candidates_train/*.parquet")):
    c = pd.read_parquet(f, columns=["q_id", "s1_id"]).astype({"q_id": str, "s1_id": str})
    c = c[c.q_id.isin(q.entity_id)]
    found |= set(zip(c.q_id, c.s1_id)) & truth
miss = q[[k not in found for k in zip(q.entity_id, q.owner)]]
print(f"val true pairs {len(q):,}; missed by candidates {len(miss):,} ({len(miss)/len(q):.4f})")
r = pd.concat([pd.read_parquet(f, columns=["entity_id", "name_raw", "addr_raw", "name_indic",
                                           "addr_norm", "name_core"])
               for f in sorted(glob.glob(f"{A}/normalize/records_train/*.parquet"))]).set_index("entity_id")
m = miss.assign(indic=r.name_indic.reindex(miss.entity_id).to_numpy(),
                empty_addr=(r.addr_norm.reindex(miss.entity_id) == "").to_numpy(),
                src=miss.entity_id.str[:2].to_numpy())
base = q.assign(indic=r.name_indic.reindex(q.entity_id).to_numpy(),
                empty_addr=(r.addr_norm.reindex(q.entity_id) == "").to_numpy())
print("\nmiss rate by country:\n", (m.groupby("country").size() / q.groupby("country").size()).round(4).to_string())
print("\nshare of misses: indic", round(m.indic.mean(), 3), "(base", round(base.indic.mean(), 3), ")",
      " empty addr", round(m.empty_addr.mean(), 3), "(base", round(base.empty_addr.mean(), 3), ")")
for x in m.sample(20, random_state=0).itertuples():
    print(f"\nQ {x.entity_id}: {r.name_raw[x.entity_id]} | {r.addr_raw[x.entity_id]}\n"
          f"S {x.owner}: {r.name_raw[x.owner]} | {r.addr_raw[x.owner]}")
