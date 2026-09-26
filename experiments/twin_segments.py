"""E31: F0.5 on val_ts by S1 segment, for the France-structure hypothesis: S1 with an
exact-address twin (another S1 of the country with the same normalised address; France
13.3% vs US 4.6% / India 5.2%, E3) and S1 whose core name contains a word of its own
address (city in the name, France 48% of ambiguous runner-ups). Also the segment shares on
test per country, to weight the val_ts segment scores into a France estimate.
Usage: .venv/bin/python experiments/twin_segments.py [run]   (default v3ts_rc)"""
import sys

import numpy as np
import pandas as pd

RUN = sys.argv[1] if len(sys.argv) > 1 else "v3ts_rc"
cols = ["entity_id", "source", "country", "name_core", "addr_norm"]


def segments(s1: pd.DataFrame) -> pd.DataFrame:
    s1 = s1.copy()
    a = s1.addr_norm.where(s1.addr_norm != "")
    s1["addr_twin"] = a.map(a.value_counts()).fillna(1).to_numpy() > 1
    s1["city_in_name"] = [bool(set(n.split()) & {t for t in ad.split() if len(t) >= 4 and not t.isdigit()})
                          for n, ad in zip(s1.name_core, s1.addr_norm)]
    s1["name_twin"] = s1.name_core.map(s1.name_core.value_counts()).to_numpy() > 1
    return s1


tr = pd.read_parquet("artifacts/v3ts/normalize/records_train", columns=cols + ["side"],
                     filters=[("source", "==", 1)])
seg = pd.concat([segments(g) for _, g in tr.groupby("country")]).set_index("entity_id")
ent = pd.read_parquet(f"artifacts/{RUN}/decide/val_per_entity.parquet").set_index("s1_id")
ent = ent.join(seg[["addr_twin", "city_in_name", "name_twin"]], how="left")
print(f"val_ts per-S1 F0.5 by segment ({RUN}):")
for c in ("addr_twin", "city_in_name", "name_twin"):
    print(ent.groupby(["country", c]).f05.agg(["mean", "size"]).round(5).to_string(), "\n")

te = pd.read_parquet("artifacts/v3/normalize/records_test", columns=cols, filters=[("source", "==", 1)])
tseg = pd.concat([segments(g) for _, g in te.groupby("country")])
print("test S1 segment shares:")
print(tseg.groupby("country")[["addr_twin", "city_in_name", "name_twin"]].mean().round(4).to_string())
vseg = seg[seg.index.isin(ent.index)]
print("\nval_ts S1 segment shares:")
print(vseg.groupby("country")[["addr_twin", "city_in_name", "name_twin"]].mean().round(4).to_string())
