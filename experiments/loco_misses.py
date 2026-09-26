"""E30: true pairs the leave-one-country-out stage 2 loses (p_loco < 0.2 while the in-domain
model has p >= 0.5), for one held-out country: examples + which stage-1 features differ most
from the true pairs it keeps.
Usage: .venv/bin/python experiments/loco_misses.py [country]   (default India)"""
import glob
import sys

import numpy as np
import pandas as pd

C = sys.argv[1] if len(sys.argv) > 1 else "India"
cty = pd.read_parquet("artifacts/v3ts/normalize/records_train",
                      columns=["entity_id", "country", "name_raw", "addr_raw", "source", "side"],
                      filters=[("side", "==", "val")]).set_index("entity_id")
a = pd.read_parquet("artifacts/v3ts_rc/rerank/scores_val.parquet")
b = pd.read_parquet("artifacts/v3ts_loco/rerank/scores_val.parquet")
m = a.merge(b[["q_id", "s1_id", "p"]], on=["q_id", "s1_id"], suffixes=("_in", "_loco"))
m = m.astype({"q_id": str, "s1_id": str})
m = m[(m.y == 1) & (cty.country.reindex(m.q_id).to_numpy() == C)]
lost = m[(m.p_in >= 0.5) & (m.p_loco < 0.2)]
kept = m[(m.p_in >= 0.5) & (m.p_loco >= 0.5)]
print(f"{C} true pairs: {len(m)}; lost by LOCO {len(lost)}; kept {len(kept)}; "
      f"p1 (stage 1) median lost {lost.p1.median():.3f} kept {kept.p1.median():.3f}")
# stage-1 features of lost vs kept
feats = []
for f in sorted(glob.glob("artifacts/v3ts/features/features_train/*.parquet")):
    d = pd.read_parquet(f).astype({"q_id": str, "s1_id": str})
    d = d.merge(pd.concat([lost.assign(lost=1), kept.assign(lost=0)])[["q_id", "s1_id", "lost"]],
                on=["q_id", "s1_id"])
    if len(d):
        feats.append(d)
F = pd.concat(feats)
num = [c for c in F.columns if c not in ("q_id", "s1_id", "y", "side", "lost")]
g = F.groupby("lost")[num].mean().T
g["diff_std"] = (g[1] - g[0]) / (F[num].std() + 1e-9)
print("\nstage-1 features, lost vs kept (mean, standardised diff), top 20:")
print(g.reindex(g.diff_std.abs().sort_values(ascending=False).index).head(20).round(3).to_string())
print("\nexamples (query | S1):")
for _, r in lost.sample(min(25, len(lost)), random_state=0).iterrows():
    print(f"p_in {r.p_in:.2f} p_loco {r.p_loco:.2f} | {cty.name_raw.get(r.q_id)} / {cty.addr_raw.get(r.q_id)}"
          f"  ||  {cty.name_raw.get(r.s1_id)} / {cty.addr_raw.get(r.s1_id)}")
