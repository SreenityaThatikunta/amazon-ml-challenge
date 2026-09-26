"""E31: why does the model distrust French one-word swaps? Stage-1 (v3) feature contributions
(LightGBM pred_contrib, log-odds) for same-address one-word-swap test pairs: France pairs with
p1 in [0.05, 0.8] vs US swap pairs with p1 >= 0.9. Prints the features whose mean contribution
differs most, with their mean values in each group.
Usage: .venv/bin/python experiments/swap_contrib.py"""
import glob
import importlib.util

import lightgbm as lgb
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("ws", "experiments/word_swap.py")
src = open("experiments/word_swap.py").read().split("# val (labels)")[0]
ws = {}
exec(src, ws)

rt = pd.read_parquet("artifacts/v3/normalize/records_test", columns=ws["COLS"]).set_index("entity_id")
dft = ws["word_df"](rt.reset_index())
model = lgb.Booster(model_file="artifacts/v3/scorer/model.txt")
feats = model.feature_name()
parts = sorted(glob.glob("artifacts/v3/features/features_test/*.parquet"))
want = {}
for c in ("France", "US"):
    rc = rt[rt.country == c]
    s = pd.concat(pd.read_parquet(f, columns=["q_id", "s1_id"]) for f in parts).astype(str)
    s = s[s.q_id.isin(rc.index)].assign(p=0.0)
    d = ws["swap_pairs"](s, rc[["name_core", "addr_norm"]], dft[c])
    want[c] = set(zip(d.q_id, d.s1_id))
rows = []
for f in parts:
    t = pd.read_parquet(f).astype({"q_id": str, "s1_id": str})
    key = list(zip(t.q_id, t.s1_id))
    for c in want:
        m = np.array([k in want[c] for k in key])
        if m.any():
            rows.append(t[m].copy().assign(grp=c))
F = pd.concat(rows, ignore_index=True)
X = np.ascontiguousarray(F[feats].to_numpy(np.float32))
F["p1"] = model.predict(X, num_iteration=model.best_iteration)
C = model.predict(X, num_iteration=model.best_iteration, pred_contrib=True)[:, :-1]
fr = (F.grp == "France").to_numpy() & (F.p1.between(0.05, 0.8)).to_numpy()
us = (F.grp == "US").to_numpy() & (F.p1 >= 0.9).to_numpy()
print(f"France swap pairs {int((F.grp == 'France').sum())} (p1 .05-.8: {fr.sum()}); US swap pairs "
      f"{int((F.grp == 'US').sum())} (p1 >= .9: {us.sum()})")
diff = pd.DataFrame({"contrib_FR": C[fr].mean(0), "contrib_US": C[us].mean(0),
                     "value_FR": F.loc[fr, feats].mean().to_numpy(), "value_US": F.loc[us, feats].mean().to_numpy()},
                    index=feats)
diff["delta"] = diff.contrib_FR - diff.contrib_US
print(diff.reindex(diff.delta.sort_values().index).head(20).round(3).to_string())
