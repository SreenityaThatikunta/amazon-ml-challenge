"""Profile build_features on one dev chunk (E24): which feature groups dominate.
Needs the dev slice run `devv3` (bash experiments/run_dev_v3.sh).
Usage (from student_resource/): .venv/bin/python experiments/profile_features.py"""
import json, time, cProfile, pstats
import pandas as pd
from ber.config import load_config
from ber.artifacts import StageRun
from ber.features import stage as fs, build as fb
cfg = load_config(["code/business_entity_resolution/configs/dev.yaml"], run_id="devv3")
st = StageRun.__new__(StageRun); st.cfg = cfg; st.inputs = {}
rec_path = cfg.run_dir / "normalize" / "records_train"
cand_dir = cfg.run_dir / "candidates" / "candidates_train"
fillers = json.load(open(cfg.run_dir / "features" / "fillers.json"))
brand = json.load(open(cfg.run_dir / "features" / "brand_units.json"))
ctx = fs.s1_context(cand_dir)
parts = json.load(open(cfg.run_dir / "candidates" / "parts_train.json"))
country = "US"
recs = pd.read_parquet(rec_path, columns=fs.REC_COLS + ["side", "owner"], filters=[("country", "==", country)])
recs, kidx, aidx = fs._country_records(recs, fillers, brand)
rec = recs.set_index("entity_id")
part = pd.read_parquet(cand_dir / parts[country][0]).astype({"q_id": str, "s1_id": str})
chunk = next(fs._query_chunks(part, 300_000))
d = fs.join_records(chunk, rec, ctx)
pr = cProfile.Profile(); t = time.time(); pr.enable()
fb.build_features(d, kidx, aidx)
pr.disable(); dt = time.time() - t
print(f"{len(d)} pairs in {dt:.1f}s = {len(d)/dt:,.0f} pairs/s")
ps = pstats.Stats(pr); ps.sort_stats("cumulative")
for (f, l, fn), v in sorted(ps.stats.items(), key=lambda kv: -kv[1][3])[:45]:
    if "ber" in f or fn in ("cpdist",):
        print(f"{v[3]:7.1f}s  {fn}  ({f.split('/')[-1]}:{l})")
