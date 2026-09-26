"""Best candidate per test query (model `main`) with its house-number relation and p, by
country; writes experiments/logs/top1.parquet. Used for the France shifted/truncated number
analysis (progress.md, T7.1). Run from student_resource/."""
import glob, pandas as pd, numpy as np
recs = pd.concat([pd.read_parquet(f, columns=["entity_id","country"], filters=[("source","==",1)]) for f in sorted(glob.glob("artifacts/main/normalize/records_test/*.parquet"))])
c = recs.set_index("entity_id").country
out=[]
for sf, ff in zip(sorted(glob.glob("artifacts/main/scorer/scores_test/*.parquet")), sorted(glob.glob("artifacts/main/features/features_test/*.parquet"))):
    s = pd.read_parquet(sf); s["q_id"]=s.q_id.astype(str); s["s1_id"]=s.s1_id.astype(str)
    b = s.loc[s.groupby("q_id").p.idxmax()]
    b["country"]=b.s1_id.map(c)
    f = pd.read_parquet(ff, columns=["q_id","s1_id","hn_rel","n_core_eq","l_conflict","box_q"]); f["q_id"]=f.q_id.astype(str); f["s1_id"]=f.s1_id.astype(str)
    out.append(b.merge(f, on=["q_id","s1_id"]))
t = pd.concat(out)
t["band"]=pd.cut(t.p,[-1,.05,.5,.95,2],labels=["<.05",".05-.5",".5-.95",">.95"])
names=["missing","equal","padded","truncated","shifted","different"]
t["rel"]=t.hn_rel.map(dict(enumerate(names)))
print("share of each best-candidate number relation by p band\n", t.groupby(["country","rel"]).band.value_counts(normalize=True).unstack().round(3))
x=t[t.rel=="shifted"]
print("\nshifted top1, name exact: p band\n", x[x.n_core_eq==1].groupby("country").band.value_counts(normalize=True).unstack().round(3))
print("\nshifted top1 with BP/PMB box: p band\n", x[x.box_q==1].groupby("country").band.value_counts(normalize=True).unstack().round(3))
t.to_parquet("experiments/logs/top1.parquet")
