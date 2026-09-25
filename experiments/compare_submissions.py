"""Compare two matching_results.tsv files per country: matches per S1, empty rate, and
how many pairs were added / removed / kept. Sanity check before an LB upload.
Usage (from student_resource/):
  .venv/bin/python experiments/compare_submissions.py OLD.tsv NEW.tsv"""
import csv
import sys

import pandas as pd


def pairs(path: str) -> pd.DataFrame:
    d = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    d["q"] = d.matched_entity_ids.str.split(",")
    return d.explode("q").query("q != ''")[["source1_entity_id", "q"]]


old_p, new_p = sys.argv[1], sys.argv[2]
s1 = pd.read_csv("dataset/test/test_source1.tsv", sep="\t", usecols=["entity_id", "country"],
                 dtype=str, quoting=csv.QUOTE_NONE).set_index("entity_id").country
a, b = pairs(old_p), pairs(new_p)
for name, d in (("old", a), ("new", b)):
    d["country"] = d.source1_entity_id.map(s1)
n1 = s1.value_counts()
rows = []
for c in n1.index:
    ac, bc = a[a.country == c], b[b.country == c]
    ka, kb = set(zip(ac.source1_entity_id, ac.q)), set(zip(bc.source1_entity_id, bc.q))
    rows.append({"country": c, "S1": n1[c],
                 "old pairs/S1": len(ka) / n1[c], "new pairs/S1": len(kb) / n1[c],
                 "old empty": 1 - ac.source1_entity_id.nunique() / n1[c],
                 "new empty": 1 - bc.source1_entity_id.nunique() / n1[c],
                 "kept": len(ka & kb), "removed": len(ka - kb), "added": len(kb - ka),
                 "reassigned q": len({q for _, q in kb - ka} & {q for _, q in ka - kb})})
print(pd.DataFrame(rows).set_index("country").round(4).to_string())
