"""Build leaderboard probe files (plan.md §0, A1): per-country edits of a
matching_results.tsv, so the LB delta isolates one country's score.

  empty:  every S1 of COUNTRY gets an empty prediction.
          LB(base) - LB(probe) = w_c * (F_c - s_c), with w_c the country's share of
          test S1 and s_c its share of true singletons (~0.05), so
          F_c ~= (LB(base) - LB(probe)) / w_c + s_c.
  swap:   predictions of COUNTRY's S1 come from DONOR, the rest from BASE.

Usage (from student_resource/):
  .venv/bin/python experiments/make_probe.py empty BASE.tsv France OUT.tsv
  .venv/bin/python experiments/make_probe.py swap  BASE.tsv France OUT.tsv DONOR.tsv
Prints the country weights; run utils/validate_submission.py on OUT before uploading."""
import csv
import sys

import pandas as pd


def read(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)


def main() -> None:
    mode, base_p, country, out_p = sys.argv[1:5]
    s1 = pd.read_csv("dataset/test/test_source1.tsv", sep="\t", usecols=["entity_id", "country"],
                     dtype=str, quoting=csv.QUOTE_NONE).set_index("entity_id").country
    if country not in set(s1):
        sys.exit(f"unknown country {country!r}; test has {sorted(set(s1))}")
    base = read(base_p)
    in_c = base.source1_entity_id.map(s1).eq(country)
    if mode == "empty":
        base.loc[in_c, "matched_entity_ids"] = ""
    elif mode == "swap":
        donor = read(sys.argv[5]).set_index("source1_entity_id").matched_entity_ids
        base.loc[in_c, "matched_entity_ids"] = (base.source1_entity_id[in_c].map(donor)
                                                .fillna("").to_numpy())
    else:
        sys.exit(f"unknown mode {mode!r}")
    base.to_csv(out_p, sep="\t", index=False, quoting=csv.QUOTE_NONE)
    w = s1.value_counts(normalize=True).round(4).to_dict()
    print(f"{mode} {country}: {int(in_c.sum())} S1 edited -> {out_p}; test S1 weights {w}")


if __name__ == "__main__":
    main()
