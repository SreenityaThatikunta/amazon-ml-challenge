"""02 - Structure of the training ground truth.

Questions:
  * Cluster sizes: how many S2 / S3 records per S1, and how does that vary by
    country?
  * How common are singletons (S1 with no match)?
  * Is each S2/S3 record owned by at most one S1 (a one-owner assignment)?
  * What fraction of S2/S3 records are unmatched distractors?
  * Do matched records always share the S1 record's country?
"""

import pandas as pd

from common import h, load_gt, load_pairs, load_source, report, vc


def main():
    with report("02_ground_truth"):
        gt = load_gt()
        pairs = load_pairs()
        s1 = load_source("train", 1).set_index("entity_id")
        s2 = load_source("train", 2)
        s3 = load_source("train", 3)

        h("Coverage")
        print(f"GT rows: {len(gt):,}   S1 rows: {len(s1):,}   "
              f"GT S1 ids == S1 file ids: {set(gt.source1_entity_id) == set(s1.index)}")
        print(f"true pairs: {len(pairs):,}   unique matched ids: "
              f"{pairs.other_id.nunique():,}   "
              f"=> one owner per S2/S3 record: "
              f"{pairs.other_id.nunique() == len(pairs)}")
        known = set(s2.entity_id) | set(s3.entity_id)
        print(f"matched ids missing from source files: "
              f"{(~pairs.other_id.isin(known)).sum()}")

        h("Cluster size per S1")
        per = pairs.assign(src=pairs.other_id.str[:2]).groupby(
            ["s1_id", "src"]).size().unstack(fill_value=0)
        per = per.reindex(s1.index, fill_value=0)
        per["total"] = per.sum(axis=1)
        per["country"] = s1.country
        print("total matches per S1 (fraction of S1):")
        vc(per.total, top=15)
        print("\nS2 matches per S1:")
        vc(per["S2"], top=10)
        print("\nS3 matches per S1:")
        vc(per["S3"], top=10)
        print("\njoint (n_S2, n_S3), top 20:")
        vc(per["S2"].astype(str) + "," + per["S3"].astype(str), top=20)

        h("By country")
        g = per.groupby("country")
        print(pd.DataFrame({
            "n_s1": g.size(),
            "singleton_frac": g.total.apply(lambda x: (x == 0).mean()),
            "mean_matches": g.total.mean(),
            "mean_S2": g["S2"].mean(),
            "mean_S3": g["S3"].mean(),
            "frac_only_S2": g.apply(lambda x: ((x.S2 > 0) & (x.S3 == 0)).mean()),
            "frac_only_S3": g.apply(lambda x: ((x.S3 > 0) & (x.S2 == 0)).mean()),
        }).round(4))

        h("Unmatched S2/S3 records (distractors)")
        matched = set(pairs.other_id)
        for name, d in (("S2", s2), ("S3", s3)):
            um = ~d.entity_id.isin(matched)
            print(f"{name}: {um.sum():,} of {len(d):,} unmatched ({um.mean():.4f})")
            print(d.assign(unmatched=um).groupby("country").unmatched.mean()
                  .round(4).to_string())

        h("Country consistency of true pairs")
        other = pd.concat([s2, s3]).set_index("entity_id").country
        same = (s1.country.loc[pairs.s1_id].values ==
                other.loc[pairs.other_id].values)
        print(f"pairs with the same country label: {same.mean():.6f}")


if __name__ == "__main__":
    main()
