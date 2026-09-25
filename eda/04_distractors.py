"""04 - Unmatched S2/S3 records (distractors) and ambiguity inside S1.

These decide precision, the part F0.5 weights double:
  * Do unmatched records look like the S1 entities (same name, or same
    address with a different name), i.e. deliberate hard negatives?
  * How often do distinct S1 entities share a name or an address?
  * Side-by-side examples of hard negatives, to learn how they are made.
"""

from collections import defaultdict

import pandas as pd

from common import (core_name_tokens, h, load_all, load_pairs, report, tokens,
                    vc)

N_MATCHED_BASELINE = 300_000


def name_key(s):
    return " ".join(core_name_tokens(s))


def addr_key(s):
    return " ".join(sorted(tokens(s)))


def add_keys(d):
    d = d.copy()
    d["nk"] = [name_key(x) for x in d.business_name]
    d["ak"] = [addr_key(x) for x in d.business_address]
    return d


def main():
    with report("04_distractors"):
        recs = load_all("train")
        pairs = load_pairs()
        owner = dict(zip(pairs.other_id, pairs.s1_id))

        s1 = add_keys(recs[recs.source == 1])
        other = recs[recs.source != 1]
        is_m = other.entity_id.isin(owner)
        um = add_keys(other[~is_m])
        mt = add_keys(other[is_m].sample(N_MATCHED_BASELINE, random_state=0))
        print(f"S1: {len(s1):,}  unmatched S2/S3: {len(um):,}  "
              f"matched baseline sample: {len(mt):,}")

        name_to_s1 = defaultdict(list)
        addr_to_s1 = defaultdict(list)
        for i, c, nk, ak in zip(s1.entity_id, s1.country, s1.nk, s1.ak):
            name_to_s1[(c, nk)].append(i)
            if ak:
                addr_to_s1[(c, ak)].append(i)

        h("S1 ambiguity")
        ns = pd.Series({k: len(v) for k, v in name_to_s1.items()})
        as_ = pd.Series({k: len(v) for k, v in addr_to_s1.items()})
        print(f"S1 entities sharing their core name with another S1: "
              f"{(ns[ns > 1].sum() / len(s1)):.4f}")
        print("core-name group sizes (by S1 entity):")
        vc(ns.repeat(ns.values).clip(upper=10), top=10)
        print(f"S1 entities sharing their exact address with another S1: "
              f"{(as_[as_ > 1].sum() / len(s1)):.4f}")
        print("most common shared S1 core names:")
        print(ns.sort_values(ascending=False).head(20).to_string())

        h("Exact-key hit rates: unmatched vs matched records")
        for label, d in (("unmatched", um), ("matched(sample)", mt)):
            nh = [(c, k) in name_to_s1 for c, k in zip(d.country, d.nk)]
            ah = [(c, k) in addr_to_s1 for c, k in zip(d.country, d.ak)]
            both = [bool(set(name_to_s1.get((c, n), ())) &
                         set(addr_to_s1.get((c, a), ())))
                    for c, n, a in zip(d.country, d.nk, d.ak)]
            d = d.assign(name_hit=nh, addr_hit=ah, both_hit=both)
            print(f"\n{label}:")
            print(d.groupby(["source", "country"])[
                ["name_hit", "addr_hit", "both_hit"]].mean().round(4))
            if label == "unmatched":
                um = d
        # For matched records, does the exact-name hit point at the true owner?
        wrong = [bool(name_to_s1.get((c, n))) and owner[i] not in name_to_s1[(c, n)]
                 for i, c, n in zip(mt.entity_id, mt.country, mt.nk)]
        print(f"\nmatched records whose exact core name only hits OTHER S1s: "
              f"{pd.Series(wrong).mean():.4f}")

        h("Hard negatives: unmatched record at an S1 address, different name")
        s1_idx = s1.set_index("entity_id")
        ex = um[um.addr_hit & ~um.both_hit].sample(40, random_state=0)
        for _, r in ex.iterrows():
            s = s1_idx.loc[addr_to_s1[(r.country, r.ak)][0]]
            print(f"  S1: {s.business_name[:38]:<38} | UNM: {r.business_name[:38]:<38} "
                  f"| {r.business_address[:50]}")

        h("Hard negatives: unmatched record with an S1 name, different address")
        ex = um[um.name_hit & ~um.both_hit].sample(40, random_state=0)
        for _, r in ex.iterrows():
            s = s1_idx.loc[name_to_s1[(r.country, r.nk)][0]]
            print(f"  {r.business_name[:30]:<30} | S1: {s.business_address[:45]:<45} "
                  f"| UNM: {r.business_address[:45]}")

        h("Unmatched records with both name and address of an S1 (!)")
        ex = um[um.both_hit]
        print(f"count: {len(ex):,}")
        for _, r in ex.head(20).iterrows():
            print(f"  {r.entity_id} {r.business_name[:40]:<40} | {r.business_address[:60]}")

        h("Random unmatched records")
        for _, r in um.sample(30, random_state=1).iterrows():
            print(f"  [{r.entity_id[:2]}/{r.country}] {r.business_name[:40]:<40} "
                  f"| {r.business_address[:70]}")


if __name__ == "__main__":
    main()
