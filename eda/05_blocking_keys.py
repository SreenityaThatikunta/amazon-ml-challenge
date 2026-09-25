"""05 - Candidate keys: how many true pairs does each keep, and at what cost?

For a sample of true pairs, measures recall of simple exact-match keys (same
country is always required), overall and for Indic-script names.
For each key it also estimates the block size: how many S1 entities a query
record would be compared against.

This decides which cheap keys to combine with the TF-IDF / embedding
nearest-neighbour search, which is the true-pair ceiling for the v3 matcher.
"""

from collections import Counter

import numpy as np
import pandas as pd

from common import (char_ngrams, core_name_tokens, h, jaccard, load_source,
                    numbers, pairs_with_records, quantiles, report, script_of)

N_SAMPLE = 100_000


def postcodes(s):
    return {x for x in numbers(s) if len(x) in (5, 6)}


def main():
    with report("05_blocking_keys"):
        s1 = load_source("train", 1)
        # document frequencies inside S1, per country
        tok_df, num_df, pc_df, first_df = Counter(), Counter(), Counter(), Counter()
        for c, n, a in zip(s1.country, s1.business_name, s1.business_address):
            ct = core_name_tokens(n)
            tok_df.update((c, t) for t in set(ct))
            if ct:
                first_df[(c, ct[0])] += 1
            num_df.update((c, x) for x in set(numbers(a)))
            pc_df.update((c, x) for x in postcodes(a))

        df = pairs_with_records(N_SAMPLE)
        rows = []
        for r in df.itertuples():
            c = r.country
            t1, t2 = core_name_tokens(r.name1), core_name_tokens(r.name2)
            n1, n2 = set(numbers(r.addr1)), set(numbers(r.addr2))
            p1, p2 = postcodes(r.addr1), postcodes(r.addr2)
            shared_tok = set(t1) & set(t2)
            rare1 = min(t1, key=lambda t: tok_df[(c, t)]) if t1 else None
            rows.append({
                "country": c,
                "indic": script_of(r.name2) not in ("Latin", "Empty"),
                "first_tok": bool(t1 and t2 and t1[0] == t2[0]),
                "any_name_tok": bool(shared_tok),
                "rarest_s1_tok": rare1 in set(t2),
                "any_addr_num": bool(n1 & n2),
                "postcode": bool(p1 & p2),
                "num_and_name_tok": bool(n1 & n2) and bool(shared_tok),
                "name_tri>=0.3": jaccard(char_ngrams(r.name1), char_ngrams(r.name2)) >= 0.3,
                "addr_tri>=0.3": jaccard(char_ngrams(r.addr1), char_ngrams(r.addr2)) >= 0.3,
                # block sizes a query record would hit (0 = key missing)
                "bs_first_tok": first_df[(c, t2[0])] if t2 else 0,
                "bs_rarest_tok": min((tok_df[(c, t)] for t in t2), default=0),
                "bs_postcode": max((pc_df[(c, x)] for x in p2), default=0),
                "bs_any_num": sum(num_df[(c, x)] for x in n2),
            })
        k = pd.DataFrame(rows)
        k["union_A"] = k.rarest_s1_tok | k.postcode
        k["union_B"] = k.union_A | k.any_addr_num
        k["union_C"] = k.union_B | k["name_tri>=0.3"] | k["addr_tri>=0.3"]
        keys = ["first_tok", "any_name_tok", "rarest_s1_tok", "any_addr_num",
                "postcode", "num_and_name_tok", "name_tri>=0.3",
                "addr_tri>=0.3", "union_A", "union_B", "union_C"]

        h("Recall of each key on true pairs")
        print(k.groupby(["country", "indic"])[keys].mean().round(4).T)
        print("\noverall:")
        print(k[keys].mean().round(4).to_string())

        h("Block sizes hit by the S2/S3 record (S1 entities per key value)")
        for c in ("bs_first_tok", "bs_rarest_tok", "bs_postcode", "bs_any_num"):
            for country, g in k.groupby("country"):
                v = g[c][g[c] > 0]
                print(f"  {c:<14} {country:<6} present={len(v) / len(g):.3f}  "
                      f"mean={v.mean():,.0f}  {quantiles(v, (0.5, 0.9, 0.99))}")

        h("Pairs missed by union_C (need embeddings / transliteration)")
        miss = df[~k.union_C.values]
        print(f"{len(miss):,} of {len(df):,} ({len(miss) / len(df):.4f})")
        for _, r in miss.head(30).iterrows():
            print(f"  [{r.src}] {r.name1[:35]:<35} ~ {r.name2[:35]:<35} "
                  f"| {r.addr1[:35]} ~ {r.addr2[:35]}")

        h("S1 document frequency of name tokens (how discriminative names are)")
        tdf = np.array(list(tok_df.values()))
        print("token DF:", quantiles(tdf, (0.5, 0.9, 0.99, 0.999)))
        print("most common S1 name tokens:",
              [(t, n) for (c, t), n in tok_df.most_common(40)])


if __name__ == "__main__":
    main()
