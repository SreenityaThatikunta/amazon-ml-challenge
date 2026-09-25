"""03 - What noise separates a true S2/S3 record from its S1 record?

Samples true pairs and measures, by source and country:
  * name: exact / normalised equality, script change, case style, accents,
    legal-suffix changes, word reorders, added/dropped/replaced tokens
    (these reveal abbreviation tables and OCR-style typo maps)
  * address: emptiness, reordering, number agreement (house-number
    truncation, zero padding), postcode presence, token overlap
  * similarity distributions, plus hardest-looking true pairs as examples
"""

from collections import Counter

import pandas as pd

from common import (LEGAL_SUFFIXES, case_style, char_ngrams, h,
                    has_non_ascii_latin, jaccard, norm, numbers,
                    pairs_with_records, quantiles, report, script_of, tokens)

N_SAMPLE = 200_000


def name_features(r):
    t1, t2 = tokens(r.name1), tokens(r.name2)
    c1 = [t for t in t1 if t not in LEGAL_SUFFIXES]
    c2 = [t for t in t2 if t not in LEGAL_SUFFIXES]
    return pd.Series({
        "raw_eq": r.name1 == r.name2,
        "lower_eq": r.name1.lower() == r.name2.lower(),
        "norm_eq": t1 == t2,
        "core_eq": c1 == c2,
        "reorder": sorted(t1) == sorted(t2) and t1 != t2,
        "suffix_changed": set(t1) - set(c1) != set(t2) - set(c2),
        "script2": script_of(r.name2),
        "case2": case_style(r.name2),
        "accent_added": has_non_ascii_latin(r.name2) and not has_non_ascii_latin(r.name1),
        "tok_jacc": jaccard(set(t1), set(t2)),
        "core_jacc": jaccard(set(c1), set(c2)),
        "tri_jacc": jaccard(char_ngrams(r.name1), char_ngrams(r.name2)),
        "n_tok1": len(t1), "n_tok2": len(t2),
    })


def addr_features(r):
    t1, t2 = tokens(r.addr1), tokens(r.addr2)
    n1, n2 = numbers(r.addr1), numbers(r.addr2)
    s1, s2 = set(n1), set(n2)
    if not r.addr2.strip():
        num = "addr2 empty"
    elif not n1:
        num = "no numbers in s1"
    elif s1 == s2:
        num = "same set"
    elif s2 < s1:
        num = "other drops some"
    elif {x.lstrip("0") for x in s1} == {x.lstrip("0") for x in s2}:
        num = "zero-padding only"
    elif any(b != a and a.startswith(b) for a in n1 for b in n2):
        num = "truncated number"
    elif s1 & s2:
        num = "partial overlap"
    else:
        num = "disjoint"
    return pd.Series({
        "addr2_empty": not r.addr2.strip(),
        "addr_norm_eq": t1 == t2,
        "addr_reorder": sorted(t1) == sorted(t2) and t1 != t2,
        "addr_tok_jacc": jaccard(set(t1), set(t2)),
        "addr_tri_jacc": jaccard(char_ngrams(r.addr1), char_ngrams(r.addr2)),
        "num_relation": num,
        "has_postcode1": any(len(x) in (5, 6) for x in n1),
        "has_postcode2": any(len(x) in (5, 6) for x in n2),
        "case_addr2": case_style(r.addr2),
    })


def token_diffs(df, top=40):
    """Counters of added, dropped and 1-for-1 replaced name tokens."""
    added, dropped, repl = Counter(), Counter(), Counter()
    for a, b in zip(df.name1, df.name2):
        t1, t2 = tokens(a), tokens(b)
        s1, s2 = set(t1), set(t2)
        d, ad = s1 - s2, s2 - s1
        dropped.update(d)
        added.update(ad)
        if len(d) == 1 and len(ad) == 1:
            repl[(next(iter(d)), next(iter(ad)))] += 1
    return added.most_common(top), dropped.most_common(top), repl.most_common(top)


def addr_token_repl(df, top=40):
    repl = Counter()
    for a, b in zip(df.addr1, df.addr2):
        s1, s2 = set(tokens(a)), set(tokens(b))
        d, ad = s1 - s2, s2 - s1
        if len(d) == 1 and len(ad) == 1:
            repl[(next(iter(d)), next(iter(ad)))] += 1
    return repl.most_common(top)


def char_substitutions(df, top=40):
    """Single-character substitutions between same-length name tokens
    (reveals OCR / typo maps such as l->i or g->6)."""
    subs = Counter()
    for a, b in zip(df.name1, df.name2):
        for x, y in zip(norm(a).split(), norm(b).split()):
            if len(x) == len(y) and x != y:
                diff = [(p, q) for p, q in zip(x, y) if p != q]
                if len(diff) == 1:
                    subs[diff[0]] += 1
    return subs.most_common(top)


def main():
    with report("03_pair_noise"):
        df = pairs_with_records(N_SAMPLE)
        print(f"sampled true pairs: {len(df):,}")
        df = pd.concat([df, df.apply(name_features, axis=1),
                        df.apply(addr_features, axis=1)], axis=1)

        bool_cols = ["raw_eq", "lower_eq", "norm_eq", "core_eq", "reorder",
                     "suffix_changed", "accent_added", "addr2_empty",
                     "addr_norm_eq", "addr_reorder", "has_postcode1",
                     "has_postcode2"]
        h("Rates by (source, country)")
        print(df.groupby(["src", "country"])[bool_cols].mean().round(4).T)

        h("Similarity distributions by (source, country)")
        for key, g in df.groupby(["src", "country"]):
            print(f"\n{key}  n={len(g):,}")
            for c in ("tok_jacc", "core_jacc", "tri_jacc", "addr_tok_jacc",
                      "addr_tri_jacc"):
                print(f"  {c:<14}", quantiles(g[c]))

        h("Script of the S2/S3 name, by (source, country)")
        print(pd.crosstab([df.src, df.country], df.script2, normalize="index").round(4))
        h("Case style of the S2/S3 name / address")
        print(pd.crosstab([df.src, df.country], df.case2, normalize="index").round(4))
        print(pd.crosstab([df.src, df.country], df.case_addr2, normalize="index").round(4))

        h("Address number relation (S1 vs S2/S3)")
        print(pd.crosstab([df.src, df.country], df.num_relation,
                          normalize="index").round(4).T)

        latin = df[df.script2 == "Latin"]
        for country, g in latin.groupby("country"):
            added, dropped, repl = token_diffs(g)
            h(f"Name token diffs ({country}, Latin-script only)")
            print("top ADDED tokens:  ", added)
            print("\ntop DROPPED tokens:", dropped)
            print("\ntop 1-for-1 REPLACEMENTS (s1 -> other):")
            for (a, b), c in repl:
                print(f"  {a:>18} -> {b:<18} {c}")
            h(f"Single-char substitutions in names ({country})")
            print(char_substitutions(g))
            h(f"Address 1-for-1 token replacements ({country})")
            for (a, b), c in addr_token_repl(g):
                print(f"  {a:>18} -> {b:<18} {c}")

        h("Hardest-looking TRUE pairs (lowest name trigram Jaccard, Latin)")
        for country, g in latin.groupby("country"):
            print(f"\n-- {country}")
            for _, r in g.nsmallest(25, "tri_jacc").iterrows():
                print(f"  [{r.src}] {r.name1[:40]:<40} ~ {r.name2[:40]:<40} "
                      f"| {r.addr1[:40]} ~ {r.addr2[:40]}")

        h("Random Indic-script pairs")
        ind = df[df.script2 != "Latin"]
        for _, r in ind.sample(min(25, len(ind)), random_state=0).iterrows():
            print(f"  [{r.src}/{r.script2}] {r.name1[:40]:<40} ~ {r.name2[:40]}")


if __name__ == "__main__":
    main()
