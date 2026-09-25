"""06 - Test set vs train, and the unseen country (France).

Questions:
  * Do the test US/India slices look like train (shift check)?
  * What do French names and addresses look like: legal forms, street-type
    abbreviations, regions, case styles, scripts?
  * Without labels: using exact-key joins between France S1 and S2/S3, what
    name and address noise do French records seem to carry, and how does it
    compare with US/India in the same test set?
"""

from collections import Counter, defaultdict

import pandas as pd

from common import (LEGAL_SUFFIXES, case_style, core_name_tokens, h,
                    has_non_ascii_latin, load_all, report, script_of, tokens,
                    vc)


def main():
    with report("06_test_france"):
        te = load_all("test")
        tr = load_all("train")

        h("Rows by (source, country): train vs test")
        print(pd.concat({"train": tr.groupby(["source", "country"]).size(),
                         "test": te.groupby(["source", "country"]).size()},
                        axis=1).fillna(0).astype(int))

        h("Name script / case / accents by (source, country), test")
        s = te.groupby(["source", "country"], group_keys=False).apply(
            lambda g: g.sample(min(len(g), 100_000), random_state=0))
        s = s.assign(script=s.business_name.map(script_of),
                     case=s.business_name.map(case_style),
                     accent=s.business_name.map(has_non_ascii_latin),
                     empty_addr=s.business_address == "")
        print(pd.crosstab([s.source, s.country], s.script, normalize="index").round(4))
        print(pd.crosstab([s.source, s.country], s.case, normalize="index").round(4))
        print(s.groupby(["source", "country"])[["accent", "empty_addr"]].mean().round(4))

        fr = te[te.country == "France"]
        for src, g in fr.groupby("source"):
            h(f"France source{src}: {len(g):,} rows")
            nt = Counter(t for x in g.business_name for t in tokens(x))
            at = Counter(t for x in g.business_address for t in tokens(x))
            print("top name tokens:", nt.most_common(40))
            print("\nlegal-form tokens:",
                  [(t, c) for t, c in nt.most_common(3000) if t in LEGAL_SUFFIXES][:20])
            print("\ntop address tokens:", at.most_common(50))
            print("\nlast address component (region / dept?):")
            vc(g.business_address.str.split(",").str[-1].str.strip(), top=20)
            print("\nsamples:")
            for _, r in g.sample(15, random_state=0).iterrows():
                print(f"  {r.entity_id:<14} {r.business_name[:40]:<40} | {r.business_address[:80]}")

        h("Unlabelled pseudo-pairs: S1 vs S2/S3 sharing an exact core name")
        te = te.assign(nk=[" ".join(core_name_tokens(x)) for x in te.business_name])
        s1 = te[te.source == 1]
        oth = te[te.source != 1]
        by_name = defaultdict(list)
        for i, c, k in zip(s1.index, s1.country, s1.nk):
            if k:
                by_name[(c, k)].append(i)
        hit = pd.Series([(c, k) in by_name for c, k in zip(oth.country, oth.nk)],
                        index=oth.index)
        print("fraction of S2/S3 whose exact core name exists in S1, by country "
              "(lower = noisier names or more distractors):")
        print(hit.groupby([oth.source, oth.country]).mean().round(4))
        # compare with train where the answer is known
        trn = tr.assign(nk=[" ".join(core_name_tokens(x)) for x in tr.business_name])
        tr_s1 = set(zip(trn[trn.source == 1].country, trn[trn.source == 1].nk))
        tro = trn[trn.source != 1]
        print("\nsame statistic on train:")
        print(pd.Series([(c, k) in tr_s1 for c, k in zip(tro.country, tro.nk)],
                        index=tro.index).groupby([tro.source, tro.country]).mean().round(4))

        print("\nFrance examples (address noise under an exact core-name match):")
        fo = oth[(oth.country == "France") & hit]
        for i, r in fo.sample(min(30, len(fo)), random_state=0).iterrows():
            a = te.loc[by_name[("France", r.nk)][0]]
            print(f"  {r.business_name[:30]:<30} | S1: {a.business_address[:45]:<45} "
                  f"| S{r.source}: {r.business_address[:45]}")

        h("Unlabelled pseudo-pairs: France S1 vs S2/S3 sharing an exact address")
        ak = lambda x: " ".join(sorted(tokens(x)))
        f1 = fr[fr.source == 1]
        by_addr = defaultdict(list)
        for i, a in zip(f1.index, f1.business_address):
            if a:
                by_addr[ak(a)].append(i)
        fo = fr[fr.source != 1]
        fo = fo[[ak(a) in by_addr for a in fo.business_address]]
        print(f"France S2/S3 records whose exact address is in France S1: "
              f"{len(fo):,} of {(fr.source != 1).sum():,}")
        for _, r in fo.sample(min(40, len(fo)), random_state=0).iterrows():
            names = [te.loc[j, "business_name"] for j in by_addr[ak(r.business_address)][:2]]
            print(f"  S1: {' / '.join(n[:30] for n in names):<45} | "
                  f"S{r.source}: {r.business_name[:40]}")


if __name__ == "__main__":
    main()
