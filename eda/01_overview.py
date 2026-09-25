"""01 - Overview of every source file (train and test).

Questions:
  * How big is each source, and how are countries split?
  * How often are fields empty or duplicated?
  * What scripts and letter cases appear in names, per source and country?
  * How long are records (this sizes the LLM prompts for the v3 matcher)?
"""

from common import (case_style, h, load_source, quantiles, report,
                    script_of, vc)


def describe(split, n):
    d = load_source(split, n)
    h(f"{split} source{n}: {len(d):,} rows")
    print("countries:")
    vc(d.country, normalize=False)
    print(f"duplicate entity_id: {d.entity_id.duplicated().sum()}")
    print(f"exact duplicate (name, address, country) rows: "
          f"{d.duplicated(['business_name', 'business_address', 'country']).sum():,}")
    print(f"empty name: {(d.business_name == '').mean():.4f}   "
          f"empty address: {(d.business_address == '').mean():.4f}")

    for country, g in d.groupby("country"):
        s = g.sample(min(len(g), 200_000), random_state=0)
        print(f"\n-- {country} (n={len(g):,})")
        print(" name chars   :", quantiles(s.business_name.str.len()))
        print(" addr chars   :", quantiles(s.business_address.str.len()))
        print(" addr commas  :", quantiles(s.business_address.str.count(",")))
        print(" name script:")
        vc(s.business_name.map(script_of), top=10)
        print(" name case style:")
        vc(s.business_name.map(case_style), top=5)
        print(" addr case style:")
        vc(s.business_address.map(case_style), top=5)
        print(f" names with digits: {s.business_name.str.contains(r'[0-9]').mean():.4f}   "
              f"names that look like domains: "
              f"{s.business_name.str.contains(r'[.](com|in|net|org|fr)$', case=False).mean():.4f}")
        # LLM sizing: ~4 chars/token for Latin, much worse for Indic scripts
        rec = s.business_name + " | " + s.business_address
        print(" record chars (name|addr):", quantiles(rec.str.len()))
        print(" samples:")
        for _, r in g.sample(5, random_state=1).iterrows():
            print(f"   {r.entity_id:<14} {r.business_name[:45]:<45} | {r.business_address[:80]}")


def main():
    with report("01_overview"):
        for split in ("train", "test"):
            for n in (1, 2, 3):
                describe(split, n)


if __name__ == "__main__":
    main()
