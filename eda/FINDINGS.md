# EDA Findings

Distilled from `reports/01`–`06` (run 2026-09-25). Numbers come from train
unless stated otherwise.

## 1. Structure (02)
- **One owner per S2/S3 record** (all 7,638,365 matched ids are unique), and a
  true pair **always has the same country label** → block strictly within
  a country and assign each S2/S3 record to at most one S1.
- **US and India have identical statistics** (5.6% singletons, 3.46
  matches/S1, 1.68 S2 + 1.79 S3, 26% of S2/S3 unmatched) → one generator.
  France will very likely follow the same parameters.
- Maximum per S1: 5 from S2, 6 from S3.
- Exact duplicate records inside a source: ~20k per source (almost surely
  the same owner).

## 2. Names are ambiguous; the address is required (04)
- **50% of S1 entities share their core name with another S1**
  (e.g. "Meridian" ×572, "Cedar" ×351 in US). Only 6.8% share an exact address.
- 1.9% of true matches have an exact core name that points **only at other**
  S1s.
- True pairs whose names don't match at all: gibberish brand names
  ("Tavosoljax", "Kelogildtavo", built from syllables like tavo/jax/kelo/gild/
  zeph/nyla/delta/onyx/xylo/brix/kor/yuma/riza/belo/dova/lum/vantage) and
  **initials** ("JB" for Jai Breweries, "VPC" for Vision Precision Clinic).
  These can only be matched by address, or by an initials feature.

## 3. Hard negatives: how the distractors are made (04)
Unmatched records mimic S1 entities in two main ways:
1. **Same address, near-identical name, with an arbitrary letter edit or
   a credential/suffix swap:**
   `Verrify`→`Verrufy`, `Peridance`→`Peridanco`, `DMD`→`DMMD Corp`,
   `MD`→`JD`, `Desert League II`→`DESERT LEAGUE IC LLC`,
   `QC Multicons`→`KQC Multic0ns`, `DJ Cross`→`RJ Cross`, `Bobby Loeb`→`Brown
   Loeb`, `Erwin Research`→`Stewart Résearch`.
2. **Same name, same street, house number shifted by a small amount:**
   805→812, 5835→5838D, 16445→16449, 3355→3364, 5134→5136, 11911→11913,
   2251→2255, 319→321. Often with an extra `PMB`/`PO BOX`.

True-pair number noise is **different**: truncation (7622→762, 6665→665,
5259→525, 5123→512), zero padding (2877→002877), or dropped numbers. It is
never a small +/− shift.
→ **A key feature is how the numbers relate: equal / truncated / padded /
dropped (fine) vs shifted by a small delta (distractor).**

True typo noise is **OCR-style**: o→0, l→1, s→5, i→l, b→8, g→6 (plus
letter drops or doubles). Distractor edits are arbitrary letter changes. The
model should be able to tell the two kinds of edit apart.

Only 44 unmatched records exactly copy an S1's name and address, so label
noise is negligible.

## 4. Noise catalogue for true pairs (03)
| | India | US |
|---|---|---|
| Exact core name equal (after normalising) | 43–44% | 51–55% |
| Legal suffix changed | 48–55% | 28–32% |
| Name in an Indian script | S2 23%, S3 12% | 0 |
| Accent injected | 5–6% | 7–8% |
| S2/S3 address empty | ~3.9% | ~4.7% |
| Address numbers identical | 57–61% | 57–67% |
| Postcode present | <1% | ~11% |

- **Added name tokens:** com/www (domains), center, services, service,
  partners, trading, enterprises, the, dba, aka, formerly, fka, "doing
  business as", nee, known; titles mr/smt/shri/sri/dr; wrappers `>>`, `--`,
  `((...))`, `[...]`, `(India)`, `(France)`.
- **Dropped:** legal suffixes (limited, private, llc, inc…), `and`, generic
  descriptors (group, associates, care, clinic, health…).
- **Replacements:** limited↔ltd, inc↔incorporated, corp↔corporation, co↔company,
  inc→lnc, center→centre, plus suffix swaps (pc→inc/llc, group→llc).
- **US addresses:** street types road→rd, drive→dr, street→st, avenue→ave,
  lane→ln, court→ct, circle→cir, place→pl, boulevard→blvd, trail→trl,
  highway→hwy, terrace→ter; state code↔full name (S3 spells states out);
  junk: `##`, `null`, `N/A`, `CITY` suffixes, `Unit`→`#`.
- **India addresses:** state↔code (MH, GJ, KA, TG, RJ, HR, KL, BR, PB, OD, DL),
  orissa→odisha, kerala→keralam; **states written in Indian scripts**
  (`महाराष्ट्र`, `ગુજરાત`); house-number prefixes plot/flat/shop→h/door.
- **Indian-script names are phonetic transliterations of English words**
  (`Premier Fortune Agro`→`प्रीमियर फॉर्च्यून एग्रो`), in 9 scripts. Words repeat
  heavily (प्राइवेट लिमिटेड = private limited), so a word dictionary learned from
  training pairs plus rule-based transliteration should cover most cases.

## 5. Candidate generation (05)
| Key (same country) | Recall, Latin | Recall, Indic | Cost |
|---|---|---|---|
| any shared name token | 0.91 | 0.03 | blocks of 50–5000 |
| any shared address number | 0.78–0.82 | 0.84 | India blocks ~35k (unusable alone) |
| postcode | 0.00–0.08 | 0.00 | cheap but rare |
| name trigram Jaccard ≥ 0.3 | 0.94–0.96 | 0.00 | needs a nearest-neighbour index |
| addr trigram Jaccard ≥ 0.3 | 0.86–0.93 | 0.87 | needs a nearest-neighbour index |
| **name OR address trigram (union C)** | **0.999** | **0.980** | |

→ Use **name TF-IDF top-k + address TF-IDF top-k per country**, plus a
**transliterated-name** index for Indic scripts. Exact token blocking is
too coarse: names are generic, and Indian house numbers are tiny.
The 0.2% missed are mostly Indic names with heavily rewritten addresses.

## 6. Test set and France (01, 06)
- The test share of India rises to 47% (train 40%). US/India test
  distributions otherwise match train.
- France S1: **~18 cities in 3 regions** (Hauts-de-France, Nouvelle-Aquitaine,
  Pays de la Loire). Legal forms SARL/SAS/EURL/SA/SASU/SCI/SNC/EI; names built
  from French word combinations (Club, Ecole, Amicale, Comite, Maison…).
- **S2/S3 swap the region for the département:** Nord / Pas-de-Calais ↔
  Hauts-de-France, Gironde ↔ Nouvelle-Aquitaine, Loire-Atlantique ↔ Pays de
  la Loire. City names are sometimes upper case or abbreviated
  (`ST.-HERBLAIN`).
- Street-type abbreviations: rue→`R`/`R.`, avenue→`AV`/`Av.`,
  boulevard→`BD`/`BLVD`, cours→`CRS`, route→`Rte.`, esplanade→`ESPL`;
  number prefixes `N°`, `Nº`, `NO.`, `#`; `bis`/`ter` suffixes.
- Accents are common (S1 16%, S2/S3 24%), so strip them before comparing.
- **The same generator is used** (gibberish names like Lyrabrix/Yumajaxx, aka/
  formerly, .com, OCR digits, typos, doubled `Frèrês`). Exact-core-name hit
  rate in test is France 0.51–0.53 vs US 0.43–0.45 vs India 0.31–0.34, so
  **France looks no harder than US** once it is normalised.

## 7. Implications for the v3 pipeline
1. Normalisation tables (sections 4 and 6) do much of the work and should be
   shared by the index and the model.
2. Candidates: character TF-IDF on names and on addresses, run per country,
   combined; plus transliterated names. Target ≥ 99.8% recall on validation.
3. The matcher must be sensitive to **exact** differences: number relations
   (truncated vs shifted), one-letter edits (OCR vs arbitrary), credential
   swaps. Give the LLM explicit hints: pre-computed number-relation and edit-
   type tags in the prompt (Ditto-style knowledge injection), not just raw
   strings.
4. Selection: the LLM chooses the owner among each S2/S3 record's top
   candidates, or "none". Constraints: one owner per record, ≤5 S2 / ≤6 S3 per
   S1.
5. Compute: ~10M queries. The LLM should only see cases a cheap scorer can't
   settle confidently.
