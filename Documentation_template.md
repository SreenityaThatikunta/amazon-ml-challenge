# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** pootato
**Team Members:** Sreenitya Thatikunta, Rudraksha Dev Sharma
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

A four-stage cascade that never compares all pairs: normalisation tuned to the noise we
measured in the data, region-blocked character-trigram TF-IDF retrieval (24.6 S1
candidates per S2/S3 record, 99.05 % of true pairs), a LightGBM pair scorer with 102
features that also acts as a learned filter (**1.08 candidates per record kept, 99.97 %
of the retrieved true pairs**), and a second LightGBM that re-ranks the survivors with
*group context*: how many other records claim the same S1, how strongly and from which
source. A per-S1 expected-F0.5 rule (one owner per record) produces the final links.
Validation macro F0.5 is **0.98895** on a geographic hold-out (10 % of the S1 entities,
whole states), up from 0.98211 for our first submission. Everything runs on one laptop
(16 GB RAM; optional NVIDIA GPU for the retrieval step) with MIT/BSD/Apache
dependencies and no pretrained model.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings that shaped the design (EDA scripts in `eda/`, summary in `eda/FINDINGS.md`,
v2 checks in `progress.md`):

- **Structure.** Every S2/S3 record belongs to at most one S1 (all 7.6 M matched ids are
  unique) and true pairs never cross countries → one owner per record, blocking within
  a country. 5.6 % of S1 are singletons; S1 have 3.46 matches on average (≤ 5 from S2,
  ≤ 6 from S3). US and India share one generator; France (test only) follows the same one.
- **Names are ambiguous, addresses decide.** 50 % of S1 share their core name with
  another S1 ("Meridian" × 572), only 6.8 % share an exact address.
- **Name noise** (true pairs): legal-form swaps and typos in legal words (`PRlVATE`,
  `Priavte`), OCR digits (`Hea1th 5ervices`), arbitrary letter typos (`Mitllls`),
  word-order transpositions, doubled words, aliases (`aka`, `dba`, `formerly`, `nee`),
  domains (`bwford.com`), injected record ids / phone numbers (`(ID: 63317)`, `#32038`,
  `- 9334025793`, ~1 % of US/India records), **filler words** appended per country
  (US `Riverside/Westgate/Holdings/Services`, India `Smt/Shri/Sweets/Infratech`, France
  `Participations/Holding/Distribution/Développement`), Indian-script transliterations
  (23 % of Indian S2 names), and **made-up brand names replacing the whole name**
  (`Grand Physical Therapy` → `Jaxdrexquo`, 1.6 % of true pairs).
- **Address noise:** abbreviations (Rd/Road, R./Rue), state ↔ code ↔ Indian-script
  state ↔ French département, reordered components, dropped units, zero padding,
  truncated house numbers (7622 → 762), digit typos in leading positions (394 → 894), and
  in India a **fake house number prepended** (`H.NO 937 33, …`). 4 % of S2/S3 addresses
  are empty.
- **Distractors** (unmatched records): copies of an S1 with a one-letter name edit,
  a credential swap (MD → JD) or the house number **shifted by a few units**
  (5835 → 5838), often with an added PO box.
- **Test differs from train:** France is new, and test has ~5.7 S2/S3 records per S1
  vs 4.7 in train (≈ 40 % unmatched vs 26 %). Our first leaderboard score (0.975) was
  ≈ 0.007 below its validation score (0.982).
- **France has a name noise US/India don't:** at the same address and house number, the
  *generic* word of a name is swapped (`Ski College SARL` → `Ski Comite SARL`,
  `BYH Club SAS` → `BYH Loisirs SAS`) or replaced by `& Fils` / `(France)`. Such
  generic-word swaps are 6.6 % of French same-street pairs but 0.1 % (US) and 1.5 % (India),
  where they are mostly decoys; the model, which never saw the pattern, gives the French
  ones p ≈ 0.4. Their house-number pattern (below) says ~80 % are true copies (likely the
  French form of the "DBA/trade name" noise in the problem statement).

### 2.2 Solution Strategy

**Approach Type:** Blocking + learned filter + group-aware re-ranker + expected-F0.5 decision.
**Core Innovation:** a *cascade* where the stage-1 classifier doubles as the final
blocking filter (tiny candidate set without losing recall), and a stage-2 re-ranker
trained cross-fitted on complete S1 groups that reasons about competing claims on the
same S1, the information a pairwise matcher cannot see.

```
records ─► [0] normalise ─► [1] TF-IDF top-k (region-blocked, 24.6/record)
        ─► [2] 102 pair features ─► stage-1 LightGBM p1 ─► keep p1 ≥ 0.001 (1.08/record)
        ─► [3] stage-2 LightGBM p2 (group context) ─► [4] one owner + expected F0.5 per S1
```

Validation: whole regions (US and Indian states) are held out (10 % of S1 per country),
so lookalike distractors, which sit at nearby addresses, stay on the same side as the
entities they imitate. Val queries (S2/S3) follow their owner; all val records are scored,
so val F0.5 includes blocking losses, singletons and every stage, like the leaderboard.

---

## 3. Candidate Generation (Blocking)

- **Normalisation first** (`src/ber/normalize/`): Latin accents stripped, `&/+/et` → `and`,
  aliases split (pre/alt parts kept), domains and handles unwrapped, injected ids and phone
  numbers dropped, OCR digit → letter fixes in mixed tokens, single-letter runs joined
  (`L.L.C.` → `llc`), legal forms canonicalised and moved out of the core (one typo or
  one adjacent swap tolerated in long legal words), stopwords and honorifics dropped,
  doubled words collapsed. Indian-script tokens: a dictionary learned from training pairs
  (1,164 words), then rule-based romanisation snapped to the S1 vocabulary by consonant
  skeleton (95.8 % exact on held-out tokens). Addresses: placeholders dropped, region
  components (state names/codes, Indian-script states, French région/département)
  extracted to a `region` field, street types canonicalised per country, unit/house
  prefixes dropped, zero padding removed; raw digit strings kept for number features.
- **Blocking keys:** three character-trigram TF-IDF indexes over S1 per country: core
  name, address words (order-free) and name + address. Each S2/S3 record takes the union
  of the top 10 of each (top 30 names when its address is empty).
- **Region blocking:** a record searches only the S1 of its region's neighbourhood,
  learned from training pairs (e.g. AP ↔ TG, Telangana addresses written as Andhra
  Pradesh); records without a detectable region search the whole country.
- **Learned filter (stage 1):** the pair scorer (section 4) keeps pairs with p1 ≥ 0.001.
  This is the last filter before the final model and defines `candidate_pairs.tsv`.
- **Candidate pairs generated (test):** TF-IDF: 243,753,102 pairs (24.4 per record);
  after the stage-1 filter (= `candidate_pairs.tsv`): 9,649,852 pairs (0.97 per record),
  5.57 per S1 entity (the S1 has 3.4 true matches on average).
- **How true matches were not lost:** recall measured on the validation side at every
  step: TF-IDF 99.05 % of true pairs (US 99.3 %, India 98.2–99.2 %), the stage-1 filter
  keeps 99.97 % of those (trade-off table in `artifacts/<run>/rerank/manifest.json`:
  0.001 → 1.08 pairs/record, 0.01 → 0.88 pairs/record at 99.78 %). The remaining misses
  are dominated by address-less records whose name dozens of S1 share.
- **Scale:** streaming per country and per region block; one country's records in memory
  at a time; optional GPU top-k (CuPy, cuSPARSE) for large blocks. Test retrieval: 78 min
  on a laptop with an RTX 4050 (2.4 h CPU-only for train).

---

## 4. Matching Model

**Features used (stage 1, 102):**
- Name features: exact core/compact equality; Levenshtein, substitutions vs indels vs
  transpositions, ratio, token-set/sort ratio, partial ratio, Jaro-Winkler; alias parts;
  initials (`VPC` ↔ `Vision Precision Clinic`); first/last token; legal-form agreement and
  conflict; domain / alias / Indic flags; share of tokens unknown to the S1 vocabulary;
  how many S1 share the exact core name. **Filler-free name key** (sorted unique core
  tokens without the injected filler words, learned without labels as tokens ≥ 2× more
  frequent in S2/S3 names than in S1 names): equality, Levenshtein, ratio, token-set
  ratio, **IDF-weighted recall/precision**, largest IDF of an unmatched token on each
  side. **Made-up-brand flag** (name is a concatenation of ~45 learned syllables: covers
  96 % of brand replacements, fires on 0.03 % of S1 names).
- Address features: equality, token-set / ratio / partial ratio; region agreement /
  conflict / missing; house-number relation (equal / padded / truncated / shifted /
  different) and log difference; number-set overlap; **S1 house number present anywhere
  in the record's numbers (and vice versa), best relation to any number, digit edit
  distance, position of the first differing digit**; PO box / PMB flags;
  **IDF-weighted overlap of address words**.
- Retrieval and context: the three TF-IDF cosines and ranks; for 8 key similarities the
  margin over the record's best *other* candidate, the gap to its best and the rank;
  candidates per record and per S1.

**Stage-2 features (re-ranker, 127):** p1 and logit; record context (rank, best/sum/count
of the record's other p1); **S1 context** (rank of the record among all records listing
the S1, best/sum/count of competing p1, the same within the record's source, and
"claims": records whose best S1 this is, their expected count and best competitor);
all 102 stage-1 features passed through (25 of them: val 0.98748; all: 0.98895).
**Domain normalisation:** 16 count-like features (per-S1 and per-record counts, name
frequency) are divided by their mean and 4 IDF-like ones by their 99th percentile, within each
country and split, because test has a different S1 density than train (section 5).

**Model type:** LightGBM (MIT) binary classifiers. Stage 1: 127 leaves, lr 0.05, early
stopping on 5 % of val records (best iteration 346), trained on 4.5 M pairs from 10 % of
the featurised train-side records. Stage 2: 63 leaves, trained on the val side only
(1.12 M pairs, complete S1 groups) with **4-fold cross-fitting** grouped by the record's
top S1: every val pair gets an out-of-fold p2 (honest validation), test pairs get the mean
of the 4 fold models. No pretrained model, far below the 8 B parameter limit.

**Threshold selection method:** one owner per record (its highest p2), then per S1 the
prefix of its records (sorted by p2) that maximises the expected F0.5
(≈ 1.25·Σp / (0.25·Σp_all + k), vs P(no match) = Π(1 − p) for the empty prediction), after
dropping p2 < min_p. min_p = 0.3 chosen on validation; the curve is flat (within 0.0001
for 0.02–0.5). Exact duplicate records share one owner.

**Prior correction per name-change kind (`decide.prior_shift = anchor`,
`src/ber/decide/anchor.py`).** A model trained on US/India cannot know how often an
unseen kind of name change is a true copy in a new country. We estimate it from the
unlabelled test data with an *anchor*: given the match status, the house-number
relation of a record and its S1 (equal / shifted / missing / ...) does not depend on how
the name was changed (on validation the true copies' profile is the same for exact
names, typos, generic-word swaps and other changes; decoys are mostly *shifted*). So for
each country and name-change kind (exact key, one rare word swapped = typo, one generic
word swapped, other), the kind's house-number histogram is a mixture of the true-copy and
decoy profiles (both measured on the exact-name kind, with p2 as soft labels), which gives
its share of true copies π by least squares. If odds(π) / odds(mean p2 of the kind),
de-biased by the same ratio measured on validation (where the model is calibrated), is
beyond 2× and the kind has ≥ 2,000 pairs, the kind's p2 are rescaled by that odds ratio
(capped at 6). On validation nothing is corrected (F0.5 unchanged). Checks on
validation: no correction fires under eight simulated label shifts where the model stays
calibrated; when one kind's scores are replaced by house-number-only scores with a wrong
prior (France's situation), the correction recovers 96–99 % of the lost F0.5. On test
it fires only for **France's generic-word swaps** (π = 0.81 vs mean p2 0.40, odds × 5.5):
16,217 more French links, nothing else changes.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, validation, 222,624 held-out S1):** **0.98895** (singletons
  0.98445; pair precision 0.99709, recall 0.97582).
  Progression: first submission 0.98211 → v2 features, stage 1 only 0.98337 → + stage-2
  re-ranker on 25 stage-1 features 0.98748 → on all 102 features 0.98895. With whole val
  regions held out per cross-fitting fold (harsher, closer to test's unseen regions):
  0.98684 vs 0.98581, so the gain transfers across regions. Stage 2 trained on US val only
  and applied to Indian val still beats stage 1 there (0.9807 vs 0.9790), our best proxy
  for France.
- **Common false positives (wrong merges)** (error analysis of the 25-feature re-ranker): 2,765 of 752,513 predicted links: 1,842
  lookalike distractors (same address, one-letter name edit or house number shifted by a
  few units, often indistinguishable from a true pair with a digit typo), 923 records of
  another S1 with the same generic name at a nearby address.
- **Common false negatives (missed matches):** 2.65 % of true pairs. More than half come
  from the 4 % of records with an **empty address** (they lose 32.5 % of their pairs):
  with half of all S1 names shared, the name alone often can't choose. Others: pairs whose
  noise mimics a distractor (house number ±2 and city renamed), made-up brand names at an
  address that hosts several S1, and retrieval ties among dozens of S1 with the same name.
- **Leaderboard (public):** first submission 0.975 (val 0.982); stage-2 re-ranker on all
  features 0.977 (val 0.98895); + domain normalisation 0.980 (val 0.98897). The widening
  val-LB gap led to adversarial validation (a classifier separates val from test pairs with
  AUC 0.99): test US has half the S1 of train US and 23 % more S2/S3 records per S1, so
  count- and IDF-based features sit on a different scale. Dividing them by their
  per-country, per-split scale before stage 2 recovered +0.003 on the leaderboard.

---

## 6. Conclusion

Measuring the generator's noise in the raw data and encoding it (filler words, brand
names, prepended house numbers, injected ids) gave the pair model better evidence; the
largest single gain came from letting a second model see the competition for each S1
entity. The cascade keeps the candidate set to about one S1 per record while retaining
99.97 % of the retrieved true pairs. The main remaining loss is records without an
address whose name is shared by many S1, which no feature of the pair can resolve.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` (Python ≥ 3.11; `requirements.txt` pinned,
`requirements-gpu.txt` optional): package `ber`, one CLI command per stage, each reading
the previous stage's artifacts from `artifacts/<run>/<stage>/` and writing its own with a
`manifest.json` (config hash, git commit, timings, peak memory, metrics). All thresholds
live in `configs/default.yaml`.

```bash
pip install -r code/business_entity_resolution/requirements.txt
pip install -e code/business_entity_resolution
B=ber   # run from the directory that contains dataset/
$B split --run main && $B normalize --run main && $B candidates --run main
$B features --run main && $B scorer --run main && $B rerank --run main
$B decide --run main && $B submit --run main   # -> output/matching_results.tsv, candidate_pairs.tsv
```

`--set data.phase=train` / `test` run the train/val side and the test side separately
(same outputs). `ber explain <entity_id>` traces one record through every stage.
Main modules: `normalize/` (text, address, geo, transliteration, lookup tables),
`candidates/` (TF-IDF, region blocking, GPU top-k), `features/` (pair + context features),
`scorer/` (stage 1), `rerank/` (stage 2), `decide/` (one owner, expected F0.5),
`submit.py` (TSVs + official validator), `eval/` (geographic split, macro F0.5).

### B. Additional Results

| Stage | Pairs per S2/S3 record | True pairs kept (val) |
|---|---|---|
| All S1 of the country | ~10⁶ | 100 % |
| TF-IDF top-k union | 24.6 | 99.05 % |
| Stage-1 filter (p1 ≥ 0.001) = `candidate_pairs.tsv` | 1.08 | 99.02 % |
| Final links | 0.72 | 97.58 % |

Stage-2 features by gain: record margin over its other candidates, p1, then the S1-side
claim features (`c_expected`, `c_best_other`, `c_sum_other`).
