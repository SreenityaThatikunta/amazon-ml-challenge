# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** pootato
**Team Members:** Sreenitya Thatikunta, Rudraksha Dev Sharma
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

A cascade that never compares all pairs: normalisation tuned to the noise we measured in
the data, region-blocked character-trigram TF-IDF retrieval (24.6 S1 candidates per S2/S3
record, 99.05 % of true pairs), a LightGBM pair scorer with 102 features that also acts as
a learned filter (**~1.07 candidates per record kept, 99.97 % of the retrieved true pairs;
5.53 candidates per S1 on test**), a small **transformer cross-encoder** (MiniLM,
22.7 M parameters, Apache-2.0) fine-tuned on the scorer's out-of-sample pairs, which reads
both records' text jointly, and a second LightGBM that re-ranks the survivors with *group
context*: how many other records claim the same S1, how strongly and from which source,
and how the cross-encoder rates the pair against the competing candidates. The re-ranker
is trained on whole held-out regions (the validation states plus 40 % more train-side
states, with every S1 group complete, as it is at test time), so it has seen many regions
and scores unseen ones well. A per-S1 expected-F0.5 rule (one owner per record) produces
the final links, with a label-free prior correction for a French name-noise pattern that
the training countries do not have. Validation macro F0.5 is **0.99060** on a geographic
hold-out (10 % of the S1 entities, whole states; 0.98918 without the cross-encoder),
**0.98988 with every validation state unseen by the model that scores it** (0.98817
without), up from 0.98211 for our first submission; public leaderboard **0.98316**
(0.982344 without the cross-encoder). Everything runs on one laptop (16 GB RAM; an NVIDIA
GPU for the cross-encoder, optional for retrieval) with MIT/BSD/Apache dependencies; the
only pretrained model is the 22.7 M-parameter cross-encoder.

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

**Approach Type:** Blocking + learned filter + transformer cross-encoder + group-aware
re-ranker + expected-F0.5 decision (a hybrid of gradient-boosted trees and a small
transformer).
**Core Innovation:** a *cascade* where the stage-1 classifier doubles as the final
blocking filter (tiny candidate set without losing recall), and a stage-2 re-ranker
trained cross-fitted on complete S1 groups that reasons about competing claims on the
same S1, the information a pairwise matcher cannot see. A transformer cross-encoder,
clearly weaker than the cascade as a matcher on its own, feeds the re-ranker its
*margins* over the competing candidates, which turn out to be strong evidence.

```
records ─► [0] normalise ─► [1] TF-IDF top-k (region-blocked, 24.6/record)
        ─► [2] 102 pair features ─► stage-1 LightGBM p1 ─► keep p1 ≥ 0.001 (~1.07/record)
        ─► [2b] MiniLM cross-encoder reads both records ─► logit + margins over competitors
        ─► [3] stage-2 LightGBM p2 (group context + cross-encoder evidence)
        ─► [4] one owner + expected F0.5 per S1
```

Data roles (whole regions, i.e. US and Indian states, never split):

| Part | Share of S1 | Used for |
|---|---|---|
| train regions | ~50 % | stage 1 (a 14 % sample of the 20 % featurised records); the cross-encoder (the other featurised records, whose p1 is out-of-sample) |
| held-out train regions ("hold") | ~40 % | stage 2 training (all records featurised; stage 1 never saw them) |
| validation regions | ~10 % | stage 2 training, cross-fitted (every val pair gets an out-of-fold p2), all metrics |

Whole regions keep lookalike distractors, which sit at nearby addresses, on the same
side as the entities they imitate. S2/S3 records follow their owner's region (unmatched
ones their own), in val and in the hold regions alike; all val records are scored, so val
F0.5 includes blocking losses, singletons and every stage, like the leaderboard.

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
  after the stage-1 filter (= `candidate_pairs.tsv`): 9,582,745 pairs (0.96 per record),
  **5.53 per S1 entity** (the S1 has 3.4 true matches on average).
- **How true matches were not lost:** recall measured on the validation side at every
  step: TF-IDF 99.05 % of true pairs (US 99.3 %, India 98.2–99.2 %), the stage-1 filter
  keeps 99.97 % of those (trade-off table in `artifacts/<run>/rerank/manifest.json`:
  0.001 → 1.07 pairs/record, 0.003 → 0.97 at 99.92 %, 0.01 → 0.88 at 99.75 %). The
  remaining misses are dominated by address-less records whose name dozens of S1 share.
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

**Stage-2 features (re-ranker, 132):** p1 and logit; record context (rank, best/sum/count
of the record's other p1); **S1 context** (rank of the record among all records listing
the S1, best/sum/count of competing p1, the same within the record's source, and
"claims": records whose best S1 this is, their expected count and best competitor);
all 102 stage-1 features passed through (25 of them: val 0.98748; all: 0.98895); the
cross-encoder's logit and its context (5 features, below).
**Domain normalisation:** 16 count-like features (per-S1 and per-record counts, name
frequency) are divided by their mean and 4 IDF-like ones by their 99th percentile, within each
country and split, because test has a different S1 density than train (section 5). The
scale of the 9 counts over an S1's group (claims, competitors) is taken from training
pairs whose S1 group is complete only: the ~9 % of training pairs whose group is
artificially empty (see "Complete S1 groups") pulled those means 9–10 % below what the
same density gives at test, where every group is complete, so the model read test's
competition as weaker than it is (public leaderboard 0.981708 → 0.982344; validation
cannot show this, as it shares the training scale).

**Model type:** LightGBM (MIT) binary classifiers. Stage 1: 127 leaves, lr 0.05, early
stopping on 5 % of val records (best iteration 330; val AUC 0.99994), trained on 3.6 M
pairs from 14 % of the featurised train-side records outside the held-out regions. Stage
2: 63 leaves, trained on the pairs of the validation regions and of the held-out train
regions (1.11 M + 4.28 M pairs with p1 ≥ 0.001; stage 1 never saw either, so p1 is
out-of-sample as on test) whose S1 group is complete (below), with **4-fold
cross-fitting** grouped by the record's top S1: every val pair gets an out-of-fold p2
(honest validation), test pairs get the mean of the 4 fold models. The only pretrained
model is the cross-encoder below (22.7 M parameters, Apache-2.0), far below the 8 B
parameter limit.

**Cross-encoder (`ber crossenc`, `src/ber/crossenc/stage.py`).** A transformer that reads
both records at once: `cross-encoder/ms-marco-MiniLM-L-6-v2` (Hugging Face, Apache-2.0,
6 layers, 22.7 M parameters), each record written as its normalised
`name core + legal form ; address + region` (at most 64 word-piece tokens per pair; 3 %
of pairs truncated). It is fine-tuned (binary cross-entropy, AdamW, 2 epochs, batch 256,
learning rate 5e-5 with 5 % linear warm-up and linear decay, fp16; ~20 min on a laptop
RTX 4050) on 982,265 pairs (68 % true) of 777,808 records from stage 1's own regions: the
featurised train-side records that stage 1 did *not* train on, kept at p1 ≥ 0.001 exactly
like stage 2's inputs. It
never sees a validation or held-out record or label (their S1 appear in its training data
only as cross-region negatives of train-side records, which could only bias it against
them), so its scores on validation, held-out and test pairs are all out-of-sample; it then
scores every stage-2 pair (15 M, ~2,800 pairs/s). **On its own it is a clearly weaker
matcher than the cascade** (AUC 0.974 on the validation stage-2 pairs, vs 0.996 for p1 and
0.9989 for p2; training log loss 0.06 vs 0.32 on validation): it learns street and city
strings of the training regions that do not transfer to new regions, and it sees house
numbers only as word pieces (`848` vs `843`), where the lookalike decoys differ. A logistic
stack of p2 and its raw logit barely helps (log loss 0.0357 → 0.0353). What helps is its
*context*: stage 2 gets the logit plus its margin over, and rank among, the record's other
candidates and the other records of the S1's group (`ce`, `ce_q_margin`, `ce_q_rank`,
`ce_s_margin`, `ce_s_rank`). `ce` and `ce_q_margin` rank 4th and 5th and `ce_s_margin` 7th
of the 132 stage-2 features by gain (after the record margin, p1 and its logit). With them,
val F0.5 rises from 0.98918 to 0.99078 and the unseen-region score from 0.98817 to 0.98988
(clean unseen-region India +0.0024, see section 5; singletons 0.98892 → 0.99385, i.e. far
fewer false links), and the public leaderboard from 0.982344 to 0.98316. The submitted
file comes from a re-run of this configuration with the packaged code (cross-encoder
retrained: val AUC 0.97417 vs 0.97485, stage-2 out-of-fold AUC 0.999059 vs 0.999088, val
F0.5 0.99060 vs 0.99078; GPU training is not bit-for-bit deterministic).

**Why stage 2 needs many regions.** Stage 2 first trained on the validation regions
only (10 % of S1). With whole validation regions held out per fold its score fell from
0.98895 to 0.98684: part of what it learns is region-specific, and every test region is
unseen by it. So we set aside whole train-side regions for it (`ber features-hold`
featurises all their records; stage 1 never trains on them): first 20 %, now 40 % of
each country's S1 (India AP, GJ, HR, MH, OD, RJ; US AL, AR, AZ, CO, CT, IA, ID, IN, LA, MD,
MO, MT, NY, OK, TN, UT, WY). Stage 1 does not suffer (val AUC 0.999939 with 40 % of the
regions removed vs 0.99994 with none). A re-ranker trained on held-out regions alone
scores the validation regions as well as one trained on other validation regions
(0.98666 vs 0.98660), so the regions are interchangeable training data. Results with
whole validation regions unseen by the scoring model, on the 191,917 validation S1 whose
groups are complete in every variant (see below): validation regions only 0.99020, + 20 %
held-out regions 0.99125, + 40 % (owner rule) 0.99156, + training on complete groups only
**0.99171** (India 0.98497 → 0.98778, singletons 0.98224 → 0.98903).

**Complete S1 groups.** The re-ranker's S1 context is only right if every record that
lists the S1 is loaded. Two fixes make training look like test, where every group is
complete:
- *Held-out records follow their owner.* A matched S2/S3 record belongs to its owner's
  region, as in the validation split, not to the region named in its own address; else
  5 % of the held-out S1's true copies were missing (mostly records whose address names
  no region: the ones that search the whole country, like 35 % of France's records).
- *Train only on complete groups.* Validation and held-out records also list S1 of the
  other train regions (cross-region candidates, whole-country searches). Those S1's own
  copies are never loaded, so their groups look empty and rejecting such a pair is
  trivially easy (9 % of the validation pairs, all negative); a model trained on them
  learns "empty group → reject", which never applies at test time. Excluding them from
  training (they are still scored) improves every validation measure: 0.98896 → 0.98923
  overall, 0.99156 → 0.99171 on unseen regions (complete-group S1).

**Threshold selection method:** one owner per record (its highest p2), then per S1 the
prefix of its records (sorted by p2) that maximises the expected F0.5
(≈ 1.25·Σp / (0.25·Σp_all + k), vs P(no match) = Π(1 − p) for the empty prediction), after
dropping p2 < min_p. min_p = 0.02 chosen on validation; the curve is flat (0.99058–0.99060
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
it fires only for **France's generic-word swaps** (final model: π = 0.81 vs mean p2 0.60,
odds × 2.4 after de-biasing; the cross-encoder already rates these swaps higher: without it
the mean p2 was 0.39 and the factor 5.4); every US/India kind stays within 0.9–1.9×, below
the threshold.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, validation, 222,624 held-out S1):** **0.99060** for the submitted
  file (US 0.99138, India 0.98936; singletons 0.99359; pair precision 0.99897, recall
  0.97353); 0.99078 in the first run of the same configuration (section 4). With every
  validation state unseen by the model that scores it (whole regions held out per
  cross-fitting fold, the harsher test-like check): **0.98988** (first run). Without the
  cross-encoder: 0.98918 and 0.98817 (0.98923 and 0.98822 without the normalisation-scale
  fix, which validation cannot see, see section 4).
  Progression: first submission 0.98211 → v2 features, stage 1 only 0.98337 → + stage-2
  re-ranker on 25 stage-1 features 0.98748 → on all 102 features 0.98895 → re-ranker
  trained on validation + 20 % held-out train regions 0.98911 (0.98776 unseen-region, vs
  0.98660 when trained on validation regions only) → 40 % held-out regions, records
  following their owner, training on complete groups only 0.98923 (0.98822) →
  normalisation scale from complete groups 0.98918 (0.98817) → + cross-encoder evidence
  0.99078 (0.98988). Stage 2 trained on US val only and applied to Indian val still beats
  stage 1 there (0.9807 vs 0.9790), our best proxy for France; a re-ranker trained only on
  the held-out regions scores the validation regions as well as one trained on them
  (0.98666 vs 0.98660).

  Loading more held-out regions also completes more validation groups (their S1's copies
  are then loaded), which makes the validation set harder, as test is. So the variants
  are compared on the 191,917 validation S1 (86 %) whose groups are complete in all of
  them ("clean"):

  | Re-ranker trained on | Folds | Val F0.5 | Clean | Clean US | Clean India |
  |---|---|---|---|---|---|
  | validation regions | whole regions held out | 0.98660 | 0.99020 | 0.99347 | 0.98497 |
  | + 20 % held-out regions | whole regions held out | 0.98776 | 0.99125 | 0.99401 | 0.98684 |
  | + 40 % held-out regions (owner rule) | whole regions held out | 0.98789 | 0.99156 | 0.99410 | 0.98750 |
  | + complete groups only | whole regions held out | 0.98822 | 0.99171 | 0.99417 | 0.98778 |
  | + normalisation scale from complete groups | whole regions held out | 0.98817 | 0.99163 | 0.99408 | 0.98773 |
  | + cross-encoder evidence (**final**) | whole regions held out | **0.98988** | **0.99309** | 0.99494 | 0.99014 |
  | + 20 % held-out regions | by S1 | 0.98911 | 0.99241 | 0.99400 | 0.98987 |
  | + 40 % held-out regions (owner rule) | by S1 | 0.98896 | 0.99261 | 0.99421 | 0.99005 |
  | + complete groups only | by S1 | 0.98923 | 0.99270 | 0.99424 | 0.99025 |
  | + normalisation scale from complete groups | by S1 | 0.98918 | 0.99263 | 0.99418 | 0.99015 |
  | + cross-encoder evidence (**final**) | by S1 | **0.99078** | **0.99400** | 0.99498 | 0.99242 |

  (Cross-encoder rows: first run of the final configuration; the submitted re-run scores
  0.99060 by S1.)
- **Cross-encoder alone vs as evidence** (all 1,112,888 validation stage-2 pairs; every
  score out-of-sample or out-of-fold; submitted run):

  | Scores | AUC | Log loss | AUC where p2 without it is 0.05–0.95 (68,986 pairs) |
  |---|---|---|---|
  | stage 1 (p1) | 0.99602 | 0.0661 | |
  | cross-encoder alone | 0.97417 | 0.3167 | 0.673 |
  | stage 2 without the cross-encoder | 0.99887 | 0.0359 | 0.862 |
  | stage 2 with the cross-encoder features | **0.99906** | **0.0320** | **0.890** |

  The transformer alone is the weakest model here, and nearly useless on the hard cases
  (0.673), which are ambiguous in the text itself; yet as evidence for the re-ranker it
  improves exactly those cases (0.862 → 0.890), because what it adds is *relative*: how
  much better it rates this pair than the record's other candidates and the S1's other
  records.
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
  features 0.977 (val 0.98895); + domain normalisation 0.980 (val 0.98897); re-ranker on
  validation + 20 % held-out regions, French prior correction 0.980973 (val 0.98911,
  unseen-region 0.98776); 40 % held-out regions, records following their owner, complete
  groups only 0.981708 (val 0.98923, unseen-region 0.98822); + normalisation scale from
  complete groups 0.982344 (val 0.98918, unseen-region 0.98817); final model (+ cross-encoder
  evidence) **0.98316** (val 0.99078, unseen-region 0.98988; first run). The widening
  val-LB gap led to adversarial validation (a classifier separates val from test pairs with
  AUC 0.99): test US has half the S1 of train US and 23 % more S2/S3 records per S1, so
  count- and IDF-based features sit on a different scale. Dividing them by their
  per-country, per-split scale before stage 2 recovered +0.003 on the leaderboard.
  The model's own expected F0.5 on test (0.9913, France 0.9870 lowest) minus its
  validation optimism predicts ≈ 0.985, so roughly half of the remaining gap was invisible
  to the model; the two causes we found (France's generic-word swaps and the empty S1
  groups of cross-region candidates, both in section 4) are addressed in the final model.
  We used the public leaderboard only as a check against regressions, not to choose
  parameters: all choices above rest on validation evidence, as the final ranking scores
  the other (private) test S1. The cross-encoder was adopted because it gained on
  validation (+0.0016), on unseen regions (+0.0017), in both countries and on singletons;
  the public leaderboard moved the same way (+0.0008). The submitted file is the re-run of
  the packaged code, not the file with the best public score.

---

## 6. Conclusion

Measuring the generator's noise in the raw data and encoding it (filler words, brand
names, prepended house numbers, injected ids) gave the pair model better evidence; the
largest single gain came from letting a second model see the competition for each S1
entity. The cascade keeps the candidate set to about one S1 per record while retaining
99.97 % of the retrieved true pairs. The late gains came from making the re-ranker's
training data look like test data: many held-out regions, every S1 group complete, and
feature scales taken from the same kind of pairs as at test. These moved the public
leaderboard (0.980 → 0.982) by more than validation could show, because validation shares
the training set's artefacts. The last gain came from a hybrid: a small pretrained
transformer cross-encoder was a weak matcher on its own (it overfits the training regions'
strings and cannot compare house numbers), but its margins over the competing candidates,
given to the re-ranker as evidence, raised validation by 0.0016, unseen regions by 0.0017
and the public leaderboard to 0.98316; each model does what it is good at. The main
remaining loss is records without an address whose name is shared by many S1, which no
feature of the pair can resolve.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` (Python ≥ 3.11; `requirements.txt` pinned,
`requirements-crossenc.txt` for the cross-encoder (torch, transformers), `requirements-gpu.txt`
optional): package `ber`, one CLI command per stage, each reading the previous stage's
artifacts from `artifacts/<run>/<stage>/` and writing its own with a `manifest.json`
(config hash, git commit, timings, peak memory, metrics). All thresholds live in
`configs/default.yaml`, which reproduces the submitted file (`rerank.extra_scores:
crossenc`; `null` gives the pipeline without the cross-encoder, no torch needed).

```bash
pip install -r code/business_entity_resolution/requirements.txt
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126   # CUDA build
pip install -r code/business_entity_resolution/requirements-crossenc.txt
pip install -e code/business_entity_resolution
B=ber   # run from the directory that contains dataset/
$B split --run main && $B normalize --run main && $B candidates --run main
$B features --run main && $B features-hold --run main   # held-out train regions for stage 2
$B scorer --run main
$B crossenc --run main   # fine-tune MiniLM (downloads cross-encoder/ms-marco-MiniLM-L-6-v2), score all stage-2 pairs; GPU ~2 h
$B rerank --run main
$B decide --run main && $B submit --run main   # -> output/matching_results.tsv, candidate_pairs.tsv
```

`--set data.phase=train` / `test` run the train/val side and the test side separately
(same outputs). `ber explain <entity_id>` traces one record through every stage.
Main modules: `normalize/` (text, address, geo, transliteration, lookup tables),
`candidates/` (TF-IDF, region blocking, GPU top-k), `features/` (pair + context features),
`scorer/` (stage 1), `crossenc/` (transformer cross-encoder: training pairs, fine-tuning,
scoring), `rerank/` (stage 2), `decide/` (one owner, expected F0.5), `submit.py` (TSVs +
official validator), `eval/` (geographic split, macro F0.5). Dependency licenses:
`tools/check_licenses.py` (all MIT/BSD/Apache/PSF except two MPL-2.0 packages used
unmodified as transitive dependencies: `tqdm`, `certifi`); the model is Apache-2.0
(22.7 M parameters).

### B. Additional Results

| Stage | Pairs per S2/S3 record | True pairs kept (val) |
|---|---|---|
| All S1 of the country | ~10⁶ | 100 % |
| TF-IDF top-k union | 24.6 | 99.05 % |
| Stage-1 filter (p1 ≥ 0.001) = `candidate_pairs.tsv` | 1.07 | 99.02 % |
| Final links | 0.72 | 97.35 % |

Stage-2 features by gain (submitted run): record margin over its other candidates (55 %
of the gain), p1 and its logit (37 %), then the cross-encoder's logit and record margin
(`ce`, `ce_q_margin`, 1.1 % each), `r_q_missing`, the cross-encoder's S1-group margin
(`ce_s_margin`) and the record/S1 context (`q_sum_other`, `s_n_top1`, ...). Without the
cross-encoder the S1-side claim features (`c_expected`, `c_best_other`, `c_sum_other`)
came right after p1.

**Evaluated and not adopted** (all on the validation regions, same stage 1; "clean" = the
validation S1 whose groups are complete in every variant; measured against the model
without the cross-encoder: val 0.98918, clean 0.99263, unseen-region 0.98817 / clean
0.99163):

| Variant | Result | Why not |
|---|---|---|
| Cross-encoder as the matcher (instead of the GBM cascade) | val stage-2 pairs AUC 0.974 vs 0.9989 | overfits the training regions' strings; house numbers only as word pieces |
| Cross-encoder logit stacked with p2 (logistic regression) | log loss 0.0357 → 0.0353 | small; its margins over competitors, given to stage 2, gain far more (adopted) |
| Stage 2 up to 3,500 trees (folds stopped at 2,280–2,400 instead of ≤ 2,000) | unseen-region clean 0.99168 vs 0.99170 | flat |
| Second stage-2 round, group features recomputed from out-of-fold p2 | unseen-region clean 0.99134 vs 0.99170, India −0.0011 | stacking shift: an unseen region's round-1 p2 looks unlike the training regions' |
| Stronger regularisation (min 2,000 per leaf, L2 10, 70 % features) | clean 0.99261 vs 0.99270 | flat / slightly worse |
| Two bags of cross-fitting (8 models) | clean 0.99265 vs 0.99263 | no measurable gain for twice the cost |
| Stage-1 filter at p1 ≥ 0.003 (−9 % candidates) | clean 0.99260 vs 0.99268; 0.05 % of true pairs lost | small F0.5 cost; 5.5 candidates per S1 is already small |
| Label-shift EM per lookalike segment | odds ratios at their bounds | the feature distribution shifts within segments |
| Importance weights from adversarial validation | effective sample 11 % | degenerate |

**Label-free calibration check on test.** Given the match status, the house-number
relation profile does not depend on the pair's segment, so the true-copy and decoy
profiles measured on labelled validation pairs let us estimate the share of true copies
among test pairs without labels. On the largest US test segment (same street, same name
key; 1.78 M records, twice the lookalikes of validation) it is 0.756 against the model's
mean p2 of 0.762 (model without the cross-encoder): the re-ranker stays calibrated under
the test's higher decoy density.
