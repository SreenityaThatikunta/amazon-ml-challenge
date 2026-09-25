# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

> Draft (2026-09-25). Numbers marked **TBD** are filled in from the final full-scale run.
> Full experiment record: `experiments/README.md`; task log: `progress.md`.

---

## 1. Executive Summary

A cascade pipeline: rule-based normalisation (tables learned from the EDA, plus an Indic → Latin word
dictionary learned from training pairs) → region-blocked character TF-IDF blocking → a LightGBM pair
scorer with ~90 string, number and *relative* features → a **second-stage cluster-context rescorer**
that looks at each Source 1 entity's other confident matches → one owner per S2/S3 record and a
per-entity **expected-F0.5** subset choice. France, which appears only in test, gets its own scorer,
trained with **synthetic French pairs** generated from the unlabelled test Source 1 France records with
the noise catalogue observed in training. It's validated on a realistic synthetic French proxy.

Validation macro F0.5 (held-out whole states): v1 0.98211 (public LB 0.975); v2 **TBD**.

---

## 2. Methodology

### 2.1 Problem Analysis
- **One owner per record:** every S2/S3 id appears in at most one ground-truth list, and true pairs
  always share the country label. So blocking is within a country, and each S2/S3 record is assigned
  to at most one S1.
- **US and India follow the same generator** (5.6% singletons, 3.46 matches per S1, ≤ 5 from S2 and
  ≤ 6 from S3). France, test-only, uses the same generator: gibberish brand names, aka/formerly, OCR
  digits.
- **Names are ambiguous, so the address is required.** 50% of S1 share their core name with another S1.
  Some true pairs don't match on name at all (gibberish brand names, initials such as "VPC" for
  "Vision Precision Clinic").
- **Distractors are lookalikes.** Same address with one arbitrary letter or legal form changed, or same
  name with the house number *shifted* by a small amount (805 → 812), often with a PMB/PO box. True-pair
  number noise is different: truncation (7622 → 762), zero padding, dropped numbers. True name noise is
  OCR-like (o→0, l→1) rather than arbitrary.
- **Transliteration:** 12–23% of Indian S2/S3 names are in one of 9 Indian scripts, as phonetic
  transliterations of the English words.
- **France (test only):** S1 names are "<City> <generic word> <legal form>". **13.3% of French S1
  share an exact address with another S1** (US 4.6%), and 1.2% share name and street at another number
  (US ~0). These S1-level twins caused most of the French ambiguity.

### 2.2 Solution Strategy
**Approach Type:** Blocking + GBDT classifier + cluster-context rescoring + expected-F0.5 decision
(hybrid).
**Core Innovation:**
1. Relative lookalike features (does *another* candidate match the house number, legal form or street
   exactly?).
2. A second-stage rescorer on cluster context (the S1's other confident members, the source caps),
   trained out-of-sample on the validation side.
3. An unseen-country scorer trained with synthetic French pairs built from the unlabelled test S1
   France records.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  - Character 3-gram TF-IDF (cosine top-k via sparse top-n), three indexes per country: name
    (k = 10), address (k = 10), name+address (k = 10). Queries run **from each S2/S3 record to S1**.
  - **Region blocking:** state or département detected from the address, with learned
    neighbourhoods (e.g. Telangana addresses written as Andhra Pradesh). Queries with no region search
    the whole country; queries with an empty address search the name index with a larger k.
  - Names are transliterated to Latin before indexing.
- **Candidate pairs generated:** ~244M for test (~24 candidates per query) **[TBD final]**.
- **How true matches were kept:**
  - Recall is measured per country, script and source on held-out states: pair recall 99.04%
    (US 99.3%, India 98.2–99.2%).
  - 68% of the remaining misses have an empty query address (see §5).

---

## 4. Matching Model

**Features used** (~90, no country feature; each means the same thing in every country):
- **Name:**
  - exact/compact equality;
  - edit distances split into substitutions vs insertions/deletions vs transpositions;
  - fuzzy ratios (rapidfuzz);
  - alias parts (aka/formerly/dba);
  - initials;
  - legal-form agreement/conflict;
  - gibberish flag (share of words unknown to the S1 vocabulary);
  - name frequency among S1;
  - name similarity *without the S1's own city word*.
- **Address:**
  - fuzzy ratios, token-set similarity, region agreement;
  - **house-number relation** (equal / padded / truncated / shifted / different, plus digit length
    and leading-digit truncation);
  - number-set overlap, PO box/PMB;
  - address words each side lacks (fuzzy, so OCR and typos still match).
- **Retrieval and context:**
  - TF-IDF cosines and ranks;
  - each score's margin over the query's best other candidate;
  - **relative agreement margins** (number, legal form, address words, initials, exact name) over
    the query's other candidates;
  - candidate counts per query and per S1.

**Model type:**
1. **Stage 1:** LightGBM binary classifier on candidate pairs from training-side states (~13.6M pairs),
   early-stopped on held-out states.
2. **Unseen-country model:** a second LightGBM, which adds 9M labelled pairs from synthetic French data
   (60k test S1 France records with S2/S3 copies drawn from the training noise catalogue, lookalike
   distractors, and real unmatched records as background). It scores pairs whose S1 country never
   occurs in training.
3. **Stage 2 (rescore):** LightGBM on each query's top-3 candidates.
   - Inputs: all stage-1 features and p, plus **cluster context**: size and summed p of the S1's other
     confident members, same-source count vs the 5/6 cap, how many members score higher, and the
     query's best name/address similarity to the members.
   - Trained on the held-out-state side, cross-fitted in 2 folds, so validation scores stay
     out-of-sample.
   - Applied to countries seen in training (on the French proxy it didn't help).
4. **Decision:** each S2/S3 record keeps its single best S1. For each S1 the pipeline keeps the prefix
   of its assigned records that maximises expected F0.5 (the empty set is one option). Exact duplicate
   records share an owner.

**Threshold selection method:** expected-F0.5 subset choice per S1, with the minimum p swept on the
held-out states.

---

## 5. Results & Error Analysis

- **F_0.5 score (macro, held-out states):**

| Version | Val F0.5 | US | India | Public LB |
|---|---|---|---|---|
| v1: stage 1 + expected-F0.5 | 0.98211 | 0.98539 | 0.97696 | 0.975 |
| v2: + lookalike features, unseen-country model, cluster rescore | **TBD** | TBD | TBD | TBD |

  On the dev slice, v2 moved 0.97655 → 0.98859. The France proxy with the unseen-country model scored
  0.971 (the proxy comes from the same generator as the training pairs, so this is optimistic).
- **Common false positives (wrong merges):**
  - lookalike twins at the same address with another legal form;
  - short French house numbers that look "truncated" (13 vs 3);
  - generic names with an empty address.
- **Common false negatives (missed matches):**
  - query with an empty address and a noisy generic name (68% of blocking misses);
  - Gurmukhi (Punjabi) names: Punjab is a held-out state, so the train-side dictionary has no Gurmukhi
    words, and the test dictionary is learned from all pairs **[if v3]**;
  - heavily rewritten Indian addresses.

---

## 6. Conclusion

Most of the gain came from understanding how the lookalike distractors are generated and giving the
model *relative* evidence ("another candidate matches exactly"). Scoring each match in the context of
its whole cluster helped too. The unseen country was handled without labels by recreating the training
noise on its unlabelled reference records, and a realistic proxy made that measurable.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`: package `ber`, one CLI stage per step. Reproduce with the command
sequence in `code/business_entity_resolution/README.md` (split → normalize → candidates → features →
synthetic French training set → scorer → rescore → decide → submit). `ber submit` writes
`output/matching_results.tsv` and `output/candidate_pairs.tsv`. `candidate_pairs.tsv` is the full
stage-1 candidate set, which the stage-1 model scores in full; the rescorer runs on a subset of it. The
official validator runs on the output. All libraries are MIT/BSD/Apache (`tools/check_licenses.py`); no
external data, APIs or geocoding are used. The unlabelled test Source 1 France records are used to
synthesise French training pairs.

### B. Additional Results
See `experiments/README.md` (E1–E17): proxy realism table, dev A/B tables, rescore sweeps, loss
breakdowns.
