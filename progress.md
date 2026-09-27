# Progress

Single source of truth for task status, runs, metrics and decisions.
Plan and task definitions: [plan.md](plan.md).

**Status keys:** ⬜ todo · 🔄 in progress · ✅ done · ⏸ blocked · ❌ dropped

---

## v2 (branch `improve/v2`, 2026-09-25 → 27)

Starting point: `main` at 6120d59 (val macro F0.5 0.98211 on the geographic split;
leaderboard ≈ 0.975, top 50). Goal: raise the leaderboard score without breaking the
rules, and ship the `candidate_pairs.tsv` that the organisers now say **counts toward the
final ranking** (smaller candidate set per S1 = ranked higher).

### v2 at a glance

| # | Change | Where | Why (the logic) |
|---|---|---|---|
| 1 | Windows support | `log.py` | `import resource` is Unix-only; peak memory now via `psutil` on Windows. The pipeline didn't start on the Windows laptop without this. |
| 2 | `data.phase = all / train / test` | `config.py`, every stage, `artifacts.py` | Experiments only need the train/val side (~2.5 h); the test side (~3 h) runs once per candidate model. A `test` pass reuses the model, region neighbourhoods and decision parameters of the train pass (manifests are merged). `data.phase` is excluded from the config hash. |
| 3 | Bounded memory | candidates, features, scorer, rerank | The laptop has 16 GB shared with the user's apps; main peaked at ~9.5 GB and hit 96 % RAM. Candidates and features now read records **one country at a time** (parquet filter) and build query TF-IDF matrices **per region block**; the scorer keeps its sample as one float32 matrix (LightGBM reads it in place: a mixed-dtype frame was silently converted to float64, doubling memory) and frees it once binned; rerank streams the val features part by part. Scorer sample: 10 % of featurised train queries (was 30 %), 5 % of val for early stopping (was 20 %); feature chunks of 1 M pairs (was 2 M). |
| 4 | Name normalisation fixes | `normalize/text.py`, `tables/name_stopwords.tsv` | Found in the raw data (see "Evidence" below): ~1 % of S2/S3 names carry injected record ids / phone numbers (`(ID: 63317)`, `#32038`, `- 9334025793`) that S1 never has → dropped (5-digit and ≥7-digit numbers, `#nnn`). Doubled words (`Dynamic Dynamic`) collapsed. Legal-word typos with swapped letters (`Priavte`, `Privtae`) now caught (OSA distance ≤ 1 instead of Levenshtein). New stopwords `of`, `et` (French "&"), `dr` (honorific the noise prepends to Indian names), `id`. On 300k Latin true pairs: exact core-name equality 58.9 % → 60.1 %. |
| 5 | Filler-free name key + IDF overlap (`k_*`) | `features/stage.py`, `features/build.py` | The generator appends filler words to S2/S3 names (US: Riverside, Westgate, Services, Holdings…; India: Smt, Shri, Sweets, Infratech…; France: Participations, Holding, Distribution, Développement…). They are learned **without labels**: tokens whose share of S2/S3 names is ≥ 2× their share of S1 names (a copy of an S1 keeps S1's token rates, so only injected tokens stand out); union over splits × countries so train and test agree. Key = sorted unique core tokens minus fillers (also removes word-order transpositions). Features: key equality / Levenshtein / ratio / token-set ratio, IDF-weighted recall & precision, largest IDF of an unmatched token on each side ("Meridian Dental" vs "Meridian Plumbing"), unmatched counts; context margins. Vectorised with sparse matrices (~4 µs/pair). |
| 6 | House-number features v2 (`hn_*`) | `features/numbers.py` | In true pairs the *first* number is often not the house number: Indian S2/S3 addresses get a fake number prepended (`H.NO 937 33, …`, `PLOT 57 1808/B4`) and components are reordered (`Unit 177, AZ, 9310 Main St` → `AZ, 9310 MAIN ST`). Main compared only first numbers (India: 12.6 % of true pairs "different"). New: S1's first number anywhere in the query numbers (and vice versa), best relation to any query number, digit edit distance, position of the leftmost differing digit (true typos hit leading digits 394→894; distractors shift the last digit 5835→5838). |
| 7 | Made-up brand names (`q_brand`, `s_brand`) | `features/stage.py` | ~1.6 % of true pairs replace the whole name with a made-up brand (`Grand Physical Therapy` → `Jaxdrexquo`). They are concatenations of ~45 syllables (tavo, jax, kelo, zeph, flux, onyx, …) learned from train pairs (maximal frequent prefixes). The flag covers 96 % of them and fires on 0.03 % of S1 names, so the model knows the name carries no evidence and must decide on the address. |
| 8 | Address-word IDF overlap (`a_idf_*`) | `features/build.py` | Street and locality words are rare, cities and street types common; exact IDF-weighted overlap of the address words (numbers excluded) helps exactly the pairs whose names can't (brands, initials) and lookalikes at another street. |
| 9 | **Stage-2 re-ranker** (`ber rerank`) | `rerank/stage.py` | Stage 1 judges each pair alone (plus the *query's* other candidates). It never sees the *S1 side*: how many other records already claim that S1, how strongly, from which source. That is what separates a true match from the lookalike distractor next to it, and a singleton from an S1 that attracts one stray record. Stage 2 = LightGBM on p1 + query context + S1 context (rank of the record among all records listing the S1, best/sum/count of competing p1, same restricted to the record's source, "claims" = records whose best S1 it is) + 19 stage-1 features. It needs complete S1 groups, so it trains on the **val side, cross-fitted in 4 folds** (grouped by the record's top S1): every val pair gets an out-of-fold p2, the val F0.5 stays honest, and test uses the mean of the 4 fold models. Pairs with p1 < 0.001 are dropped first. |
| 10 | `candidate_pairs.tsv` from the final model's input | `submit.py` | Main didn't write it (3 GB at 2.4 e8 TF-IDF pairs). The rules: "the exact set of records you feed into your matching model … if your pipeline has several blocking/filtering stages, the last one". With stage 2 that is the stage-1-filtered set (~1.3 pairs per record), a few candidates per S1 instead of ~140. Validator now also checks the candidate file. |
| 11 | Exact expected F0.5 (`decide.method = expected_f_exact`) | `decide/expected_f.py` | The ratio-of-expectations approximation is biased on the tiny groups here (1–6 candidates per S1). Exact expectation via Poisson-binomial distributions (brute-force checked in tests). Both variants are reported by `ber decide`. |
| 12 | Ablation switch `scorer.drop_features` | `scorer/gbm.py` | Train the scorer without a feature group (`k_*`, names) to measure what each change buys. |
| 13 | Optional GPU top-k (`candidates.device = auto`) | `candidates/tfidf.py`, `requirements-gpu.txt` | Stage 1 is the slowest step on the laptop (train India alone: 101 min). With CuPy (MIT) the S1 block is uploaded once as CSR, query chunks are densified on the GPU, one cuSPARSE product gives all similarities, and 10 arg-max passes give the top-10 (a full sort was slower than the CPU). Only blocks with ≥ 50k S1 go to the GPU: on small blocks the CPU kernel is as fast. Checked against the CPU path: identical k-th similarities (≤ 3e-7); only ties are broken differently. Large synthetic block: 27.9 s CPU → 2.6 s GPU. Falls back to CPU when CuPy/CUDA is missing. |
| 14 | Domain normalisation in stage 2 (`rerank.domain_norm`, v2.2) | `rerank/stage.py` | See "Leaderboard vs validation" below: size/density-dependent stage-2 inputs divided by their per-country, per-split scale (LB 0.977 → 0.980). |
| 15 | **Anchor-based prior correction per name-change kind** (`decide.prior_shift = anchor`, v2.3) | `decide/anchor.py`, `decide/stage.py` | France swaps the generic word of a name at the same address ("Ski College" → "Ski Comite"), a pattern absent from US/India; the model is uninformed there (p ≈ 0.4). The share of true copies per kind is estimated without labels from the house-number relation (independent of the name change given the match status), de-biased on val, and the kind's odds are corrected when they are off by > 2×. Only France's generic swaps qualify (×5.5). Details: "v2.3" below. |

### Evidence behind the changes (raw-data checks, 2026-09-25)
- **Name noise per country (test S2/S3 vs S1, % of names):** US/India carry `(ID: …)` 0.26–0.32 %, phone numbers 0.26–0.32 %, `#nnnnn` 0.26–0.32 %; France none of those but quotes (0.05 %), `et` for `&` (0.27 %) and far more accents (17 %). Injected honorifics in India only (8 %).
- **Token add/drop rates (S1 vs S2/S3 names):** added — US `riverside/westgate/lakeside/midtown/…` (~0.5 % each, never in S1), `services`, `holdings`; India `smt/shri/sri/dr/mr` (~1.3 % each), `sweets/bakery/hardware/…`; France `participations` (2.4 %), `holding`, `distribution`, `international`, `développement`, `groupe`. Dropped — legal forms, `group`, `associates`, `center`, `india` (~21 % of the time).
- **True-pair name residuals after normalisation (300k Latin pairs):** 60 % equal, 9.5 % typo ≤ 3, 5.4 % query adds a token, 5.3 % drops one, 2.5 % reordered, 17 % other (domains, heavy typos, brand replacement, initials). True-pair typos are *not* just OCR-style: arbitrary letter substitutions/insertions are common (`Mitllls`, `Amvhation`, `Hfabor`).
- **House numbers in true pairs:** equal 63 % (India) / 73 % (US); "different" 12.6 % / 4.7 % — mostly the prepended fake number and reordering; "shifted" 3.4 % / 1.7 % (digit typos).
- **Test is not train-shaped:** 5.5–5.8 S2/S3 records per S1 in test vs 4.7 in train, i.e. ~40 % of test records are unmatched vs 26 % in train (estimated from main's predicted links). If the extra records are lookalikes, precision on test is at more risk than val suggests: see open questions.

### Results
| Date | Run | Change | Pair recall (TF-IDF) | Val F0.5 | US | India | Singletons | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-09-25 | main (reported) | baseline | 99.04 % | 0.98211 | 0.98539 | 0.97696 | 0.97707 | Mac, from progress below |
| 2026-09-25 | dev2 (dev slice) | v2 features, stage 1 only | 99.73 % | 0.98079 | | | | main's dev slice: 0.97655 |
| 2026-09-25 | dev2 (dev slice) | + stage-2 re-ranker + exact expected F | | **0.98771** | | | | p2 AUC 0.9985 vs p1 0.9951 on kept pairs |
| 2026-09-25 | v2 (full) candidates | v2 normalisation, CPU top-k | 99.046 % (cluster 96.79 %) | | | | | 24.5 cands/query, 2.53e8 train pairs; India 101 min + US 44 min on the laptop CPU; India S3 Latin 98.2 % is the weakest segment |
| 2026-09-25 | v2 (full) | v2 normalisation + features, **stage 1 only** (4.5M training rows, 102 features) | 99.046 % | 0.98337 | 0.98617 | 0.97898 | 0.97771 | pair P 0.99655 / R 0.9586; AUC 0.99994; val +0.0013 over main with a third of main's training rows |
| 2026-09-25 | v2 (full) | **+ stage-2 re-ranker** (expected-F, min_p 0.25) | 99.046 % | **0.98748** | 0.98923 | 0.98473 | 0.98220 | pair P 0.99645 / R 0.97349; p2 AUC 0.99917 vs p1 0.99620 on the kept pairs; sweep flat (0.98743–0.98748 for min_p 0.02–0.5); exact expected-F 0.98744 (no gain, kept the approximation) |

**Stage-2 experiments (2026-09-26, run `v2x` = v2's upstream stages via junctions, val only):**

| Stage-2 variant | CV folds | Val F0.5 | US | India | Singletons | Pair P / R |
|---|---|---|---|---|---|---|
| 25 stage-1 features (first v2 file) | by S1 | 0.98748 | 0.98923 | 0.98473 | 0.98220 | 0.99645 / 0.97349 |
| **all 102 stage-1 features** | by S1 | **0.98895** | | | 0.98445 | 0.99709 / 0.97582 |
| all, 127 leaves, lr 0.03 | by S1 | 0.98897 | 0.99046 | 0.98663 | 0.98381 | 0.99706 / 0.97605 |
| 25 features | whole regions held out | 0.98581 | 0.98860 | 0.98144 | | |
| all 102 features | whole regions held out | 0.98684 | 0.98970 | 0.98234 | | |
| (stage 1 only, for reference) | — | 0.98337 | 0.98617 | 0.97898 | 0.97771 | 0.99655 / 0.9586 |

- Region-held-out CV (each fold = whole val regions, largest first to the smallest fold;
  Uttar Pradesh, most of India's val data, is one fold) is the harsher check that the gain
  transfers to unseen regions (test regions are all unseen by stage 2). All-features wins
  there too (+0.0010), and both beat stage 1 alone by ≥ 0.0025. Chosen: all 102 features,
  default size, folds by S1 (each fold model then sees every val region).
- A first region split put all Indian regions in one fold, i.e. stage 2 trained on US val
  only and applied to India: India still improved over stage 1 (0.9807 vs 0.9790). That
  is the closest thing we have to a France check.

**Refreshed submission v2.1 (2026-09-26 03:33, stage 2 on all 102 features):** val 0.98895;
test 5,947,349 links, same `candidate_pairs.tsv` (9,649,852); validator PASS; 4.2 % of S1
predictions differ from the first v2 file; links per S1 France 3.438 / India 3.422 / US 3.444.
Files: `output/` and `D:ml\submissions2.1\` (first v2 file kept in `...\submissions2\`).
Stage-2 test pass peaks at 9.3 GB with all 102 features.

**Test pass (2026-09-26 02:38, `output/`):** validator PASS (matching + candidate file).
5,933,335 links (main: 5,835,284); `candidate_pairs.tsv` 9,649,852 pairs = **5.57 per S1**
(TF-IDF: 243.8 M = 141 per S1). Per country the predictions look alike and match val:
links per S1 France 3.436 / India 3.415 / US 3.432 (val 3.380, val truth 3.460); empty S1
5.5 / 5.6 / 5.7 % (val 5.8 %). So France is not under-linked. Test has more records whose
best p2 is in the ambiguous 0.01–0.5 band (India 7.4 % of records vs 3.5 % on val), which
fits the larger share of unmatched records in test; France also has more in 0.5–0.99
(6.9 % vs 4.0 %), a possible precision risk there. Timings (laptop): candidates 78 min
(GPU), features 115 min, scorer 12 min, rerank 6 min, decide + submit 2 min.

Stage-2 input (= `candidate_pairs.tsv`): at p1 ≥ 0.001, **1.075 pairs per S2/S3 record**
keeping 99.97 % of the true pairs that TF-IDF found (24.6 per record before the filter).
Top stage-2 features: the record's margin over its other candidates, p1, then the S1-side
claims (`c_expected`, `c_best_other`, `c_sum_other`), i.e. exactly the group context stage 1 lacks.

### Leaderboard vs validation (2026-09-26)
| File | Val F0.5 | Public LB | Gap |
|---|---|---|---|
| main (6120d59) | 0.98211 | 0.975 | 0.007 |
| v2.1 (stage 2 on all features) | 0.98895 (0.98684 with whole regions held out) | 0.977 | 0.012 (0.010) |
| **v2.2 (+ domain normalisation of size-dependent stage-2 features)** | 0.98897 | **0.980** | 0.009 |
| v2.3 (+ anchor prior correction: France's generic-word swaps) | 0.98897 | not submitted (decision below) | |
| v2.4 (stage 2 on val + 20 % held-out train regions, stage 1 without them, + v2.3's anchor correction) | 0.98911 (unseen-region CV 0.98776) | 0.980973 | 0.008 |
| v2.5 (40 % held-out regions by owner, stage 2 on complete groups only) | 0.98923 (unseen-region CV 0.98822) | 0.981708 | 0.0075 |
| **v2.6 (+ normalisation scale of the S1-group counts from complete groups)** | 0.98918 (val-blind change) | **0.982344** | 0.0068 |

Submission policy (2026-09-26, with the user): we assumed the public board shows 3
decimals (it shows 6, as the v2.4 score 0.980973 revealed), so a v2.3 upload looked
uninformative (expected gain ≈ +0.0007). The public part is presumably a random part of the same test set
(hundreds of thousands of S1, sampling noise ≈ 0.0001), so differences between our files
do carry over to the private part, but only large ones are visible. Decision: v2.3 is not
uploaded on its own; v2.4 (which includes it) is uploaded once as a check against a
regression; the final choice rests on validation and robustness evidence.

v2.2 is now the default (`rerank.domain_norm: true`) and the file in `output/` (2026-09-26 08:14).
Submissions: 5 per team per day (2 used on 2026-09-26: v2.1, v2.2).

Val gained +0.0068 but the leaderboard only +0.002, so what test does differently matters
more than further val gains. Checks so far (`D:\aml\analysis\shift.py`, `lowp_inspect.txt`,
`france_inspect.txt`):

- **Test has ~2× the "shifted house number" lookalikes per S1.** Top candidate of each
  record has a shifted number: US 32.2 % (val 20.9 %), India 17.5 % (10.7 %), France 23.5 %;
  legal-form conflicts US 7.5 % (5.0 %), India 2.7 % (1.8 %); records with no plausible
  candidate (p1 < 0.01) 32–34 % vs 21–23 %. Per S1: ≈ 0.85 shifted near-copies on test vs
  ≈ 0.45 on val (US); val says only 12 % of those are true matches.
- **…but that doesn't explain the gap.** Doubling val's shifted lookalikes (69,607 duplicated
  unmatched records) costs only −0.0003 F0.5, and halving the odds of shifted pairs wins back
  < 0.0001. The model already rejects them (they sit at p1 ≈ 1e-4).
- **Label-shift EM per segment (Saerens et al. 2002) is unreliable here:** most segment odds
  ratios hit the bounds (0.2 / 5), even in "exact name + equal number" (99.9 % true), because
  the feature distribution shifts *within* segments (the EM assumption fails). Kept as
  `decide.prior_shift=segment_em` (off by default); not submitted.
- **France looks healthy** on inspection (25 random S1): links are correct noisy copies
  (typos, initials `MU` for `MED Union SAS`, made-up brands at the same address), link counts
  per S1 match US/India. Weak spot: address-less records with generic names (as in US/India).
- **Records without a candidate are lookalikes** in test as in val (same name + a filler,
  same street, house number moved 5–20), not a recall hole.
- Region-held-out CV says ~0.002 of the val score is stage 2 learning region-specific patterns.

- **Adversarial validation** (`D:\aml\analysis\adversarial.py`; LightGBM telling val kept pairs
  from test kept pairs on the 127 stage-2 inputs, 3-fold): AUC **0.986** (US), **0.990** (India),
  **0.9997** (France vs US+India val). Top separating features are the *size/density-dependent*
  ones: IDF of unmatched tokens (`k_idf_unmatched_q`, `a_idf_unmatched_*`), per-S1 counts
  (`s_n`, `s_n_top1`, `s_n_cands`, `c_expected`) and `s_name_freq`. Cause: test US has **half
  the S1 of train US** (663k vs 1.32M) and 23 % more S2/S3 records per S1. Even on near-certain
  matches (p1 > 0.999) the 90th percentile of `s_name_freq` is 31 on test vs 61 on val (US) and
  unmatched-token IDF is lower by 0.69 = log 2. Importance weights from this classifier are
  degenerate (effective sample share 11 %), so reweighting is not usable as is.
- Response (v2.2 candidate): **domain normalisation** of those features in stage 2
  (`rerank.domain_norm`): count-like columns ÷ their mean and IDF-like columns ÷ their 99th
  percentile, within each country and split, so a stage 2 trained on val reads test on the same
  scale. Caveat: part of the difference is genuine (a common name really has fewer competitors
  in a half-size pool), so only the leaderboard can tell which way is right.

  Result: val 0.98897 (neutral, as expected on one domain); test file
  `D:ml\submissions2.2_domainnorm\` (validator PASS, 5,912,771 links, 3.413 per S1) differs
  from v2.1 on 4.1 % of S1 (France 6.4 %, India 4.0 %, US 3.3 %). The stage-1-only probe
  differs from v2.1 on 9.4 % of S1.

### v2.3 (2026-09-26): France's generic-word swaps, anchor-based prior correction

Session goal (team decision): a well-generalising, defensible approach over squeezing the
public score. Scripts and outputs: `D:\aml\analysis\{selfest,swap_check,fr_swaps,fr_hn_mix,
anchor_proto,anchor_sim,anchor_sim2}.py/.txt`.

**1. Where the model itself is unsure on test.** Expected F0.5 of the chosen predictions
under the model's own p2 (exact Poisson-binomial), per country:

| | val actual | val expected | test expected |
|---|---|---|---|
| US | 0.99046 | 0.99542 | 0.99366 |
| India | 0.98662 | 0.99400 | 0.99072 |
| France | – | – | **0.98697** |

The model knows test is harder, and France hardest. Subtracting val's optimism gives ≈ 0.985
for test, so ≈ 0.005 of the LB gap is not visible to the model at all. France has 6.0 % of
its records in the uncertain band p2 ∈ [0.2, 0.9) vs 2.7 % (US) and 4.0 % (India).

**2. France's uncertain records are one pattern:** same address, equal house number, and a
name whose *generic* word differs: `Ski College SARL` → `Ski Comite SARL`, `BYH Club SAS` →
`BYH Loisirs SAS`, `Dessin Club` → `Dessin & Fils`, `Bio College SAS` → `Bio SAS France`.
Counting one-token swaps on filler-free name keys where both tokens are in ≥ 0.1 % of the
country's S1 names ("generic swap"), among each record's top pair on the same street:

| | generic swaps | model's view |
|---|---|---|
| val US | 77–256 (0.1 %), mostly decoys (y 0.29) | calibrated |
| val India | 2,573 (1.5 %), y 0.55 | calibrated |
| **test France** | **71,767 (6.6 %)** | p2 mean 0.40; 18.3k of the 52k equal-number ones linked |

The model has never seen this pattern (US/India swaps are almost all rare words, i.e.
typos) and answers with its US/India prior. Generic words are *not* added by the noise
(their S2/S3 rate is 0.86× their S1 rate for every one of them), but "& Fils" (1.55×) and
"France" (1.21×) are; the T-swaps (generic ↔ generic) are symmetric (club→école 308,
école→club 285), i.e. random replacement.

**3. Are they true copies? Evidence without labels.**
- *House-number relation is independent of the name change given the match status*
  (val: the true copies' hn_rel profile is the same for equal names, typos, generic swaps
  and other changes, ~76 % equal / 2 % shifted in the US; decoys are 56–82 % shifted). In
  France the generic swaps' profile (73.6 % equal, 16.0 % shifted) matches the exact-name
  copies' (74.8 %, 13.3 %), while US test generic swaps look like decoys (23 % / 54 %).
  A two-component mixture gives ~80 % true copies (0.78–0.82 whichever decoy profile is
  assumed); French decoys would need ≥ 58 % equal house numbers for a 50/50 split, far above
  any decoy type in US/India (0.5–25 %).
- *Not "sibling" businesses:* the deduplicated French S1 file does contain same-address
  siblings (1,071 pairs like `Nantes Club SARL` / `Nantes Foyer SAS`, vs 10 in the US), but
  the swaps' top S1 sits at a shared address 13.3 % of the time, exactly the base rate of
  French S1 (13.3 %) and of exact-name copies (13.2 %): no clustering at sibling addresses.
- The problem statement lists "DBA/trade names" as name noise; this looks like the French
  form of it.

**4. Method (`ber/decide/anchor.py`, `decide.prior_shift = anchor`).** Per country and
name-change kind k ∈ {equal key, rare swap (typo), generic swap, other}, over each query's
assigned pair on the same street (address token-set ≥ 80):
- hn_rel histogram h_k = π_k h1 + (1 − π_k) h0 → π_k by least squares; h1, h0 from the
  exact-name kind with the model's p as soft labels (on val within 0.002 of the true
  profiles);
- r_k = odds(π_k) / odds(mean p over k) = how far the model's prior for the kind is off;
- the estimator is biased for the non-exact kinds (edited-name decoys keep their house
  number more often than same-name decoys), so the val ratios (model calibrated there, r
  should be 1) are pooled into bias factors b = {typo 1.44, generic 1.17, other 1.74} and
  divided out;
- a kind is corrected only when the de-biased ratio is beyond 2× (either way; the largest
  single-country val deviation from the pooled bias is 1.7×) and it has ≥ 2,000 pairs:
  p' = r p / (r p + 1 − p), r ≤ 6.

**5. Validation on labelled val (`anchor_sim.py`, `anchor_sim2.py`).**
- Val itself: no kind corrected, val F0.5 **0.98897** (unchanged, by construction).
- No false alarms under label shift where the model stays calibrated: removing 90 % of the
  decoys or 60 % of the true copies of a kind (US/India × typo/other, 8 scenarios): de-biased
  ratios 0.83–1.17, nothing corrected.
- Detection + recovery where the model is uninformed with a wrong prior (France's
  situation): replacing p in one kind by q(hn_rel) computed with prior 0.4:

| kind (true share) | real p | wrong-prior q | + anchor | right prior (oracle) |
|---|---|---|---|---|
| US typo swaps (0.81) | 0.98896 | 0.97600 | **0.98413** | 0.98429 |
| India other (0.86) | 0.98896 | 0.97020 | **0.98082** | 0.98123 |
| US other, prior 0.95 (0.90) | 0.98896 | 0.98389 | 0.98389 (r 0.66, below threshold) | 0.98410 |

  The correction recovers 96–99 % of the loss and stays off when the error is small.

**Limits (`anchor_legal.py`).** The anchor assumption is kind-specific. Along another
axis, legal-form conflicts at the same street with an equal name key, the val bias is
large and unstable (odds ratio 1.03 in the US but 13.8 in India, where the model's 1.1 %
is right: 2.2 % true), because India's legal-form decoys keep their house number. So the
correction is only used for name-change kinds, whose val biases are moderate and similar
across countries (typo 1.26/1.54, generic 1.11/1.99 with n = 256 in the US, other
1.84/1.69). France's generic swaps stay beyond 2× even with the worst of these (6.48 /
1.99 = 3.3), so the direction of the correction is robust; its size (3.3–5.5×) is not.
The residual risk: French decoys created as "another business at the same address"
that keep the house number would look like true copies to the anchor; nothing in the
data points to them (no clustering at shared addresses), but only the leaderboard can
rule them out.

**6. Effect on test (run `v23` = v2.2's stages via junctions + anchor decide).** Only
France's generic swaps are corrected (π_anchor 0.815 vs mean p 0.404 → r 6.48 / bias 1.17 =
**5.53**); every other country × kind stays within 0.87–1.81 (US generic swaps 1.81, below
2). File `D:\aml\submissions\v2.3_anchor\`: validator PASS, 5,928,875 links (+16,217 /
−113 vs v2.2), all in France: 15,705 French S1 changed (6.05 %), 746 went from empty to
linked; candidate_pairs.tsv unchanged (9,649,852). Expected LB effect is small (≈ +0.0006
to +0.0008 if ≥ 90 % of the new links are right, ≈ 0 at 70 %): a 3-decimal leaderboard may
not show it.

### v2.4 (2026-09-26): stage 2 trained on more regions (val + 20 % held-out train regions)

**Why.** Stage 2 (the re-ranker, worth +0.0056 on val) trained only on the val side:
10 % of the S1 (US: 9 states; India: 5 regions, mostly Uttar Pradesh and Punjab). With
whole val regions held out per fold it scored 0.98684 vs 0.98895 when every fold saw
every region, so part of what it learns is region-specific, and every test region is
unseen by it. More regions in its training set should make it more robust.

**How.** `rerank.hold_frac = 0.2`: per country, whole train-side regions are set aside
(seeded, same greedy rule as the val split): India AP, GJ, HR, KL, OD, RJ (21.7 % of
India's S1), US AR, CT, IA, ID, IN, MD, NY (19.8 %). `ber features-hold` featurises all
their queries: 50.6 M pairs of 2.09 M records, 61 min, peak 7.6 GB; identical to
`ber features` (checked on the 1.0 M pairs both computed: all 103 columns equal). The
stage-1 scorer trains without them (3.52 M rows instead of 4.52 M; val AUC 0.999938 vs
0.99994, logloss 0.00293 vs 0.00290) and scores them out-of-sample. Stage 2 loads val and
hold pairs (group features and domain scales over both); `rerank.train_sides` picks the
training rows.

**Finding: incomplete S1 groups made part of val too easy.** Stage 2 only ever loads val
(and now hold) records. 16.2 % of v2.2's kept val pairs (181,360; all negatives; p2 ≈
0.001) sit on train-side S1 whose owners are never loaded, so their S1 groups look
empty and rejecting them is trivial. At test time every group is complete. With the hold
regions loaded, 40,520 of them get their owners' claims (3.2 true claims per S1 on
average), and a val-only stage 2 (E0) scores them badly (median p2 0.022, 3.7 % above
0.5, 905 outrank the record's true S1): it had never seen a cross-region candidate whose
S1 is strongly claimed by its own copies. That is what whole-country searchers meet on
test: records without a detectable region are 20.9 % of the kept test pairs (mostly
France's 35 % region-less records). So v2.2's val score is flattered by these pairs, and
v2.2 may be weak on this case at test time; training on val + hold, which contains many
such cases, addresses it. (E0 vs v2.2 therefore mixes this with the weaker stage 1:
−0.0005; on the 1.05 M pairs not affected, stage-2 logloss +7.6 %, on the 41.6 k
affected ×38.)

**Experiments** (run `v24`, same new stage 1 everywhere, 222,624 val S1; min_p 0.2 in all):

| | stage 2 trained on | folds | val F0.5 | US | India | singletons | pair P / R | p2 AUC / logloss |
|---|---|---|---|---|---|---|---|---|
| E0 | val | by S1 | 0.98846 | 0.99000 | 0.98605 | 0.98525 | 0.99763 / 0.97228 | 0.999132 / 0.0334 |
| **E1** | **val + hold** | by S1 | **0.98911** | 0.99043 | 0.98704 | 0.98701 | 0.99799 / 0.97306 | 0.999251 / 0.0311 |
| E2 | val | whole regions | 0.98660 | 0.98968 | 0.98176 | 0.98172 | 0.99750 / 0.96831 | 0.998773 / 0.0396 |
| **E3** | **val + hold** | whole regions | **0.98776** | 0.99029 | 0.98377 | 0.98717 | 0.99798 / 0.97009 | 0.999016 / 0.0357 |
| E4 | hold only | – (val never seen) | 0.98666 | 0.98865 | 0.98355 | 0.98701 | 0.99827 / 0.96502 | 0.997323 / 0.0626 |

- On unseen regions (whole regions held out per fold) more regions help: **+0.0012**
  (E3 vs E2), India +0.0020, singletons +0.0055.
- With folds by S1: +0.00065 (E1 vs E0); E1 is above v2.2's 0.98897 although its val
  now contains the harder complete-group negatives.
- A stage 2 trained only on the hold regions (E4) does as well on val as one trained on
  other val regions (0.98666 vs 0.98660): the hold regions are as useful as val for
  training, and the model transfers across regions. Its val logloss is poor (0.063):
  trained on one set of regions it is overconfident elsewhere (precision 0.99827, recall
  0.96502), the region-specificity that mixing regions reduces.
- Decision: **v2.4 = E1** (stage 2 on val + hold, folds by S1) + the v2.3 anchor correction.

**v2.4 test pass** (`D:\aml\run_v24_test.sh`; decision parameters re-chosen with the default
sweep: val F0.5 0.98911, min_p 0.02, the sweep is flat 0.98909–0.98911 over 0.02–0.5; min_p
0.3 changes 633 of 1.73 M test S1 (0.04 %), so the automatic choice was kept). Anchor
correction: again only France's generic swaps (π_anchor 0.780 vs mean p2 0.264: the new
stage 2, which saw more Indian regions where such swaps are mostly decoys, is even more
sceptical; de-biased ratio 8.8, capped at 6); every US/India kind 0.86–1.68.
Files `D:\aml\submissions\v2.4_hold\` = `output\` (official validator PASS, also with
`--check-ids`): 5,906,819 links; `candidate_pairs.tsv` 9,750,222 pairs = 5.63 per S1 (the
new stage 1 keeps 1 % more); links per S1 France 3.420 / India 3.397 / US 3.420 (v2.2: 3.403
/ 3.406 / 3.425); empty S1 5.4–5.7 % as before; S1 changed vs v2.2: France 8.5 %, India
5.1 %, US 3.5 %. **Public LB 0.980973** (2026-09-26 ~14:00; the board shows 6 decimals,
not 3 as assumed above, so small differences are visible).

### v2.5 (2026-09-26 evening, third session): more hold regions, complete hold groups

**1. The hold records were picked by the wrong region.** v2.4's `features-hold` took the
train-side S2/S3 records whose *own* address is in a hold region (`D:\aml\analysis\
hold_check.py`). The val split instead puts a matched record where its *owner* is. The
own-region rule missed 78,428 true copies of hold S1 (5.0 % of them; mostly the
4–5 % of true copies whose address names no region, i.e. exactly the records that search
the whole country, like 35 % of France's) and took in 35,894 copies of S1 outside the
hold regions (whose groups are then incomplete). New option `rerank.hold_by`:
`query_region` (v2.4, still the default until v2.5 is chosen) | `owner` (the val rule:
owner's region if matched, own region if not). Unit test in `tests/test_split.py`.

**2. France's region-less records.** In train, a record with no detectable region is a
true copy 95.5 % (India) / 97.7 % (US) of the time vs 73 % otherwise (the noise drops the
state from 4–5 % of true copies and from 0.3–0.6 % of decoys). Test US/India look the same
(2.6–2.9 % region-less). **France: 34–36 % of S2/S3 records have no region** (S1: 0 %):
its S2/S3 addresses often end at the city (`77 AV LEON JOUHAUX, LILLE`). The explicit
region features have almost no gain in the models (`r_q_missing` rank 95/102 in stage 1,
116/131 in stage 2), so the direct effect should be small; `D:\aml\analysis\fr_region.py`
checks whether the model treats France's region-less records differently (to run).

**3. New stage-2 options (off by default, unit-tested):**
- `rerank.train_complete` ("fix B"): train stage 2 only on pairs whose S1 lies in a
  loaded region (val side or a hold region), i.e. whose group is complete; the rest are
  still scored. The rerank log now reports incomplete-group pairs per side.
- `rerank.rounds = 2`: a second cross-fitted round whose group features (`g2_*`) are
  recomputed from the first round's out-of-fold p2 instead of p1 (sharper claims and
  competition), next to all round-1 columns; test = round-1 fold mean → g2 features →
  round-2 fold mean. `rounds = 1` is unchanged (same code path, same columns).
- Fair comparison across hold sets: `D:\aml\analysis\clean_eval.py` scores val F0.5 also
  on the "clean" val S1 (none of their kept pairs involves a record with a kept pair to a
  non-val S1), which every variant sees with complete groups.

**4. Run `v25`** (junctions to v2's normalize/candidates/features): `features-hold` with
`hold_frac 0.4, hold_by owner` → 3,990,154 hold records (v2.4: 2,087,204); hold regions
India AP GJ HR MH OD RJ (39.4 % of India's S1), US AL AR AZ CO CT IA ID IN LA MD MO MT NY
OK TN UT WY (38 %). Stage 1 then trains on ~51 % of the S1 regions; `scorer.
train_query_frac` 0.14 keeps its sample at ~3.5 M rows as in v2.4. Experiments:
`D:\aml\run_v25.sh`.

`features-hold` took 79 min (97.5 M hold pairs, peak 8.85 GB). Refactor check (run
`v25chk` = v2.4's stage 1 + hold features, E1 settings, rounds = 1): fold models 0–2
bit-identical to v2.4's; fold 3 stopped early at another iteration (1887 vs 1967; the
first trees are identical), i.e. LightGBM's timing-dependent choice between row- and
col-wise histograms, not the code (max |Δp2| 0.06, mean 0.0005 on the fold's rows). The
v24 rerank log also shows the hold-rule problem directly: 42,605 hold pairs on S1 outside
the hold regions, 35,637 of them positives (own-region rule).

**5. Results** (stage 1 of v25: 3.62 M rows, val AUC 0.999939 / logloss 0.00294, same as
v2.4's 0.999938 / 0.00293; kept pairs: val 1,112,888 of which 102,055 on incomplete groups
(v2.4: 137,258), hold 4,281,230 (2.92 M positives), none of the positives on an incomplete
group (v2.4: 35,637)). "Clean" = the 191,917 val S1 (86 %) whose groups are complete in
every variant; "affected" = the other 30,707, which carry the empty-group artefact and get
harder as more groups are completed, so only the clean column compares hold sets fairly.

| Stage 2 | folds | val F0.5 | clean | clean US | clean India | clean singletons |
|---|---|---|---|---|---|---|
| v2.4 E2: val only | region | 0.98660 | 0.99020 | 0.99347 | 0.98497 | 0.98224 |
| v2.4 E3: val + hold 0.2 (own region) | region | 0.98776 | 0.99125 | 0.99401 | 0.98684 | 0.98703 |
| v2.5 E3: val + hold 0.4 (owner) | region | 0.98789 | 0.99156 | 0.99410 | 0.98750 | 0.98825 |
| **v2.5 E3b: + train on complete groups only** | region | **0.98822** | **0.99171** | 0.99417 | 0.98778 | 0.98903 |
| v2.4 E1: val + hold 0.2 | by S1 | 0.98911 | 0.99241 | 0.99400 | 0.98987 | 0.98703 |

Fix B (`train_complete`) helps on the clean S1 (+0.00015) *and* on the affected ones
(0.96637 vs 0.96496), although its OOF logloss is worse (0.0399 vs 0.0384: it now scores the
artefact pairs, empty groups on val, without the shortcut it never learned). The shortcut
"empty S1 group → reject" was costing real decisions, not just calibration on artefacts.

Production folds (by S1; each fold model sees every region):

| Stage 2 | val F0.5 | clean | clean US | clean India | clean singletons | affected | OOF AUC / logloss |
|---|---|---|---|---|---|---|---|
| v2.4 E1 (LB 0.980973) | 0.98911 | 0.99241 | 0.99400 | 0.98987 | 0.98703 | 0.96844 | 0.999251 / 0.0311 |
| v2.5 E1: hold 0.4 (owner) | 0.98896 | 0.99261 | 0.99421 | 0.99005 | 0.98842 | 0.96617 | 0.999018 / 0.0341 |
| **v2.5 E1b: + complete groups only** | **0.98923** | **0.99270** | 0.99424 | 0.99025 | 0.98921 | 0.96753 | 0.998869 / 0.0359 |

(The affected column and the OOF AUC/logloss are not comparable between v2.4 and v2.5:
v2.5 completes 35 k more val groups, which makes those pairs harder, as at test time.)
**Decision: v2.5 = E1b** (hold 0.4 by owner, train on complete groups, folds by S1): best
in every column of both protocols, above v2.4 on the full val despite its harder val.

On unseen regions the larger, owner-ruled hold set adds +0.0003 on the clean S1 (India
+0.0007, singletons +0.0012) on top of v2.4's +0.0010: diminishing, same direction.
3 of 4 folds stopped at the 2,000-round cap (1662/1998/2000/1995): with 5.4 M rows stage 2
is capacity-limited.

**6. v2.5 test file** (`D:\aml\run_v25_test.sh v2.5_hold04_complete --set
rerank.train_complete=true`, 2026-09-26 21:27): decision sweep flat (0.98916–0.98923 for
min_p 0.02–0.5; 0.02 chosen); `D:\aml\submissions\v2.5_hold04_complete\` = `output\`,
validator PASS, 5,885,129 links, `candidate_pairs.tsv` 9,582,745 pairs = **5.53 per S1**
(v2.4 5.63; the new stage 1 keeps fewer pairs at p1 ≥ 0.001: 1.069 vs 1.078 per val
record, 99.974 % of the retrieved true pairs). Anchor: again only France's generic swaps
(π 0.805 vs mean p2 0.392, de-biased odds ratio 5.40, applied); every US/India kind
0.91–1.74. Links per S1 France 3.445 / India 3.380 / US 3.399 (v2.4 3.420 / 3.397 /
3.420); S1 changed vs v2.4: France 10.0 %, India 5.0 %, US 3.8 %. Default config, README
and methodology updated, zip rebuilt (CRC OK), pushed (8a81125). **Public LB 0.981708**
(2026-09-26 ~21:40, uploaded by the user as the regression check): +0.000735 over v2.4,
more than val showed (+0.00012 full val, +0.0003 clean, +0.00046 unseen-region clean), as
expected if test (every region unseen, every group complete) gains most from these
changes. Uploads 2026-09-26: v2.1, v2.2, v2.4, v2.5 (4 of 5).

**6b. Stage-2 capacity: no gain.** E3b with a 3,500-round cap (`E3b_r3500`): folds stopped
at 1859/2282/2295/2401 instead of ≤ 2000, but val 0.98819 vs 0.98822, clean 0.99168 vs
0.99170 (India −0.0001, US +0.00004): the 2,000 cap was not really binding (flat
validation curve). Dropped; its by-S1 twin was not run.

**6c. Second stage-2 round (`rerank.rounds 2`): rejected.** Unseen regions (E3r2b vs E3b,
both on complete groups): val 0.98831 vs 0.98822 but **clean 0.99134 vs 0.99170** (US
+0.0001, **India −0.0011**, singletons +0.0007; affected +0.003). The fold holding Uttar
Pradesh drops from AUC 0.9897 (round 1) to 0.9827 (round 2). Stacking shift: the round-2
model learns from round-1 p2 of regions the round-1 models had seen, then meets the p2 of
a region no round-1 model saw, which is how every test region looks. The code stays (off)
for the record; its by-S1 twin was stopped to save time.

**6d. Stronger stage-2 regularisation** (run `v25b` = v25's stage 1 via junctions;
`min_data_in_leaf 2000, lambda_l2 10, feature_fraction 0.7`, new config keys
`rerank.lambda_l2` / `rerank.feature_fraction`, defaults unchanged): unseen regions
(E3b_reg vs E3b) val 0.98822 vs 0.98822, clean 0.99172 vs 0.99170 (US +0.00006, India
−0.00005, singletons +0.0003); better calibrated (OOF logloss 0.0391 vs 0.0399, Uttar
Pradesh fold AUC 0.9912 vs 0.9897) but no F0.5 change. By S1 (E1b_reg vs E1b): val
0.98913 vs 0.98923, clean 0.99261 vs 0.99270, singletons 0.98799 vs 0.98921: slightly
worse. Rejected; v2.5's stage-2 parameters stay.

**6e. Smaller candidate set (`rerank.min_p1` 0.003): small cost, not adopted.** Kept val
pairs 1,013,391 vs 1,112,888 (−9 %; 0.97 vs 1.07 per record), 397 fewer true pairs.
Unseen regions (E3b_minp003): val 0.98818 vs 0.98822, clean 0.99163 vs 0.99170; by S1
(E1b_minp003): val 0.98917 vs 0.98923, clean 0.99260 vs 0.99268 (India −0.00015,
singletons −0.0003). The F0.5 leaderboard decides the ranking and 5.53 candidates per S1
is already small, so 0.001 stays.

**6f. Domain-normalisation scale from complete groups (`rerank.norm_complete`).** The
S1-group counts (s_n, c_expected, c_sum_other, ...) are divided by their per-country mean
over the training pairs, which include the ~9 % of pairs with artificially empty groups;
that pulls the mean (and so the scale) down, so every training value sits on a larger
scale than the same density gives at test, where every group is complete. With the option
the S1-group counts take their scale from complete-group pairs only (query-side and
stage-1 counts, IDF columns: unchanged). Val cannot show this (val shares the training
scale; only the pooling of the two countries changes), so it is a candidate for an LB
check, not a validated gain. Run v25b `E1b_nc` + test pass → `D:\aml\submissions\
v2.6_normcomplete\`.
Measured shift (rerank log, complete-only mean / all-pairs mean): c_expected, c_sum_other,
s_n_hi_other ×1.093 (India) / ×1.100 (US), s_n ×1.048 / ×1.021. Val as expected neutral:
0.98918 vs 0.98923, clean 0.99263 vs 0.99268, OOF AUC identical (0.998869), logloss
0.03590 vs 0.03591 (run-to-run noise level). Test file (validator PASS, 2026-09-27 00:35):
5,865,018 links (−0.3 %), links per S1 France 3.424 / India 3.368 / US 3.391 (v2.5 3.445 /
3.380 / 3.399), S1 changed vs v2.5: France 3.2 %, India 1.7 %, US 1.1 %; the model is a bit
stricter about already-claimed S1 (test's claims now read ~9 % larger relative to
training, which is their true relation). Anchor: French generic swaps mean p2 0.344,
de-biased ratio 6.58 → capped at 6 (v2.5: 5.40). candidate_pairs.tsv identical to v2.5's.
Decision rule: upload once as the check; adopt if the LB does not drop (≥ v2.5 − 0.0002),
since it removes a measured train/test mismatch and costs nothing on val; else keep v2.5.
**Public LB 0.982344** (2026-09-27 ~09:00, uploaded by the user): +0.000636 over v2.5 →
**adopted as v2.6** (default `rerank.norm_complete: true`, `output\` = this file, docs,
zip, push). Uploads 2026-09-27: v2.6 (1 of 5).

**7. France's region-less records are not a problem** (`D:\aml\analysis\fr_region.py`,
`fr_region.txt`, v2.5 scores). In France, 34.9 % of the records with a kept pair have no
region; on their same-street, equal-key top pairs the house-number anchor gives a
true-copy share of 0.869 vs the model's mean p 0.873 (odds ratio 0.97; region present:
0.860 vs 0.858), with the same hn profile (equal 77 %, shifted 11 %). In US/India "no
region" is mostly an *empty address* (their top pairs are never on the same street; val
precision of the top pair 0.67–0.70 vs 0.85), a different population; France's are
addresses that stop at the city. They are more often uncertain (8.6 % in p2 0.2–0.9 vs
5.3 %; 12 % have their best candidate on another street, as whole-country searchers),
but nothing says the model misjudges them. No change.

### v2.6 → next (2026-09-27 morning): where the remaining loss is

**Val errors of v2.6** (`D:\aml\analysis\errors_v2.py v25b`, `errors_v26.txt`): 97.14 % of
true pairs kept; dropped (p2 too low, median 0.50) 1.10 %, not a candidate / filtered
0.97 %, another S1 won 0.79 %. Records with an **empty address** keep only 57 % of their
true pairs (about 60 % of all missed pairs; shared names, unresolvable without an
address). 2,990 false links of 751,142.

**Self-estimate** (`selfest.py v25b`, expected F0.5 under the model's own p2, min_p 0.02):
val actual 0.98918 vs expected 0.99471 (optimism 0.0055: US 0.0049, India 0.0065); test
expected US 0.99266, India 0.98921, **France 0.98448**, all 0.98982 → minus val's optimism
≈ 0.9843 vs LB 0.98234: ~0.002 of the gap is invisible to the model (~0.005 in the v2.3 era).

**Where the model is unsure** (`fr_uncertainty.py`, per record's top pair, uncertainty
p(1−p), raw p2 before the anchor correction): per record France 0.0176, India 0.0123, US
0.0090 (val 0.0064–0.0076). France: generic swaps 38 % of its uncertainty (anchor-corrected
×6 in decide), same-street "other" name changes 22 %, empty-address exact names 14 %. On
val every segment is calibrated (mean p2 = true rate within 0.01). On test the biggest
segment, same-street records with an equal name key, has a much lower mean p2 (US 0.762 vs
val 0.865, India 0.791 vs 0.877): test has about twice the same-street lookalikes.

**Is the model calibrated there? Label-free check** (`calib_anchor.py`): given the match
status the house-number relation profile does not depend on the segment, so the true-copy
(h1) and decoy (h0) profiles are taken from *labelled* val and each test country's
histogram is fit as π·h1 + (1−π)·h0 (sanity: on val it recovers the true rate exactly).
**US: π 0.756 vs mean p2 0.762 (odds ratio 0.97), and the mixture reproduces the test
histogram** → the model is calibrated on the US's largest test segment despite twice the
lookalikes; the val→LB gap is not a global over-confidence. India and France cannot be
judged this way: their house-number formats differ from val's (India test has 11 % missing
house numbers vs 20 % in val's true copies; French decoys are 11 % shifted vs 82 % in the US).
**France's blocking** is not losing matches: records with and without a detected region
(the latter search all of France) have a kept candidate equally often (78.7 % vs 78.9 %).

**Bagging stage 2 (`rerank.bags 2`: a second fold split, early-stopping split and seed;
val = mean of the two out-of-fold p2; run v25c `E1b_nc_bag2`): no measurable gain.** OOF
AUC 0.998892 vs 0.998869 / 0.998875 per bag, logloss 0.035677 vs 0.035901; val 0.98920 vs
0.98918, clean 0.99265 vs 0.99263 (singletons −0.0004). Not worth doubling training and
test time; stays off (default 1).

**v2.6 on unseen regions** (run v25c `E3b_nc`): val 0.98817, clean 0.99163 (v2.5's E3b:
0.98822 / 0.99170): neutral, as a change validation cannot see must be.

**Stage-1 domain normalisation: not pursued.** Stage 1 barely uses the density-dependent
features (1.2 % of its gain together; s_name_freq 0.34 %, rank 13); it is dominated by
relative features (cos_combo_margin 71.5 %, cos_combo_gap 8.7 %, hn_best_rel 5.2 %), so
normalising them would hardly move p1 (and val cannot check it).

**Decision (2026-09-27 ~10:40): v2.6 is the final file.** Every principled change that
validation can measure is now flat (tree cap, second round, regularisation, bagging,
smaller candidate set); the remaining test-alignment checks found no further mismatch
(US calibration anchor, France blocking). Official validator with `--check-ids`: PASS
(1,732,544 rows each; 97,150 empty S1 = 5.6 %; 25,130 S1 without candidates).

**LB probe prepared:** `D:\aml\submissions\probe_v24_france_empty\` = the v2.4 file with
all 259,452 French S1 (14.98 % of test S1) left empty (v2.4 left 5.35 % of them empty).
LB(v2.4) − LB(probe) = 0.1498 × (F_France − s_France), so F_France ≈ Δ / 0.1498 +
s_France (French singleton rate, ~0.055 ± 0.01): tells whether France explains the
val → LB gap. Only if the user wants to spend an upload on it.

### Memory: stage 2 and the scorer without the big in-RAM copies (2026-09-26)

The laptop has ~4–5 GB free while the user works (Windows' page file is on D:, 18.6 GB,
and peaked at 11.3 GB during the earlier runs, i.e. stages swapped heavily). The GPU
does not help here: the pip LightGBM has no GPU build (checked) and the memory goes to
data preparation, not model training. Changes (logic unchanged):
- `rerank`: stage-1 scores and features read as Arrow tables; ids compared in Arrow (no
  25.6 M Python strings); the category dtypes over all 12.5 M train / 11.7 M test records
  replaced by integer codes of the kept pairs only (the group features only need the
  grouping); kept rows written straight into one preallocated float32 matrix (no
  per-part frames, no concat copy); LightGBM gets that matrix directly.
- `scorer`: two passes (sample masks from the ids, then rows straight into
  preallocated matrices; no concatenation copy).
- Check (run `v2m` = v2.2 re-run with the new code): identical fold models (best
  iterations 742/1001/1131/857) and **bit-identical** p2 for all 1,118,624 val pairs
  and all 9,649,852 test pairs (max |Δp| = 0.0). Peak RSS: train pass **3.5 GB (was 7.8)**,
  test pass **5.0 GB (was 8.25)**.

**LB probes prepared** (the only way to see test performance by part):
`D:\aml\submissions\probe_france_empty\` (v2.1, every French S1 left empty → France's own
F0.5 by subtraction) and `D:\aml\submissions\probe_stage1\` (v2 without stage 2, val 0.98337 →
does stage 2 transfer to test?). Next: adversarial validation on stage-2 inputs
(`D:\aml\analysis\adversarial.py`) and a test-shaped (importance-weighted) stage 2
(`rerank.weights_file`).

### v2 error budget (val, 770,168 true pairs; `D:\aml\analysis\errors_v2.py`)
| Outcome | Share of true pairs | Notes |
|---|---|---|
| matched | 97.35 % | |
| dropped (p2 too low) | 1.22 % | median p2 0.46: inherently ambiguous cases, e.g. true pairs whose house number moved by 2 *and* city changed (`334 Acrebrook Dr, Northampton` → `336 …, Florence`), i.e. exactly what a distractor looks like |
| never a candidate | 0.98 % | 1/3 have the owner's exact core name but lose a tie: dozens of S1 share it ("Noida Solutions", "Ghaziabad Enterprises", 60 per name country-wide), 2/3 of those have no address at all |
| another S1 won | 0.45 % | mostly made-up brand names at an address shared by several S1 (`489 Boylston St, Unit 2` hosts two S1) and address-less records whose name several S1 share |

- **Records with an empty address** (~4 % of records) lose 32.5 % of their true pairs vs 1.3 % for the rest: **more than half of all missed pairs**. With 50 % of S1 names shared, most of these are unrecoverable.
- False links: 2,765 of 752,513 kept (P 0.9963): 1,842 link an unmatched record (a lookalike), 923 a record owned by another S1.
- Checked and dropped: (i) exact-name blocking (a 4th candidate source) would recover only ~800 val pairs (0.1 %) that still have an address: not worth a 7 h re-run; (ii) brand names are drawn independently per record (an S1's S2 and S3 copies get different brands), so they can't be linked across records.

### Open questions / what the user can help with
- ~~Which file produced the ≈ 0.975 leaderboard score?~~ **Answered 2026-09-25: main's
  `output/matching_results.tsv`** (val 0.98211). So val over-estimates the public
  leaderboard by ≈ 0.007. Candidate causes, to check on the v2 test predictions:
  (a) France (15 % of test S1, no labels anywhere) scoring well below US/India;
  (b) the larger share of unmatched records in test (~40 % vs 26 %) costing precision
  and singletons. Val gains should therefore be read as relative, not absolute.
- Leaderboard feedback on the v2 file, to compare with its val score.

---

## Task board

| ID | Task | Status | Notes |
|---|---|---|---|
| — | EDA (eda/, FINDINGS.md) | ✅ | 2026-09-25 |
| — | Local F0.5 scorer (utils/score.py) | ✅ | 2026-09-25 |
| — | Research notes (docs/RESEARCH_NOTES.md) | ✅ | 2026-09-25 |
| T0.1 | Project scaffold, venv, requirements, license check | ✅ | 2026-09-25; `.venv` (Py 3.12), all deps permissive, pytest green |
| T0.2 | Config, logging, artifact IO, CLI skeleton | ✅ | 2026-09-25; `ber --help` lists all stages; noop writes manifest; 23 tests |
| T0.3 | Geographic train/val split + dev slice | ✅ | 2026-09-25; val US 10.3% / IN 9.8%; dev ≈40k/30k S1 incl. val; leakage (candidate-based) measured in T2.2 |
| T0.4 | eval/metrics.py matching utils/score.py | ✅ | identical to utils/score.py on 3000 random cases |
| T1.1 | Text normaliser | ✅ | exact core-name equality on true pairs +4–8 pts vs EDA baseline |
| T1.2 | Address parser + tables | ✅ | address token Jaccard on true pairs +11–35 pts (US S3 0.48→0.83) |
| T1.3 | Learned Indic→Latin dictionary + fallback | ✅ | val exact-token match 95.8% (dict 84%, rules-only 23%) |
| T1.4 | `normalize` stage (train + test) | ✅ | 24.2M records in 214 s |
| T2.1 | Chunked TF-IDF top-k per country | ✅ | sparse_dot_topn; region-blocked, streamed per block |
| T2.2 | Combine candidates + recall report | ✅ | full: pair recall 99.04%, cluster 96.78%, 24.5 cands/query, 83 min |
| T2.3 | Tune k / index mix to recall targets | 🔄 | full: US 99.3%, India 98.2–99.2% ❌ (targets 99.8/99) — recall is a ceiling; revisit |
| T2.4 | (conditional) Dense ANN fallback | ⬜ | |
| T3.1 | Number-relation feature | ✅ | |
| T3.2 | Edit-type / initials / gibberish features | ✅ | |
| T3.3 | String + context features | ✅ | ~72k pairs/s |
| T3.4 | LightGBM train + calibration | ✅ | dev val AUC 0.99992, logloss 0.0040 |
| T3.5 | Minimal decision + submission (**first LB**) | ✅ | output/matching_results.tsv: official validator PASS; ready for LB upload |
| T3.6 | Router (uncertain band) | ✅ | dev: band [0.05,0.95] = 10.3% of queries, scorer top-1 precision there 0.59 |
| T4.1 | Expected-F0.5 subset selection | ✅ | dev val F0.5 0.97655 (best threshold 0.97615) |
| T4.2 | Duplicate tying / capacity assignment | ✅ | tying: 0 changes on dev (duplicates already agree) → kept as safety; capacity not needed yet |
| T4.3 | `ber explain` + error dumps | ✅ | `ber explain <id>`; dev loss decomposition done |
| T5.1 | Synthetic French pairs + French val proxy | 🔄 | `ber synth-france` + configs/france_proxy.yaml; proxy candidate recall 99.75%; F0.5 needs main model |
| T5.2 | Prompt v1 + token budget | ✅ | select-v1 with hint tags; token count to be measured on Kaggle |
| T5.3 | LLM training data | 🔄 | `ber llm-export` (OOF scores for train prompts); needs full-scale scorer |
| T5.4 | Kaggle LoRA fine-tune Qwen3-1.7B | ⏸ | scripts ready (kaggle/); **needs user: Kaggle account + upload** |
| T5.5 | Kaggle batched inference + throughput | ⏸ | kaggle/infer.py + `ber llm-import` ready; needs T5.4 |
| T5.6 | Blend LLM + GBM (**LB**) | ⬜ | |
| T5.7 | (optional) Qwen3-4B / wider band | ⬜ | |
| T6.1 | France pseudo-labelling | ⬜ | |
| T6.2 | Error-analysis loop | ⬜ | |
| T6.3 | Clean-checkout reproducibility | ⬜ | |
| T6.4 | Documentation + submission zip | ⬜ | |

---

## Metrics

### Validation runs
| Date | Run id | Stage / change | Cand. recall | Val F0.5 | US | India | Indic | Singletons | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-09-25 | dev | first end-to-end: TF-IDF (region-blocked) → LightGBM → one-owner → expected-F0.5 | 99.74% | **0.97655** | 0.98929 | 0.96373 | 0.92228 | 0.97261 | dev index is small (70k S1) → optimistic; Indic val = Punjab (Gurmukhi unseen in train) |
| 2026-09-25 | main | **full scale**: same pipeline; scorer on 13.6M sampled train pairs (AUC 0.99993) | 99.04% | **0.98211** | 0.98539 | 0.97696 | 0.97244 | 0.97707 | pair P 0.9964 / R 0.9555; expected-F robust (0.9820–0.9821 for min_p 0.02–0.5) |

### Leaderboard submissions
| Date | Run id | Val F0.5 | Public LB | Δ vs prev | Notes |
|---|---|---|---|---|---|

---

## Decisions log

| Date | Decision | Why |
|---|---|---|
| 2026-09-25 | Target: beat 0.971 public LB | User goal |
| 2026-09-25 | Build v3 (LLM select matcher) in the first iteration, backed by TF-IDF blocking + GBM router | User choice; see plan §1 |
| 2026-09-25 | All stage recommendations accepted (plan §1) | User approval |
| 2026-09-25 | Compute: Kaggle GPUs for LLM training/inference; local Mac for everything else | User's available compute |
| 2026-09-25 | Unlabelled test data may be used for France (synthetic pairs + pseudo-labels) | User approval; document in methodology |
| 2026-09-25 | Query direction S2/S3 → S1; one owner per record | EDA: all matched ids unique |
| 2026-09-25 | Don't produce candidate_pairs.tsv (`submit.write_candidates=false`) | User: not needed for our submission; it is ~3 GB / 2.4e8 ids and the official validator needs ~30 GB RAM to check it |

---

## Log

### 2026-09-25
- Explored repo, problem statement, data (≈2.4 GB; train 2.2M S1 / 5.0M S2 / 5.3M S3; test adds France).
- Added `utils/score.py` (macro F0.5; tested on the README example and GT vs GT = 1.0).
- Research on prior art → `docs/RESEARCH_NOTES.md`.
- EDA scripts 01–06 → `eda/reports/`, summary in `eda/FINDINGS.md`. Key findings:
  one-owner structure; identical US/India generator; lookalike distractors
  (shifted house numbers, one-letter name edits); name OR address trigram
  similarity keeps 99.9% (Latin) / 98.0% (Indic) of true pairs; France looks
  no harder than US once normalised.
- Wrote `plan.md` and this file.
- **T0.1 done.** Scaffold at `code/business_entity_resolution/` (src layout, `pyproject.toml`, package
  `ber` with stage subpackages, `tests/`, `configs/`, `kaggle/`, `tools/`). Root `.venv` on Python 3.12.0
  (arm64); pinned `requirements.txt`: numpy 2.5.3, pandas 3.0.6, pyarrow 25.0.1, scipy 1.18.1,
  scikit-learn 1.9.1, rapidfuzz 3.14.6, lightgbm 4.7.0, PyYAML 6.0.3, pytest 9.1.1.
  `tools/check_licenses.py` → all permissive. Smoke test passes. `.gitignore`: added `.venv/`, `artifacts/`,
  `*.egg-info/`, `.pytest_cache/` (and fixed a missing newline that merged two entries).
  Notes: pandas 3.x defaults (string dtype, copy-on-write) apply; local machine has **16 GB RAM / 10 cores**,
  so stage 1 must be chunked per country.
- **T0.2 done.** Modules in `src/ber/`:
  - `config.py`: typed dataclass config from YAML; `extends:`; `--set a.b=v` overrides (YAML-typed);
    **strict** (unknown keys / wrong scalar types → `ConfigError`); root auto-detected (dir containing
    `dataset/`, or `BER_ROOT`); `cfg.hash()` excludes run_id.
  - `log.py`: `ber.*` loggers, `timed()` blocks with duration + **peak RSS**.
  - `io.py`: strict TSV reading (`QUOTE_NONE`, no NA parsing, column/prefix checks), GT reader,
    `write_id_lists` enforcing the submission rules (every key once, S2/S3 ids only, dedup).
  - `artifacts.py`: `StageRun` context manager → `artifacts/<run>/<stage>/{log.txt, manifest.json, outputs}`;
    the manifest is written **even on failure** (status + traceback); downstream stages refuse failed/missing
    inputs with an actionable message and warn on config-hash mismatch.
  - `cli.py`: `ber <stage> [-c cfg.yaml] [-s k=v] [-r run]`; `config` and `noop` implemented, the other
    12 stages are registered and name the task that implements them.
  - `configs/default.yaml` (every key documented), `configs/dev.yaml` (`extends`, run_id=dev, slice=dev).
  - Tests: config (extends, overrides, strictness, circular extends, hash), io (id-list format and
    rules, quote handling), artifacts (manifest, failure recording, missing upstream), CLI. **23 passed.**
  - Bug caught by tests: a stage run outside the CLI wrote an empty log.txt (logger level never set);
    `StageRun` now calls `setup_logging` itself.
- `.gitignore` reorganised into sections: data, pipeline outputs (`artifacts/`, `output/`, submission zip),
  EDA caches/reports, `docs/`, model files (`*.safetensors`, `*.bin`, `*.pt`, `*.gguf`, `checkpoints/`,
  `models/`), secrets (`.env`, `kaggle.json`), Python/tool caches, editors/OS. Removed a duplicate `.venv/`.
- **T0.3 done.** `ber split` (`src/ber/eval/split.py`) + `normalize/geo.py` + `normalize/tables/regions.tsv`
  (every state/region spelling seen in the data → code, incl. Indic-script state names; France region↔département).
  - Whole regions are held out per country; the dev slice = small whole regions (dev-val chosen first, part of val).
    S2/S3 follow their owner (matched) or their own detected region (unmatched); undetectable → stable hash.
  - Full-train run: 112 s, peak 4.5 GB. Val: US 10.27% (FL KY MA MN ND NE RI SC SD), India 9.81% (PB UP + 3 tiny).
    Dev: US 27.0k train + 13.1k val S1; India 17.4k train (AP) + 13.0k val (PB).
  - A record's own detected region = its owner's region: US 99.99–100%, India 98.6–98.8% → unmatched
    records are placed reliably. Only ~0.1–0.2% of unmatched records needed the hash fallback.
  - The true leakage rate (val queries with train-side candidates) needs candidates → measured in T2.2.
  - Bugs caught on the real run (both now covered by tests): (1) dev slice had no val (US 0, India 4 S1) because
    all val regions were bigger than the dev-val target → pick dev-val first; (2) India has no state between
    4 and 13k S1, so dev-val was still ~empty → fall back to the region closest to the shortfall. Also
    `make_split` crashed on an empty query frame. 50 tests pass.
- **T0.4 done.** `eval/metrics.py` (per-entity F0.5 + `summarize` by segment), identical to `utils/score.py`.
- **M1 done (T1.1–T1.4).** `normalize/{text,address,translit,geo,lexicon,stage}.py`, tables as TSV
  (`legal_forms`, `name_stopwords`, `credentials`, `street_types`, `addr_junk`, `ocr_digits`, `regions`,
  `translit_overrides`).
  - Name: fold/accents, alias split (aka/fka/dba/formerly/nee), domains, OCR digit fixes, single-letter runs →
    known units or initials, legal forms canonical (with one-typo tolerance for long words: "PRlVATE").
  - Address: placeholders dropped, region components removed → `region`, street types per country,
    unit/house prefixes dropped, zero padding stripped; raw digit strings kept for number features.
  - On 100k real true pairs: exact core-name equality 0.52–0.57 → 0.56–0.65; address token Jaccard
    0.48–0.75 → 0.76–0.86. Speed ~100k fields/s/core.
  - Transliteration: dictionary learned from train-side pairs (1,164 words) + rules + **consonant-skeleton snap
    to the S1 vocabulary** (S1 is fully known at inference, test included). Val exact token match 95.8%.
    Gurmukhi words were entirely unseen in training because Punjab is a val region; the skeleton fallback covers them.
  - Full run: 12.5M train + 11.7M test records in 214 s (8 workers), peak 4.4 GB.
  - New deps: indic-transliteration (MIT; pulls tqdm MPL-2.0 + roman ZPL-2.1, reviewed in check_licenses),
    sparse_dot_topn (Apache-2.0), psutil (BSD).
- **M2 (T2.1–T2.2) done; T2.3 in progress.** `candidates/{tfidf,generate}.py`: three char-3gram TF-IDF indexes
  (name, address, name+address) → union of top-10 each; all 3 cosines + ranks per pair.
  - Full-scale reality check (India, country-wide): combo recall@10 only 95.7% and ~65 min/index →
    **region blocking** with learned neighbourhoods (AP↔TG: Telangana addresses written as "Andhra Pradesh";
    KS↔OK); no-region queries search the whole country; empty-address queries get k_name=30.
  - Streams one block at a time (parts on disk, incremental recall report) → dev peak 2.3 GB.
  - Dev: pair recall 99.74%, cluster recall 99.11%, 23.7 candidates/query. Misses: 45% have an empty
    query address + generic name; the rest are Gurmukhi names with odd transliterations.
  - Cross-split candidates (val query → train S1) 3.5%: descriptive only, no label leakage.
- **M3 (T3.1–T3.4) done.** `features/{numbers,build,stage}.py`: 70+ features (name edit ops incl. substitutions vs
  indels, fuzzy ratios, alias, initials, legal agreement/conflict, vocabulary OOV, name ambiguity; address ratios,
  region agreement, **house-number relation equal/padded/truncated/shifted**, number sets, PO box; TF-IDF
  cosines/ranks; per-query margins/gaps/ranks; per-S1 candidate counts). `scorer/gbm.py`: LightGBM, early stopping
  on val. Dev: AUC 0.99992; top features: rank_combo, num_q_only, cos_combo_gap, hn_logdiff, hn_rel.
- **T4.1 done.** `decide/{expected_f,stage}.py`: one owner per query → per-S1 expected-F0.5 prefix.
  **Dev val F0.5 = 0.97655** (US 0.989, India 0.964; S1 with Indic matches 0.922; singletons 0.973).
- Bug fixes along the way: `list[float]` config fields were rejected (container hint unpacked as a union).
- **Dev loss decomposition** (sum of 1−F over val S1): 78% missed matches only, 11% wrong matches only, 7% singletons
  given a match, 4% both. Missed true pairs: 3,666 were the query's top candidate but dropped (median p 0.32), 626
  assigned to another S1, 318 not in candidates. Wrong kept matches have p 0.54–0.98 → the LLM band.
  `ber explain` example: an S1 with 3 true matches had 6 candidates at p>0.98 (exact-name lookalike + PMB, one-letter
  edit at the same address, exact name with empty address) → group-level reasoning needed.
- **T4.2**: duplicate tying changes nothing on dev (identical records already get identical scores); kept as a safety net.
- **T4.3**: `ber explain <entity_id>` traces a record through normalize → candidates → features → scorer → decide.
- **T3.5**: `ber submit` writes `output/matching_results.tsv` + `output/candidate_pairs.tsv` for all test S1 (integer
  codes for ~10^8 candidate pairs) and runs `utils/validate_submission.py`.
- **LLM stage (T3.6, T5.2, T5.3; Kaggle T5.4/T5.5 prepared)**:
  - scorer writes 3-fold **out-of-fold** scores for train pairs (honest probabilities for training prompts);
  - `ber llm-export`: routes queries with best p in [0.05, 0.95] (dev 10.3%), top-5 candidates in seeded shuffled
    order, prompt `select-v1` with hint tags (name edit kind, legal form, house-number relation, extra numbers/PO box,
    region, address similarity, scorer p); labels 1..5 / 0 = none;
  - `kaggle/train_lora.py` (Qwen3-1.7B, LoRA r=16, fp16, loss = CE over the 10 answer digits), `kaggle/infer.py`
    (single forward pass, digit probabilities), `kaggle/README.md` (workflow);
  - `ber llm-import`: blends p = w·p_llm + (1−w)·p_scorer for routed queries, w chosen on val;
    `decide.scores_stage=llm-import` uses it.
- **T5.1**: `ber synth-france` builds labelled French data from 20k test S1 France records (noise catalogue + lookalike
  distractors; 83k S2/S3 records, 69k true pairs). `configs/france_proxy.yaml` runs the pipeline on it with the main
  model (`scorer.model_from_run`). Proxy candidate recall 99.75%.
- Infrastructure: `data.with_test` / `cfg.run_test`; stages can read inputs from another run (`data.split_run`).
- Performance lesson: two heavy jobs at once on 16 GB swap badly (the candidate run dropped to 77 MB resident) →
  run heavy stages one at a time. Candidate generation profile (Gujarat, 260k queries × 57k S1): 37 s; sparse
  top-k dominates. Empty-address queries now search only the name index.
- **Full-scale run (main)**:
  - candidates: 83 min (train 48 min, test 34 min), peak 9.5 GB; pair recall **99.04%** (US 99.3%, India 98.2–99.2%),
    cluster recall 96.8%, 253M train / ~240M test pairs. Region neighbourhoods got bloated by tiny misdetected regions
    (min_share 0.001 too low for small regions) — cost, not correctness.
  - features: 101 min (71M train pairs incl. 20% of train-side queries + all val; 244M test pairs), peak 9 GB.
  - scorer: streams parts, float32, query-level sample (30% of featurised train queries = 13.6M rows; 20% of val for
    early stopping); AUC 0.99993, logloss 0.0031, best iteration 548; 3-fold OOF AUC 0.99994; 34 min incl. test.
    Top features: cos_combo_margin, cos_combo_gap, num_q_only, cos_combo_rank, num_s_only, n_jw_margin, hn_logdiff, hn_rel.
  - decide: **val F0.5 0.98211** over 222,624 val S1 (US 0.9854, India 0.9770, Indic-matched 0.9724, singletons 0.9771).
- **submit**: `output/matching_results.tsv` (1,732,544 rows; 5,835,284 matched pairs; 94,844 empty = 5.5%, singleton
  rate in train 5.6%) — **official validator PASS**; own checks: 0 unknown ids, 0 one-owner violations.
  - Gotchas: `decide` test loaded all 244M test scores with string ids → 15 GB swap; now streams one-owner per part
    (peak 9.5 GB, 2.7 min). The official validator, when `--candidate` is omitted, silently loads
    `./output/candidate_pairs.tsv` if it exists → ~30 GB → every CLI run hung. `ber submit` now runs it from `utils/`
    on the matching file only; candidate_pairs.tsv is off by default (user decision). `submit` now takes 18 s.
- Next: user uploads to LB; France proxy F0.5; LLM export; candidate recall (T2.3).

---

## Open questions
- Local RAM is 16 GB: full-train TF-IDF top-k (T2.1) must stream per country/chunk; watch peak memory.
- Kaggle specifics to confirm at T5.4: GPU type available (P100 vs 2×T4), session RAM, dataset size limits.

## Exploration branch `explore/cross-encoder` (2026-09-27): a different architecture

The user asked for a genuinely different architecture next to the GBM cascade (v2.6 stays
the submission on `improve/v2`; this work lives on its own branch). Downloads approved by
the user: PyTorch 2.14 + CUDA 12.6 (download.pytorch.org), transformers / tokenizers /
huggingface_hub / xgboost (PyPI), model `cross-encoder/ms-marco-MiniLM-L-6-v2`
(Hugging Face, Apache-2.0, 22 M parameters). Scripts: `D:\aml\ce\{prep,train,score,
alone_eval}.py`; data `D:\aml\ce\data`, models `D:\aml\models`.

**Transformer cross-encoder (Ditto-style).** Both records' normalised text
(`name core + legal form ; address + region`) is read jointly by a pretrained MiniLM
fine-tuned to predict the match. Training pairs: stage 1's own regions (train side,
outside the hold regions), only records stage 1 did *not* train on (stable_unit(q_id,
seed + 4) ≥ 0.14), kept at p1 ≥ 0.001 exactly as stage 2's inputs → 982,265 pairs (68 %
true), so the cross-encoder is out-of-sample on val, hold and test. RTX 4050, fp16, batch
256, max 64 tokens (3 % truncated), 2 epochs at ~2,000 pairs/s (a first run hit CUDA
out-of-memory in the epoch-end validation pass; fixed with smaller, OOM-safe inference
batches). Results on 200 k val stage-2 pairs:

| | AUC | log loss |
|---|---|---|
| stage 1 (p1) | 0.99608 | |
| stage 2 (v2.6 p2) | 0.99886 | ~0.031 |
| cross-encoder, epoch 1 / 2 | 0.97341 / **0.97452** | 0.306 / 0.307 |

Training log loss was ~0.075 against 0.307 on val: the transformer generalises poorly from
the training regions to new ones (street and city strings), and it sees house numbers only
as word pieces ("848" vs "843"), where the lookalike decoys differ. **As a standalone
matcher it is clearly worse than the GBM cascade.** Complementarity on all 1.11 M val
pairs (5-fold logistic stacking grouped by S1): log loss 0.03574 → 0.03526, AUC 0.998869 →
0.998891; in the uncertain band (p2 0.05–0.95, 69 k pairs) the cross-encoder alone has AUC
0.683 (p2 0.862, stacked 0.864): the remaining hard cases are ambiguous in the text itself.
