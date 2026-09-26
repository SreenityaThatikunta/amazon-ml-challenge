# Plan: Business Entity Resolution (Amazon ML Challenge 2026)

**Goal:** beat the public leaderboard top of **0.989 macro F0.5** (as of 2026-09-26; was 0.986 on 09-25, 0.971 before).
**Current best:** `improve/v2` v2.2 → val 0.98897, **public LB 0.980** (v2.1 0.977; main v1 0.975, main v2 0.972; `Rudy_commit_1` 0.961).
**§0 below (plan v4) is the active plan and replaces the priority order in §4.**
**Inputs:** [eda/FINDINGS.md](eda/FINDINGS.md), [docs/RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md).
**Tracking:** every task, run, metric and decision is logged in [progress.md](progress.md).

---

## 0. Plan v4: from 0.980 to > 0.989 (2026-09-26)

Supersedes plan v3 (git history: `plan.md` at 54d96e9) and the priority order in §4.
Work happens on branch **`v3`** = `improve/v2` v2.2 + main's features (E19) + `features.prune_test` (E22).

### 0.1 Evidence so far

| File | Val F0.5 | Public LB | What changed |
|---|---|---|---|
| main v1 | 0.98211 | 0.975 | baseline |
| main v2 | 0.98906 | 0.972 | + rescore, lookalike features, **synthetic-French model** |
| improve/v2 v2.1 | 0.98895 | 0.977 | + v2 features, stage-2 rerank |
| improve/v2 **v2.2** | 0.98897 | **0.980** | + stage-2 **domain normalisation** (val: no change) |
| Rudy_commit_1 | ? | 0.961 | separate pipeline |

What these five scores say:
1. **The val/test density shift is where LB points are, and val can't see it.** v2.2 only rescaled 20
   size-dependent stage-2 features per country and split: val +0.00002, LB **+0.003**. Val gains that ignore
   the shift don't transfer (v2: val +0.007 → LB +0.002; main v2: val +0.007 → LB −0.003).
2. **The shift is still in stage 1.** The stage-1 scorer is trained at train density and scores test with raw
   `s_n_cands`, `s_n_top1`, `s_name_freq`, `q_n_cands`, the `k_idf_*` / `a_idf_*` IDF features and the
   TF-IDF cosines/ranks (IDF fitted on each pool). Its p1 also decides which pairs reach stage 2 (p1 ≥ 0.001).
3. **The test shape differs by country** (S1 counts in the data):

   | | train S1 | test S1 | test/train | records per S1, val → test | records with p1 < .05 per S1, val → test |
   |---|---|---|---|---|---|
   | US | 1,323,633 | 663,106 | **0.50** | 4.67 → 5.76 | 1.14 → 2.12 |
   | India | 883,188 | 809,986 | **0.92** | 4.68 → 5.82 | 1.15 → 2.14 |
   | France | — | 259,452 | — | → 5.53 | → 1.90 |

   So US test = a half-size S1 pool; India test = nearly the same pool size with about 1 extra junk record per S1.
   A test-shaped validation needs **two knobs**: an S1 thinning rate and extra unmatched records.
4. **France is unmeasured.** It has 3× the ambiguous records per S1 (p1 .5–.95: 0.37 vs 0.12–0.17), and
   synthetic-French training hurt on the LB. The probes are built (`output/probes/`) but not uploaded.
5. **Budget:** 5 submissions a day (2 used on 2026-09-26). The gap to the top is **+0.009**; aim for ≥ 0.991.

### 0.2 Levers and expected gains (to be replaced by measurements)

| # | Lever | Evidence | Expected LB |
|---|---|---|---|
| L1 | v3 = v2.2 + main's lookalike/relative features + `k_name_noaddr` 100 | dev +0.003 over v2's dev | +0.000–0.002 |
| L2 | Remove the shift from **stage 1** (domain-normalise or drop the density features) | same mechanism as v2.2's +0.003 | +0.001–0.003 |
| L3 | **Test-shaped data**: validate *and* train on a val/train side reshaped to test density | fixes the cause, not just the symptom | +0.002–0.004 |
| L4 | Stage 2 trained on far more data (train side as well as val, OOF p1; cheap now with pruning) | stage 2 is the strongest stage; now trained on 222k val S1 only | +0.001–0.002 |
| L5 | France: per-country probes → pseudo-label training / France thresholds | France ≈ 0.94 est. (15% of S1) | +0.002–0.006 |
| L6 | LLM select on the remaining ambiguous band | France-heavy band | +0.001–0.003 |
| L7 | Seed/feature-set ensemble of stage 2; per-country expected-F `min_p` on the test-shaped val | small but reliable | +0.0005–0.001 |

### 0.3 Step 0 (today, no compute): spend the remaining 3 submissions on probes
| Probe | File | Answers |
|---|---|---|
| P1 | v2.2 with France left empty (build on Windows: `make_probe.py empty <v2.2>.tsv France out.tsv`) | **France F0.5 of our best file**; how much of the 0.009 gap is France |
| P2 | `output/probes/p_v2_with_v1_france.tsv` | did synthetic-French training cause main v2's drop? |
| P3 | `probe_stage1` (prepared on Windows) or v2.2 with **US** left empty | stage-2 transfer on test, or US share |

Log each result in progress.md. From P1: F_FR ≈ (0.980 − LB_P1) / 0.1498 + 0.05.
**Decision rule:** if F_FR ≥ 0.975, France isn't the bottleneck, so L2–L4 come first. If F_FR ≤ 0.96, L5 moves up to run in parallel with L2.

### 0.4 Phase 1 (days 1–2): v3 on the LB, plus the stage-1 shift fix
| ID | Task | Exit criterion |
|---|---|---|
| 1.1 | Full v3 run, two passes (`data.phase=train` → `test`, `features.prune_test` on), run `v3`. Mac, ~5–7 h; or the Windows machine (GPU top-k, ~3–4 h) | validator PASS; the re-ranker's kept-pair count equals the pruned test pairs (checks E22 on real data); **LB** (isolates L1) |
| 1.2 | `scorer.domain_norm` for **stage 1**: the same count/IDF scaling as `rerank.domain_norm`, per country and split, applied before stage-1 training and scoring. Reuse `rerank.domain_normalize`, extended with `rank_*` and the TF-IDF cosines if the adversarial check flags them | unit test; adversarial AUC (val vs test, stage-1 inputs) drops; val not down |
| 1.3 | E18 variant: `scorer.drop_features=[s_n_cands, s_n_top1]` instead of normalising | compared with 1.2 on the Step-2 val; keep the better one |
| 1.4 | Test pass only for 1.2/1.3 (the test candidates are reused; features, scorer, rerank, decide and submit are rerun, ~2 h) | **LB** (isolates L2) |

### 0.5 Phase 2 (days 2–3): a validation that sees the shift, then train on it
| ID | Task | Exit criterion |
|---|---|---|
| 2.1 | **`data.reshape`** (in `split`, config-driven, deterministic hash): per country, (a) keep a fraction of S1 (US ~0.5, India ~0.92) and turn their S2/S3 copies into unmatched records (owner cleared), (b) optionally inject extra unmatched records by copying real unmatched records from other regions of the same country. Applies to the train and val sides; test untouched. Tune both knobs so the val side matches test on records/S1, p1 bands/S1, `s_n_cands` / `s_name_freq` quantiles | Shape table per country within ~10% of test; adversarial AUC val_ts vs test ≤ 0.75 (was 0.986) |
| 2.2 | **Gate:** val_ts must reproduce the known LB deltas: v2.2 − v2.1 (`rerank.domain_norm` on vs off) ≈ +0.003, and v3 vs v2.2 with the sign of 1.1 | Both signs right, magnitudes within 2×. If it fails, fix 2.1 before trusting any val_ts number |
| 2.3 | **Test-shaped training (L3):** stage 1 and stage 2 trained on the reshaped train/val sides; several reshaping seeds give extra training draws | val_ts up; **LB** |
| 2.4 | **Bigger stage 2 (L4):** featurise 100% of train-side queries, keeping only p1 ≥ 0.001 pairs (the pruning trick), with honest OOF stage-1 p1 on the train side (`scorer.oof_folds`); train stage 2 on train + val groups | val_ts up; **LB** |
| 2.5 | Decisions per country: `min_p` and the "none" prior chosen on val_ts per country (the p calibration moves with the density) | val_ts up |

From here on **every decision is made on val_ts**; plain val is only a regression check.

### 0.6 Phase 3 (days 3–5, driven by P1): France
| ID | Task | Exit criterion |
|---|---|---|
| 3.1 | France error dump on the new test predictions: the p .5–.95 band by T7.1 tag (S1 twins at the same address, legal-form conflict, different street at the same number, short numbers, initials) | Top 3 tags with counts |
| 3.2 | **Pseudo-label training:** confident France test pairs (p2 > 0.99, unique owner, clear margin) as positives; runner-ups / rejected twins as hard negatives; added to stage 1 and stage 2 training at a weight chosen on the P1-style probe. Rehearse on US/India val_ts first (pseudo-labels from a model that never saw them must raise val_ts) | Rehearsal gain on val_ts; **France probe up** |
| 3.3 | Twin-aware decision: when ≥ 2 S1 share an address (13% of France S1), assign each record to the twin that fits its name, legal form and initials best, or to none if they tie | unit tests; val_ts not down; **LB** |
| 3.4 | France thresholds: `min_p` / "none" prior for France set so that links/S1 and the empty rate match US/India on test (shared generator) | **LB** |

### 0.7 Phase 4 (days 4–6, only if still < 0.991): LLM band + ensemble
- 4.1 Band = records with p2 in [0.05, 0.95] after phases 1–3. Measure the queue size; it must fit ~30 Kaggle GPU-h a week
  (Qwen3-1.7B, letter probabilities from one forward pass; the code is in `kaggle/`, `ber llm-export/import`).
- 4.2 Train on val_ts OOF prompts + France pseudo-labels; blend weight chosen on val_ts; **LB**.
- 4.3 Stage-2 ensemble (3 seeds × {domain-norm, test-shaped}) averaged before decide; **LB**.

### 0.8 Machines and runs
- **Mac:** v3 runs (`artifacts/v3`, ~15–20 GB with pruning; 138 GB free). Heavy stages one at a time.
- **Windows laptop** (has the v2.2 artifacts + GPU top-k): probes P1/P3, and quick stage-2-only experiments
  on v2.2 (e.g. stage-1 domain norm via a rescored test pass) while the Mac runs 1.1.
- Every run and LB score goes into `experiments/README.md` (E-numbers) and progress.md.

### 0.9 Order, gates and submission budget (5/day)
```
Day 0  Step 0 probes P1–P3 (3 subs) ─────────────────────────────► France decision rule
Day 1  1.1 full v3 run ► LB │ 1.2/1.3 code + tests │ 2.1 reshape code
Day 2  1.4 stage-1 shift fix ► LB │ 2.1–2.2 val_ts + GATE
Day 3  2.3 test-shaped training ► LB │ 2.4 bigger stage 2 ► LB │ 3.1 France dump
Day 4  3.2–3.4 France ► LB (probe first)
Day 5+ 4.x LLM band / ensemble if < 0.991; final pick = best LB file that val_ts agrees with,
       reproducible from a clean checkout (T6.3) with docs + zip (T6.4)
```
- Ship a change only if it improves val_ts (after the gate) **and** doesn't lower the France probe.
- Keep ≥ 1 submission a day spare for a final re-submission.

---

## 1. Decisions (agreed 2026-09-25)

| Stage | Decision |
|---|---|
| 0 Normalisation | Rule tables from the EDA (suffixes, abbreviations, states/départements, junk tokens, scanner-typo map), plus an Indic→Latin **word dictionary learned from training pairs**, falling back to rule-based transliteration. Keep raw **and** normalised fields. |
| 1 Candidates | Character 3-gram TF-IDF top-k, **per country**, on (a) names, (b) addresses, (c) transliterated names, combined. Queries go **from each S2/S3 record to S1**. Dense-embedding ANN only as a fallback if recall falls short. |
| 2 Cheap scorer | LightGBM on similarity + context features (number relation, edit type, suffix/initials, rank/gap). It settles confident cases and sends the **uncertain band** to the LLM. |
| 3 LLM matcher (v3) | **Select framing**: one S2/S3 record + its top-N S1 candidates → `A/B/…/none`, scored from a single forward pass (letter probabilities, no text generation). **Qwen3-1.7B or 4B** (Apache-2.0) + LoRA, with **hint tags** in the prompt. |
| 4 Decision | One owner per S2/S3 record → per-S1 **expected-F0.5 subset selection** (with "none" as an option) → tie exact duplicate records together. Try capacity-constrained assignment and keep it if it helps. |
| 5 Validation / France | Split S1 by **state/region** (whole geographies held out). French normalisation tables + **synthetic French training pairs** built from unlabelled test S1 France records. Pseudo-labelling is allowed (user-approved). |
| 6 Compute | Local Mac for stages 0–2 and 4. **Kaggle GPUs** (P100 or 2×T4, ~30 GPU-h/week, 12 h sessions) for LoRA training and LLM inference. |

---

## 2. Architecture

```
dataset/*.tsv
   │
   ▼
[0] normalize ──► artifacts/<run>/records.parquet        (raw + normalised + parsed fields)
   │
   ▼
[1] candidates ─► artifacts/<run>/candidates.parquet     (query_id, s1_id, source_index, tfidf_sim, rank)
   │                       └─► recall report (per country / script / source)
   ▼
[2] features ──► artifacts/<run>/features.parquet
    scorer    ─► artifacts/<run>/scores_gbm.parquet      (p_match per pair)
    route     ─► artifacts/<run>/llm_queue.parquet        (uncertain queries only)
   │
   ▼
[3] llm (Kaggle) ► artifacts/<run>/scores_llm.parquet     (p per candidate + p_none per query)
   │
   ▼
[4] decide ────► artifacts/<run>/decisions.parquet
    submit    ─► output/matching_results.tsv, output/candidate_pairs.tsv
   │
   ▼
[5] evaluate ─► F0.5 by country / script / singleton, error dumps
```

Each stage reads the previous stage's artifacts and writes its own, along with
a `manifest.json` (config hash, git commit, row counts, timings). Any stage can
be re-run, inspected or swapped on its own.

### Code layout (`code/business_entity_resolution/`)

```
README.md                  how to reproduce end-to-end
requirements.txt           pinned
configs/
  default.yaml             paths, k values, thresholds, model names
  dev.yaml                 small-slice overrides (fast iteration)
src/ber/
  config.py                typed config (dataclasses) + yaml loading
  log.py                   structured logging, stage timers
  io.py                    TSV readers, parquet read/write, submission (id-list) writers
  artifacts.py             StageRun: per-stage dir, log.txt, manifest.json (status/timing/memory/
                           config hash/git/inputs/outputs/metrics), upstream input resolution
  normalize/
    text.py                case/accents/punct, junk tokens, suffix split, typo map
    address.py             number extraction, street types, state/region/département maps
    translit.py            learned Indic word dictionary + rule fallback
    tables/*.tsv           all lookup tables as editable data (not in code)
  candidates/
    tfidf.py               char n-gram vectoriser + chunked sparse top-k
    generate.py            per-country indices, combine sources, recall report
  features/
    numbers.py             number relation: equal/truncated/padded/dropped/shifted
    edits.py               edit-type classifier: scanner-typo vs arbitrary; credential swaps
    strings.py             similarity measures (rapidfuzz), initials, gibberish flag
    build.py               pair + context features (rank, gap, counts)
  scorer/
    gbm.py                 LightGBM train/predict, calibration
    route.py               confident vs uncertain split
  llm/
    prompt.py              select-prompt builder with hint tags (one place, versioned)
    data.py                training examples from candidates (hard negatives)
    train.py               LoRA fine-tuning (runs on Kaggle)
    infer.py               batched letter-probability scoring (runs on Kaggle)
  decide/
    expected_f.py          per-S1 expected-F0.5 subset selection
    assign.py              one-owner assignment, duplicate tying, capacity option
  france/
    synth.py               synthetic French pairs + distractors from the noise catalogue
  eval/
    metrics.py             macro F0.5 (same rules as utils/score.py)
    split.py               geographic train/val split
    report.py              per-stage metrics, error dumps
  explain.py               trace one entity through every stage (debug tool)
  cli.py                   `ber <stage> --config ...`
tests/                     pytest: normalisation, numbers, edits, expected-F, metrics
kaggle/                    notebooks for LoRA training and inference
```

### Code conventions (clean and debuggable)
- **One stage = one CLI command = one artifact.** Stages never depend on each
  other's in-memory state.
- **Config over constants.** Every threshold, k and model name lives in yaml.
  The run dir stores the resolved config.
- **Lookup tables are data** (`tables/*.tsv`), reviewed and extended without
  touching code.
- **Dev slice:** `--config configs/dev.yaml` runs the full pipeline on a few
  states in minutes. Use it for every change before a full run.
- **`ber explain <entity_id>`** prints raw/normalised records, candidates with
  features, GBM score, the LLM prompt and probabilities, and the final decision.
  It's the main debugging tool.
- Deterministic seeds; type hints; small pure functions; pytest for every
  normalisation and decision rule. No notebooks in the pipeline itself (Kaggle
  notebooks only call `src/`).
- License check for every dependency and model added (MIT/Apache/BSD only;
  final model ≤8B).

---

## 3. Validation protocol

- **Split:** hold out ~10% of S1 per country **by state** (US state code, Indian
  state). A query S2/S3 record goes to the side of the S1 it matched, or, if
  unmatched, to the side of its top candidate. Report the leakage rate (val
  queries with a train-side candidate).
- **Dev slice:** 2–3 small states per country, for fast iteration.
- **Metrics at every stage:**
  - candidate recall (pairs and whole S1 clusters) and candidates per query
  - GBM F0.5 on its own, and the share of queries routed to the LLM
  - final macro F0.5 overall / per country / Indic vs Latin / singletons
  - pair precision and recall
- **France proxy:** synthetic French validation set (from T5.1) as a stand-in,
  plus a sanity check on the pseudo-pair statistics from `06_test_france`.
- **Leaderboard submissions** only at milestones. Record each one in progress.md
  with its validation score, so we can see how well validation predicts the
  leaderboard.

---

## 4. Re-plan after first LB submission (2026-09-25)

### Gap analysis
- Test S1 mix: **US 38% / India 47% / France 15%** (France absent from train and val).
- If US (0.985) and India (0.977) transfer from val, LB 0.975 implies **France ≈ 0.94**.
  Even a perfect France gives only ≈ 0.984, so **US/India must also improve** to pass 0.986.
- France test signals: two strong candidates (top > 0.5, runner-up > 0.2) on 0.79% of
  queries vs 0.24% for the US; 3.48 predicted matches/S1 vs 3.37 for the US; 4.9% empty vs
  5.6% → over-matching. Synthetic France proxy (`artifacts/synthfr`): F0.5 **0.888**,
  singleton acc 0.545 (proxy may be harsher than the real data).
- France failure modes seen in test (wrong runner-ups scored high):
  - same number, different street ("Rue du Magasin" vs "Rue du Chateau") → p 0.90
  - legal form conflict at same address (SAS vs SARL) → p 0.85
  - initials mismatch ("TL" vs "Tourcoing Amicale") → p 0.74
  - S1 names are "<City> <generic word> <legal form>" ("Lille Club EURL"): the city token in
    the name carries almost no information but counts as name evidence.
- Val loss: **80% missed matches only**, 10% wrong matches only, 7% singletons given a match.
  Pair recall 0.955 vs candidate recall 0.990 → ~3.5 pts of recall lost at scorer/decision,
  ~1 pt at blocking.

### Priority order (supersedes the M5 → M6 order below)
1. **M7 France fixes + pseudo-labels** (local), since France is the largest per-entity gap.
2. **M8 second-stage cluster-context scorer** (local), the main US/India recall lever.
3. **M5 LLM matcher** on the band that M7/M8 still can't settle (Kaggle).
4. **T2.3/T2.4 candidate recall** (India 98.2%; Indic names, empty addresses).
5. **M6** polish, reproducibility, docs.

Every change is validated on val (US/India) **and** the France proxy, and each LB submission is
logged in progress.md with the val → LB delta.

### M7: France (new, priority 1)
| ID | Task | Exit criterion |
|---|---|---|
| T7.1 | France error dump: top wrong/ambiguous test queries + proxy errors via `ber explain`; check that the proxy distractors resemble real test lookalikes, and fix `france/synth.py` if not | Written catalogue of France failure modes in progress.md; proxy realism verdict |
| T7.1b | Proxy fix (T7.1 verdict: unrealistic): index **all** France S1 as confusers, generate copies for a sample and score only the sampled S1; draw (S2, S3) cluster sizes **jointly**; add unrelated unmatched records (real France queries with p1 < 0.01); keep the proxy distractor mix close to the real tag rates in `eda/reports/07` | Proxy stats (p1 bands, empty S1, name-freq p90, ambiguity) within ~2× of real France |
| T7.2 | Name feature: strip the record's **own city/region token** from the name before comparing; feature for "city is the only shared word" | Tests; proxy F0.5 up; val not down |
| T7.3 | Street-name mismatch at the same number (incl. street type: Impasse vs Rue); initials-mismatch feature; **relative context features**: another candidate of the same query has an equal house number / same street / same legal form / matching initials; TRUNCATED only for ≥ 3-digit numbers (French 13↔3 and 103↔10 are distractors) | Tests; wrong runner-up p in the T7.1 examples drops below 0.2; val not down |
| T7.4 | Pseudo-labels: confident France test decisions (p > 0.99, unique owner, clear margin) as positives + their runner-up candidates as hard negatives → retrain scorer with them | Proxy F0.5 up; France predicted matches/S1 and empty rate move toward US/India |
| T7.5 | Per-country calibration / decision check for France (predicted match count, empty rate) | **LB submission**; France estimate from the LB delta logged |

### M8: Cluster-context scorer (new, priority 2)
| ID | Task | Exit criterion |
|---|---|---|
| T8.1 | From stage-1 scores, build per-S1 confident clusters (one owner per query) | Artifact `clusters.parquet`; cluster stats logged |
| T8.2 | Second-order features: max/mean similarity (name, address, numbers) of the query to the S1's other confident members; cross-source agreement (S2↔S3); remaining capacity vs the 5 S2 / 6 S3 cap; the query's rank among the S1's candidates | `features2` built with OOF stage-1 scores for train (no leakage) |
| T8.3 | Stage-2 LightGBM → decide | **Val pair recall up at equal precision; val F0.5 gain logged; LB submission** |

### Status after the first build day (2026-09-25 evening)
- Done: T7.1 (error dump), T7.1b (realistic proxy), T7.2/T7.3 (lookalike + relative features),
  unseen-country scorer with synthetic French training (replaces T7.4 for now), M8 rescore (+ all
  stage-1 features). **Dev val 0.97655 → 0.98859.** Details: `experiments/README.md` E1–E17.
- Running: full-scale v2 (features → scorer → rescore → decide → submit). The next leaderboard check
  needs the user's upload.
- Next (v3 upstream rebuild): all-pairs transliteration dictionary for test (E15),
  `candidates.k_name_noaddr` 30 → 100 (E16). Then the LLM stage on what's left, if time allows.

### Updates to existing tasks
- **`output/candidate_pairs.tsv` is required** in the final zip (README: the exact set the model runs
  inference over, one row per test S1, matches ⊆ candidates). Reverses the earlier "don't produce it"
  decision. `submit.write_candidates` is now on by default and `ber submit` checks matches ⊆ candidates
  itself (the official validator needs ~30 GB for that check). Whenever the pipeline changes, the file
  must be the **last** candidate set before the model: stage-1 candidates while the GBM scores all of
  them; if M8 or the LLM ever prunes candidates before scoring, write the pruned set instead.
- T5.3/T5.6: route only the queries still uncertain after M7/M8; include France pseudo-labels in LLM training data.
- T6.1 is replaced by T7.4.
- T2.3: India recall 98.2% remains open; do after M8 unless an error dump shows blocking misses dominate.

---

## 5. Tasks

Status is tracked in [progress.md](progress.md). Each task has an **exit
criterion**, and it isn't done until that criterion is met and logged.

### M0: Foundations
| ID | Task | Exit criterion |
|---|---|---|
| T0.1 | Project scaffold: package layout, venv, pinned requirements (pandas, pyarrow, numpy, scikit-learn, rapidfuzz, lightgbm, pytest, pyyaml), license check | `pip install -r requirements.txt` works; `pytest` runs |
| T0.2 | Config, logging, artifact IO with manifests, CLI skeleton | `ber --help` lists stages; a no-op stage writes a manifest |
| T0.3 | Geographic train/val split + dev slice | Split stats logged; leakage rate reported |
| T0.4 | `eval/metrics.py` matching `utils/score.py` on shared test cases | Unit tests pass; identical score on a sample file |

### M1: Normalisation (stage 0)
| ID | Task | Exit criterion |
|---|---|---|
| T1.1 | Text normaliser: case, accents, punctuation, wrappers (`>>`, `--`, brackets), junk tokens (aka/formerly/dba/fka/nee, www/.com, titles), legal-suffix split, scanner-typo map | Tests from FINDINGS §4 examples pass |
| T1.2 | Address parser: numbers (+ bis/ter/letters), street-type tables US/FR, state↔code US/IN, region↔département FR, junk (`null`, `N/A`, `##`, `PMB`, `CITY`), Indic-script state names | Tests pass; on true pairs, token overlap goes up compared with raw text |
| T1.3 | Learned Indic→Latin word dictionary from train pairs + rule fallback (indic-transliteration) | ≥90% of Indic tokens in val true pairs map to the S1 token |
| T1.4 | `normalize` stage over train + test | `records.parquet` for both splits; timings logged |

### M2: Candidate generation (stage 1)
| ID | Task | Exit criterion |
|---|---|---|
| T2.1 | Chunked char-3gram TF-IDF top-k per country (name, address, transliterated name) | Runs on full train on the Mac in reasonable time and memory |
| T2.2 | Combine + dedup, ranks per source index, recall report | Report per country/script/source |
| T2.3 | Tune k and the index mix | **Pair recall ≥ 99.8% (Latin), ≥ 99% (Indic)** at ≤ ~20 candidates/query; if not met → T2.4 |
| T2.4 | *(conditional)* Dense ANN fallback (multilingual-e5-small / bge-m3, MIT) for missed segments | Recall targets met |

### M3: Features + GBM + routing (stage 2), **first submission**
| ID | Task | Exit criterion |
|---|---|---|
| T3.1 | Number-relation feature (equal/truncated/padded/dropped/**shifted Δ**) | Tests on FINDINGS §3 examples |
| T3.2 | Edit-type features (scanner typo vs arbitrary; credential/suffix swaps; initials; gibberish-name flag) | Tests |
| T3.3 | String similarities (rapidfuzz) + context features (rank, gap to best/2nd, n candidates, S1 name frequency) | `features.parquet` built |
| T3.4 | LightGBM train/val, calibration | Val log-loss/AUC logged; feature importance saved |
| T3.5 | Minimal decision (argmax + threshold) + submission writer + validator | **Val F0.5 logged; first leaderboard submission** |
| T3.6 | Router: uncertain band by calibrated probability / margin | Share routed + GBM error rate inside/outside band logged |

### M4: Decision layer (stage 4)
| ID | Task | Exit criterion |
|---|---|---|
| T4.1 | Per-S1 expected-F0.5 subset selection (incl. empty) | Unit tests; val F0.5 ≥ argmax+threshold |
| T4.2 | Duplicate-record tying + (optional) capacity-constrained assignment | Kept only if val F0.5 improves |
| T4.3 | `ber explain` debug tool + error dumps (worst FPs/FNs per segment) | Works on any entity id |

### M5: LLM matcher (stage 3, Kaggle)
| ID | Task | Exit criterion |
|---|---|---|
| T5.1 | Synthetic French pairs + distractors (noise catalogue applied to test S1 France) and French validation proxy | Generated set's noise statistics resemble `06_test_france` pseudo-pairs |
| T5.2 | Prompt v1: select format, raw + normalised fields, hint tags, GBM score; token budget measured | Median prompt ≤ ~600 tokens at N=5 |
| T5.3 | Training data: uncertain-band queries (+ a sample of easy ones + French synthetic), balanced "none" | Dataset card logged (sizes, label mix) |
| T5.4 | Kaggle LoRA fine-tune Qwen3-1.7B (fp16, T4/P100) | Val selection accuracy logged; checkpoint saved as a Kaggle dataset |
| T5.5 | Kaggle batched inference (letter probabilities), throughput measured | Tokens/s logged; full test queue fits the GPU budget |
| T5.6 | Blend LLM with GBM → decision layer | **Val F0.5 improves over M4; leaderboard submission** |
| T5.7 | *(if budget allows)* Qwen3-4B / wider band / N | Kept only if val F0.5 improves |

### M6: France and final polish
| ID | Task | Exit criterion |
|---|---|---|
| T6.1 | ~~Pseudo-label France~~ → moved to T7.4 | — |
| T6.2 | Error-analysis loop on val segments (Indic, singletons, hard negatives) | Each fix logged with its Δ F0.5 |
| T6.3 | End-to-end reproducibility run from a clean checkout | **Both** output files (matching_results.tsv **and** candidate_pairs.tsv) regenerate; validator PASS; matches ⊆ candidates |
| T6.4 | Documentation_template.md filled in, code README, submission zip | Zip matches the required structure, incl. `output/candidate_pairs.tsv`; candidate count and recall ceiling reported in the doc |

---

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Kaggle GPU budget (~30 h/week) too small for LLM inference on test | Route only the uncertain band; N≤5; prefill-only scoring; 1.7B before 4B; measure throughput early (T5.5) |
| Candidate recall ceiling on Indic names | Learned dictionary (T1.3); dense ANN fallback (T2.4) |
| France behaves differently from the proxy | Country-agnostic features; synthetic + pseudo-labels; check on the leaderboard. **Materialised:** France ≈ 0.94 on LB estimate → M7 |
| Pseudo-label confirmation bias (France) | Only very confident, clear-margin decisions; hard negatives from runner-ups; judge on proxy + LB, not on pseudo-label fit |
| Val over-optimistic vs LB (US/India) | Log val → LB delta per submission; if US/India gap appears, revisit the split |
| Validation over-optimistic | Geographic split; compare val vs leaderboard at every submission |
| Memory on a Mac with 10M+ records | Parquet + chunked processing; per-country indices |
| Uploading data to Kaggle | Private Kaggle dataset; only the derived artifacts the notebooks need |
| License violation | Check each dependency/model at the moment it's added (T0.1 and ad hoc) |
