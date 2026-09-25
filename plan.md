# Plan: Business Entity Resolution (Amazon ML Challenge 2026)

**Goal:** beat the public leaderboard top of **0.971 macro F0.5** (as of 2026-09-25).
**Inputs:** [eda/FINDINGS.md](eda/FINDINGS.md), [docs/RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md).
**Tracking:** every task, run, metric and decision is logged in [progress.md](progress.md).

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
  io.py                    TSV readers, parquet artifacts, manifests, submission writers
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

## 4. Tasks

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
| T6.1 | Pseudo-label high-confidence France test matches → fine-tune / recalibrate | Gain on the French proxy; LB check |
| T6.2 | Error-analysis loop on val segments (Indic, singletons, hard negatives) | Each fix logged with its Δ F0.5 |
| T6.3 | End-to-end reproducibility run from a clean checkout | Both output files regenerate; validator PASS |
| T6.4 | Documentation_template.md filled in, code README, submission zip | Zip matches the required structure |

---

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Kaggle GPU budget (~30 h/week) too small for LLM inference on test | Route only the uncertain band; N≤5; prefill-only scoring; 1.7B before 4B; measure throughput early (T5.5) |
| Candidate recall ceiling on Indic names | Learned dictionary (T1.3); dense ANN fallback (T2.4) |
| France behaves differently from the proxy | Country-agnostic features; synthetic + pseudo-labels; check on the leaderboard |
| Validation over-optimistic | Geographic split; compare val vs leaderboard at every submission |
| Memory on a Mac with 10M+ records | Parquet + chunked processing; per-country indices |
| Uploading data to Kaggle | Private Kaggle dataset; only the derived artifacts the notebooks need |
| License violation | Check each dependency/model at the moment it's added (T0.1 and ad hoc) |
