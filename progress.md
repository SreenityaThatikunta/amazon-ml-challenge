# Progress

Single source of truth for task status, runs, metrics and decisions.
Plan and task definitions: [plan.md](plan.md).

**Status keys:** ⬜ todo · 🔄 in progress · ✅ done · ⏸ blocked · ❌ dropped

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
| T6.1 | France pseudo-labelling | ❌ | moved to T7.4 |
| T6.2 | Error-analysis loop | ⬜ | |
| T6.3 | Clean-checkout reproducibility | ⬜ | |
| T6.4 | Documentation + submission zip | ⬜ | |
| T7.1 | France error dump + proxy realism check | ✅ | catalogue in `eda/reports/07_france_errors.txt`; old proxy unrealistic → T7.1b |
| T7.1b | Realistic France proxy | ✅ | all 259k S1 indexed, 20k scored; joint cluster sizes; 37.8k real background records. Realism table within ~1.3× of real France on every stat; proxy F0.5 (main model) **0.87597**: used as a *relative* metric |
| T7.2 | Strip own city token from names; city-only-overlap feature | ✅ | in `lookalike_features` |
| T7.3 | French legal-form conflict, street-mismatch-at-same-number, initials mismatch | ✅ | + relative margins; dev +0.0017 |
| T7.4 | France pseudo-labels → retrain scorer | 🔄 | replaced for now by synthetic French training + unseen-country model (proxy 0.880 → 0.971); pseudo-labels only if the LB shows a France gap |
| T7.5 | France calibration check (**LB**) | ⬜ | |
| T8.1 | Stage-1 confident clusters | ✅ | `ber rescore` (`cluster/stage.py`) |
| T8.2 | Second-order cluster-context features | ✅ | cluster size/sum/cap/rank + member similarity; tests `test_cluster.py` |
| T8.3 | Stage-2 LightGBM → decide (**LB**) | 🔄 | dev **0.97823 → 0.98593**; full-scale run queued (E11) |

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
| 2026-09-25 | main | 0.98211 | **0.975** | — | LB top now 0.986. Test S1 = US 38% / India 47% / France 15%; if US/India transfer from val, France ≈ 0.94 |

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
| 2026-09-25 | ~~Don't produce candidate_pairs.tsv (`submit.write_candidates=false`)~~ **reversed below** | User: not needed for our submission; it is ~3 GB / 2.4e8 ids and the official validator needs ~30 GB RAM to check it |
| 2026-09-25 | **Produce candidate_pairs.tsv** (`submit.write_candidates=true` by default) | User: required in the final submission zip. `ber submit` checks matches ⊆ candidates itself (memory-light) |

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

### 2026-09-25 (re-plan)
- First LB: **0.975** (val 0.98211); LB top moved to 0.986. Gap analysis and new milestones M7 (France) and M8
  (cluster-context scorer) added to plan.md §4. Priority: M7 → M8 → M5 (LLM) → T2.3/T2.4 → M6.
- candidate_pairs.tsv was missing from `output/` because it was switched off, not deleted. Re-enabled by default;
  `ber submit` now also checks every match is one of its S1's candidates (new test in tests/test_io.py).
- **T7.1 France error dump** (`eda/07_france_errors.py` → `eda/reports/07_france_errors.txt`, 10 min, peak 5.7 GB;
  US = 20% query sample). No labels for real France, so this finds *suspects*, not errors.
  - **Where the France loss probably is: confident false matches.** Confident queries (p1 > 0.5) per S1:
    France 3.54 vs US 3.42 / India 3.40, with the same generator. That excess of ~0.13/S1 ≈ 4% of France
    predictions, which is enough to explain France ≈ 0.94.
  - **France S1 is full of S1-level lookalikes that training never showed:** S1 sharing its exact address with
    another S1: France **13.3%** vs US 4.6% / India 5.2%; same name + same street, different number: France
    **1.2%** vs ~0 in US/India ("Lille Amicale SARL" at 13 and at 3 Rue du Vaisseau Le Vengeur). Names are
    "<City> <generic word> <legal form>", so "city in S1 name" is 48% of ambiguous runner-ups (US 1%).
  - Ambiguous queries (p1 > 0.5, p2 > 0.2): France 0.79% vs US 0.24%. Runner-up tags, France vs US:
    S1 twin (same core name as the winner) 40% / 76%; empty query address 17% / 84%; different street, same
    number 23% (US 2%); query gibberish 24%; initials mismatch 8%; legal conflict 5%.
    US ambiguity is almost only empty addresses; France ambiguity has **fixable** causes.
  - Concrete bad runner-ups: legal conflict at same address (SAS vs SARL p 0.87, SARL vs EURL 0.63);
    street type differs (Impasse vs Rue de l'Hermitage 0.89); **truncated number while another candidate
    matches exactly** (13 vs 3: 0.987; 103 vs 10: 0.80); initials of the other candidate (AC → "Mérignac
    Societe" 0.88). These call for **relative** features: "another candidate has an equal number / same street
    / same legal form / matching initials".
  - House-number relation of the best candidate, share with p < 0.05: shifted France 87% / US 92% (fine);
    truncated France **77%** vs US 11%: French numbers are short, so a different number on the same street
    often looks "truncated" (13↔3, 103↔10, 42↔4). The TRUNCATED rule (also leading-digit drops) is
    unreliable for 1–3 digit numbers.
  - **Proxy verdict: not realistic, so its 0.888 is not a France estimate.** vs real France: p1 < 0.05 2.5% vs
    30% (no unrelated unmatched records); empty S1 1.0% vs 4.9% (bug: S2/S3 cluster sizes drawn
    independently → P(both 0) too small); S1 name-frequency p90 4 vs 38 (20k of 259k S1 indexed → S1 twins
    ~13× too sparse); 99.6% of proxy FPs are synthetic distractors, mostly same name + shifted number
    (33%), which the real data rejects fine. The proxy tests the wrong thing: it lacks the S1-twin confusions
    that drive real France errors.

- **T7.1b realistic proxy** (`ber synth-france` + `configs/france_proxy.yaml`, `decide.eval_s1`): every test
  S1 France is indexed (twins confuse the proxy as they do the real test) but only a 20k sample gets copies
  and is scored; (S2, S3) cluster sizes drawn jointly (scored singletons 5.5%, was 1.55%); 37.8k real test
  France records with best main p < 0.01 added as unrelated unmatched records; synthetic ids are 10-digit.
  Realism (real France vs proxy): p1<.05 0.30/0.33, ambiguous 0.79%/0.77%, pred/S1 3.48/3.68,
  empty S1 4.9%/4.1%, top1 shifted 23%/24%, S1 name-frequency p90 38/37; shifted-number top1 rejected
  (p<.05) 87.6%/86.3% with the same shift sizes. Proxy F0.5 with model `main`: **0.87597** (pair P 0.887,
  R 0.937; singletons 0.640). This is lower than the LB implies for France (~0.94), so the absolute level is
  pessimistic: use it to compare models. Remaining FPs: 99% synthetic distractors (legal-form swap at the
  same address, shifted number with p ≈ 0.9).
- New code (tests pass, 126): `features/build.py` lookalike features (name without the S1's city words,
  address words each side lacks with fuzzy matching, house-number digit length and leading-digit truncation,
  **relative margins** of number / legal / address-word / initials / exact-name agreement over the query's
  other candidates); `scorer.extra_train` (labelled pairs of other runs added to training only);
  `france.exclude`; `configs/france_train.yaml` (synthetic French training set, disjoint from the proxy).
- **Experiments E6–E11** (full record in `experiments/README.md`):
  - lookalike features: dev +0.0017;
  - synthetic French training: proxy 0.880 → 0.971 but dev −0.0014, hence the **unseen-country model**
    (a 2nd scorer used only for S1 countries absent from training);
  - **M8 rescore: dev 0.97823 → 0.98593**, cross-fitted; it hurts the France proxy, so it's applied to
    seen countries only.
  - Full-scale v2 queued behind the features rerun (started 18:37, after the E7 speed fix).

---

## Open questions
- Local RAM is 16 GB: full-train TF-IDF top-k (T2.1) must stream per country/chunk; watch peak memory.
- Kaggle specifics to confirm at T5.4: GPU type available (P100 vs 2×T4), session RAM, dataset size limits.
