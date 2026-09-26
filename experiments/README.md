# Experiments log

Every command run against the pipeline, what it was for, where its log is and what it
showed, newest last. The pipeline itself is documented in
[code/business_entity_resolution/README.md](../code/business_entity_resolution/README.md);
task status and decisions are in [progress.md](../progress.md) and [plan.md](../plan.md).

All commands run from `student_resource/`, with `B=.venv/bin/ber`.
Heavy stages run **one at a time** (16 GB RAM).

## Files here

| File | What |
|---|---|
| `ab_dev_france.sh` | Dev A/B for the France work: new features (B1), + synthetic French training (B2); each scored on dev val and on the France proxy |
| `top1_by_country.py` | Best candidate per test query with its house-number relation and p, by country (writes `logs/top1.parquet`) |
| `logs/` | Stage and script logs (text tracked; parquet ignored) |

Related scripts elsewhere: `eda/07_france_errors.py` (France error dump + proxy realism;
report in `eda/reports/07_france_errors.txt`, proxy-only report `eda/reports/07_france_proxy.txt`).

---

## 2026-09-25

### E1. First LB submission
- `$B submit --run main`: `output/matching_results.tsv` (val 0.98211), validator PASS.
- **Public LB 0.975** (top 0.986).

### E2. candidate_pairs.tsv (stopped)
- `$B submit --run main` with `submit.write_candidates=true` (now the default). Stopped by the user
  after ~12 min to free the machine for E3. `output/matching_results.tsv` untouched.
- To do before the final zip: rerun `$B submit --run main` (15–25 min, ~3 GB file).

### E3. France error dump (T7.1)
- `.venv/bin/python eda/07_france_errors.py` (10 min, peak 5.7 GB). Log `logs/07_france_errors.log`.
- `.venv/bin/python experiments/top1_by_country.py` (number relation of the best candidate by country).
- Findings (details in progress.md, T7.1):
  - France has ~0.13 extra confident queries per S1 vs US/India, so ~4% of France predictions are
    probably false.
  - France S1 is full of S1-level twins: 13.3% of S1 share their exact address with another S1 (US 4.6%),
    1.2% share name + street at another number (US ~0).
  - Ambiguous France cases have fixable causes (different street at the same number, legal form, short
    "truncated" numbers, initials); ambiguous US cases are 84% empty addresses.
  - Real France shifted-number lookalikes are rejected fine (87.6% p < 0.05).
  - The old proxy was not realistic.

### E4. Realistic France proxy (T7.1b)
- Code: `france/synth.py` (all S1 indexed, sample scored, joint cluster sizes, real background records),
  `decide.eval_s1`, configs `france_proxy.yaml`.
- Commands (log `logs/proxy_synthfr_v2.log`, ~6 min):
  ```bash
  $B synth-france --run main
  for st in split normalize candidates features scorer decide; do
    $B $st -c code/business_entity_resolution/configs/france_proxy.yaml; done
  .venv/bin/python eda/07_france_errors.py --proxy-only
  ```
- Result: realism table within ~1.3× of real France on every statistic; proxy F0.5 with model
  `main` **0.87597** (pessimistic vs the LB-implied ~0.94, so it's used as a relative metric).

### E5. Synthetic French training set (run `frtrain`)
- Commands (log `logs/frtrain_build.log`, ~7 min):
  ```bash
  $B synth-france --run frtrain -s seed=7 -s france.n_entities=60000 \
      -s france.exclude=artifacts/main/synth-france/data/eval_s1.tsv
  for st in split normalize candidates features; do
    $B $st -c code/business_entity_resolution/configs/france_train.yaml; done
  ```
- 60k S1 (disjoint from the proxy's 20k scored S1), 207k true pairs, 9.07M featurised pairs.

### E6. Dev A/B: new features and synthetic French training
- `bash experiments/ab_dev_france.sh` (log `logs/ab_dev_france.log`, ~15 min).
  B2 uses `scorer.extra_train=[frtrain]`, `extra_train_frac=0.39` (same synthetic : real ratio as the
  full run).

| Variant | Dev val F0.5 | US | India | France proxy | Proxy P / R | Proxy singletons |
|---|---|---|---|---|---|---|
| B0 old features (dev model) | 0.97655 | 0.9893 | 0.9637 | 0.88666 | | |
| B1 + lookalike features | **0.97823** | 0.9898 | 0.9666 | 0.87962 | 0.883 / 0.939 | 0.663 |
| B2 = B1 + synthetic French training | 0.97683 | 0.9891 | 0.9645 | **0.97122** | 0.981 / 0.964 | 0.937 |

- The lookalike features help US/India (+0.0017). Synthetic French training lifts the proxy by +0.09,
  which is overstated: proxy and training set come from the same generator. It costs dev −0.0014.
- **Decision:** score seen countries with the B1-style model and unseen countries (France) with the
  B2-style model (country-agnostic features; the country only picks which model to apply).

### E7. Feature speed
- The new features made `features` 2.5× slower (dev 116 s → 285 s). Profile: fuzzy address-word
  matching in a Python generator. Fix: rapidfuzz `process.extractOne` + per-record precomputed
  `addr_words` / `house` / `name_city`. Lookalike features 14.0 s → 4.5 s per 300k pairs.

### E8. Full-scale features v2 (running)
- `$B features --run main` (log `logs/main_features_v2.log`, started 18:37). Overwrites
  `artifacts/main/features`.

### E9. Unseen-country model (scorer routing)
- Code: `scorer.unseen_extra_train` / `unseen_base_frac`, `model_info.json`. Pairs whose S1 country is
  absent from training are scored by a 2nd model trained with the extra runs (synthetic French);
  the rest by the main model.
- Smoke test (log `logs/routing_smoke_dev.log`, small 30% dev model): 100% of proxy pairs routed to the
  unseen model; proxy 0.97059, dev 0.97684. The plumbing works.

### E10. M8 cluster-context rescore (`ber rescore`, dev)
- `$B scorer -c dev.yaml -s scorer.oof_folds=1; $B decide -c dev.yaml` rebuilt the B1 baseline
  (0.97823, log `logs/dev_B1_baseline.log`).
- `$B rescore -c dev.yaml; $B decide -c dev.yaml -s decide.scores_stage=rescore`
  (log `logs/dev_rescore_v1.log`, 16 s): **dev val 0.97823 → 0.98593** (US 0.9915, India 0.9803;
  pair recall ~0.95 → 0.977 at precision 0.995; val log-loss 0.0270 → 0.0134). Val is cross-fitted
  (2 folds by the query's best S1), so these scores are out-of-sample.
- On the France proxy (`rescore.model_from_run=dev`, log `logs/proxy_rescore_dev.log`) the US/India
  rescore *lowers* F0.5 0.97059 → 0.96617, so test rescoring is restricted to seen countries.
- Sweep `bash experiments/sweep_rescore_dev.sh` (log `logs/sweep_rescore_dev.log`):

| k | conf | dev val F0.5 | US | India |
|---|---|---|---|---|
| 3 | 0.3 | **0.98632** | 0.9913 | 0.9813 |
| 3 | 0.5 | 0.98593 | 0.9915 | 0.9803 |
| 3 | 0.7 | 0.98590 | 0.9910 | 0.9808 |
| 5 | 0.3 | 0.98628 | 0.9916 | 0.9810 |
| 5 | 0.5 | 0.98592 | 0.9912 | 0.9806 |
| 5 | 0.7 | 0.98593 | 0.9916 | 0.9803 |

  Differences (±0.0004) are within noise on 26k dev S1. Keep k=3, conf=0.5 for v2, and re-check conf at
  full scale.

### E12. Parallel features + address-word matcher fix
- `features.workers` (default min(4, cpu−2)): `build_features` runs on query-aligned sub-chunks in
  worker processes. Dev features 285 s → 114 s (while the full run competed for CPU). Test
  `test_parallel_features_match_serial`.
- Bug found by diffing dev features against the pre-E7 build (1% of `a_q_unmatched` differed):
  rapidfuzz `extractOne(score_cutoff=0.8)` with a *similarity* scorer converts the cutoff to an integer
  edit budget and drops pairs at exactly 0.8 ("naga"/"nagar"). Now uses `normalized_distance` with
  cutoff `1 − 0.8`: 0 mismatches against the reference definition on 20k random address pairs.
  Test `test_unmatched_word_threshold_inclusive`.
- The E8 full run (serial, boundary bug) was stopped at 10/81 train parts (log
  `logs/main_features_v2_aborted.log`); E11 restarted at 19:00 with features as its first step.

### E11. Full-scale v2 chain (running since 19:00)
- `bash experiments/run_main_v2.sh` (log `logs/run_main_v2.log`): features (parallel) →
  scorer (+ unseen model, frtrain) → decide (stage-1, kept as `logs/main_v2_decide_stage1.json`)
  → rescore → decide (rescored) → submit (+ candidate_pairs.tsv) → France proxy with the new models.
- The v1 submission is archived as `output/archive/matching_results_v1_val098211_lb0975.tsv`.

### E13. Rescore rounds (dev)
- `rescore.rounds` (round r+1 rebuilds clusters/owners from round-r out-of-fold p). Logs
  `logs/dev_rescore_rounds{1,2,3}.log`. The dev baseline was rebuilt first with the E12 features (log
  `logs/dev_v2_baseline.log`): stage-1 0.97823 and rescore 0.98593, unchanged by the matcher fix.

| rounds | dev val F0.5 | val AUC per round |
|---|---|---|
| **1** | **0.98593** | 0.999832 |
| 2 | 0.98560 | 0.999832, 0.999799 |
| 3 | 0.98433 | 0.999832, 0.999799, 0.999744 |

- Chaining out-of-fold probabilities compounds noise, so **keep rounds=1**.

### E14. Where the dev loss is after rescore
- `.venv/bin/python experiments/loss_breakdown.py dev` (log `logs/loss_breakdown_dev_rescore.log`):
  loss **India 70%** / US 30%; fn-only 61%, fp-only 19%, singletons given a match 13%. Missed true pairs
  2.3%: dropped at low p 1381 (median p 0.45), not a candidate 318, assigned to another S1 234, not in
  the rescored top-3 140.
- India dev misses (log `logs/india_misses_dev.log`): mostly **Gurmukhi** names that obviously match
  ("ਸ਼ਕਤੀ ਐਨਰਜੀ ਪ੍ਰਾਈਵੇਟ ਲਿਮਟਿਡ" = Shakti Energy Private Limited, same address, p 0.02).

### E15. Indic scripts and the transliteration dictionary
- Log `logs/indic_scripts_test.log`. The train-side dictionary has **0 Gurmukhi words**: all 10.8k
  Gurmukhi train names are on the val side, because Punjab is a val region. Gurmukhi names with unknown
  words: 32–33% (other scripts ≤ 6%). Test has 12.5k Gurmukhi names (0.27% of Indian queries), so this
  is worth at most ~+0.0007.
- Fix (code, used from the next normalize run): `normalize.test_dict_all_pairs` learns a 2nd dictionary
  from **all** train pairs (`translit_dict_all.parquet`) and uses it for test records only; train/val
  records keep the train-side one, so val stays honest.
- Val India (PB, UP, CH, MN, UK) includes Punjab's Gurmukhi, so val India likely *underestimates*
  test India.

### E16. Candidate recall misses (full scale)
- `.venv/bin/python experiments/candidate_misses.py main` (log `logs/candidate_misses_main.log`, ~12 min):
  0.95% of val true pairs are never candidates (India 1.35%, US 0.69%). **68% of misses have an empty
  query address** (base rate 4.4%): with no address only the name index is searched (k=30,
  country-wide), and noisy generic names ("Summit Éuiy LLC") fall outside the top 30. The rest are mostly
  Indian names in Devanagari script with partial addresses.
- Dev (log `logs/empty_addr_recall_dev.log`): empty-address true pairs have recall 0.835 (others 0.984);
  0.882 when the owner is a candidate; decision precision 0.976 (others 0.996).
- Plan: `candidates.k_name_noaddr` 30 → 100 (+~31M test pairs, +13%), expected ~+0.0005.

### E17. Rescore with more stage-1 features (dev)
- `rescore.stage1_feats` (list; `["*"]` = every stage-1 feature column). Logs
  `logs/dev_rescore_extfeats.log`, `logs/dev_rescore_allfeats.log`,
  `logs/dev_rescore_allfeats_default.log`.

| stage-1 features in rescore | dev val F0.5 | US | India | val AUC |
|---|---|---|---|---|
| short list (6) | 0.98593 | 0.9915 | 0.9803 | 0.999832 |
| 30 selected | 0.98814 | 0.9917 | 0.9846 | 0.999856 |
| **all (91)** | **0.98859** | 0.9920 | 0.9852 | 0.999868 |

- **Default is now `["*"]`**; v2 uses it. At full scale the test path would need ~12 GB for
  30M top-3 pairs × 100 features, so cluster/member features are now built globally and the stage-1
  features are fetched and scored part by part (`predict_parts`, test
  `test_predict_parts_matches_direct`).
- Total dev gain of the v2 stack over v1: **0.97655 → 0.98859 (+0.0120)**.

### E18. v2 on the LB: 0.972 (< v1 0.975); density shift
- `main` v2 (val 0.98906) scored **LB 0.972**, below v1's 0.975.
- `experiments/logs/val_vs_test_density.log`: test has ~1 extra record per S1 with p1 < .05 (1.9–2.1 vs
  1.15 on val) and ~60% more records per S1 overall, so per-S1 count features shift (`s_n_cands` 364 → 576,
  `s_n_top1` 5.7 → 9.1) in a way val can't show.
- Code (uncommitted WIP, committed 2026-09-26): `scorer.drop_features=[s_n_cands, s_n_top1]`; rescore
  skips them (`DENSITY_FEATS`) and counts only plausible cluster members (`s_n_plaus`, p ≥ 0.05) instead of
  every assigned record; features are written as float32/int32 with zstd (29 GB → ~13 GB,
  `experiments/compact_features.py` rewrites existing parts); `submit` maps dictionary-encoded ids once
  (`_global_codes`, test `test_global_codes_dictionary_and_plain`).
- Not yet validated by a run. `compare_submissions.py` v1 → v2 also shows France losing pairs
  (+8.4k / −21.5k), so the synthetic-France unseen model may account for part of the drop (plan.md §0).

### Next: v3 upstream rebuild (after the v2 LB check)
- normalize (all-pairs dictionary for test) → candidates (k_name_noaddr=100) → features → scorer
  (+ unseen model) → rescore → decide → submit. About 4 h, heavy stages one at a time.

