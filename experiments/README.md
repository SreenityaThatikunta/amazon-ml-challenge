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

### E11. Full-scale v2 chain (queued)
- `FEATURES_PID=<pid> bash experiments/run_main_v2.sh` (log `logs/run_main_v2.log`). Waits for E8,
  then: scorer (+ unseen model, frtrain) → decide (stage-1, kept as `logs/main_v2_decide_stage1.json`)
  → rescore → decide (rescored) → submit (+ candidate_pairs.tsv) → France proxy with the new models.
- The v1 submission is archived as `output/archive/matching_results_v1_val098211_lb0975.tsv`.
