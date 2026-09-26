# Handoff context: Amazon ML Challenge 2026 (Business Entity Resolution), team "pootato"

Written 2026-09-26 ~14:00 IST at the end of the second long Claude Code session (the first
one's handoff is in git history, commit e83a912). Read this first, then `progress.md` (the
detailed log with the logic and every number; the newest sections are "v2.3", "v2.4" and
"Memory"). The problem statement text is at `D:\aml\problem_pdf.txt` (the PDF is
`C:\Users\Rudraksha\Downloads\6ab674645103d_emails_comms_amazon_ml_challenge_2026.pdf`).

---

## 1. Situation

- **Best file: v2.4, public LB 0.980973** (v2.2 0.980 — its exact 6-decimal value is on the
  leaderboard page, ask the user; v2.1 0.977, main 0.975). It is
  in `output\` and `D:\aml\submissions\v2.4_hold\`; the code on branch `improve/v2` (pushed)
  reproduces it with the default config. `pootato_submission.zip` (project root) was
  rebuilt from it (see section 9 for the final checklist).
- The **public LB shows 6 decimals** (not 3, as the previous handoff implied), so it can
  resolve differences of ~0.0005. It is computed on a subset of the test S1; the private LB
  (final ranking) on the rest. `candidate_pairs.tsv` and the code are also reviewed:
  smaller candidate set per S1 ranks higher.
- **Deadline:** Unstop timer ends ≈ **Sun 27 Sep 2026 23:59 IST** (verify on the page).
  **5 LB submissions per team per day**; used on 2026-09-26: v2.1, v2.2, v2.4 (3 of 5).
- Team priority (explicit): a well-generalising, defensible approach over squeezing the
  public score. The user uploads files themselves; we prepare them and say which one.

## 2. People, repo, rules for working

- Team: Sreenitya Thatikunta (wrote `main`), Rudraksha Dev Sharma (the user; email
  rudrakshash@gmail.com, git user `RudrakshaSh`).
- GitHub: https://github.com/SreenityaThatikunta/amazon-ml-challenge
  - `main` = Sreenitya's pipeline (val 0.98211, LB 0.975). **Never commit to main.**
  - `improve/v2` = all our work. Push works (Git Credential Manager). Commit messages end
    with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  - Ask the user before opening a PR / merging into main (not done so far).
- Keep `progress.md` updated with the **logic** behind every change (append to the v2
  section at its top) and record every LB submission and why.
- **RAM:** 16 GB laptop, the user keeps Chrome/ChatGPT/Claude open (often only 2–6 GB
  free). Run one heavy stage at a time; don't run analyses next to a stage. Windows' page
  file is on D: (18.6 GB), so stages swap rather than crash, but slowly.
- Downloads (pip packages, models) need the user's explicit OK first (name, source, size).
- The user likes short status updates while long jobs run.

## 3. Machine and folders

- Windows 11, 16 threads, 16 GB RAM, RTX 4050 Laptop 6 GB (CUDA 13.1). C: has ~10 GB free →
  everything on D:. GPU is used only by the candidates stage (CuPy top-k); the pip LightGBM
  has no GPU build (checked).
- `D:\aml\amazon-ml-challenge\` = the git clone (`dataset\` gitignored, `.venv\` Python 3.13
  with `requirements.txt` + `cupy-cuda13x`, CLI `.venv\Scripts\ber.exe`).
- `artifacts\<run>\<stage>\` (gitignored). Runs that matter:
  - `v2` = full production upstream (normalize, candidates, features: train + test) and
    the v2.2 stage 1. **Don't delete** (other runs link to it).
  - **`v24` = the current final run**: `normalize/candidates/features` are junctions to v2;
    `features_hold, scorer, rerank, decide, submit` are its own. `v24\rerank` holds the
    final stage-2 fold models (experiment E1) and `v24\decide\decision_params.json` the
    final decision parameters incl. the anchor bias factors. `v24\exp\E*` = the saved
    outputs/models of each stage-2 experiment. **For new experiments use a new run dir with
    junctions** (as `v24m` does), not `v24`, or the final models get overwritten.
  - `v2x` = v2.2's stage 2 + decide; `v23` = v2.3 (v2.2 + anchor); `v2m` = memory-refactor
    equivalence check; `v24m` = scratch (min_p 0.3 check); `dev2`, `main` = old.
- `D:\aml\submissions\{v2, v2.1, v2.2_domainnorm, v2.3_anchor, v2.4_hold, probe_*}\` = every
  candidate file. `D:\aml\logs\` stage logs; `D:\aml\analysis\` analysis scripts + outputs
  (see section 7); `D:\aml\run_v24.sh` (stage-1 retrain + five stage-2 experiments) and
  `D:\aml\run_v24_test.sh` (final decide + test pass + submit) are the templates for new runs.
- `C:\Users\Rudraksha\OneDrive\Desktop\Amazon ML\` = the user's *other*, older pipeline:
  not the one we use; don't upload its zip.

## 4. How to run (details: `code/business_entity_resolution/README.md`)

Git Bash, from `D:\aml\amazon-ml-challenge`, `export PYTHONUTF8=1`. Stages:
`split normalize candidates features features-hold scorer rerank decide submit`, each
`ber <stage> --run <run> [--set key=value ...]`; `--set data.phase=train|test` runs one side
(a test pass reuses the train pass's models/params; needs the train pass first).
Tests: `cd code/business_entity_resolution && ../../.venv/Scripts/python.exe -m pytest -q
-p no:cacheprovider --basetemp=/d/aml/pytest_tmp` (141 pass).

Timings / peak RAM (laptop, full data):

| Stage | Time | Peak RAM |
|---|---|---|
| normalize | 7 min | 4 GB |
| candidates | ~2.5 h train (GPU helps) + 78 min test | 8 GB |
| features | 64 min train + 115 min test | 10 GB |
| features-hold (hold_frac 0.2) | 61 min (16 min of it = per-S1 candidate context) | 7.6 GB |
| scorer | 8 min train (+ val/hold scoring) + 12 min test | 4.8 GB |
| rerank (val + hold) | 14 min train + 10 min test | 4.9 GB |
| decide | 2–5 min per pass (full min_p sweep ~4 min) | 4.6 GB |
| submit | 1 min | 4.1 GB |

Cheap loop: anything after features-hold (scorer/rerank/decide) re-runs in minutes in a
junction run; changing normalisation/candidates/features costs a full re-run (~6–8 h).

## 5. What the pipeline is now (v2.4)

Normalise → region-blocked char-3gram TF-IDF top-k (3 indexes, ~24.6 S1 per record, 99.05 %
of true pairs) → 102 pair features → **stage 1** LightGBM (p1), trained on train regions
*outside* the held-out ones; keep p1 ≥ 0.001 (**= `candidate_pairs.tsv`, 5.63 per S1,
99.97 % of retrieved true pairs**) → **stage 2** re-ranker LightGBM on p1 + S1/record group
context + all 102 stage-1 features, with domain normalisation of 20 size-dependent
features, **trained cross-fitted (4 folds by the record's top S1) on the val regions (10 %)
+ 20 % held-out train regions** (India AP GJ HR KL OD RJ, US AR CT IA ID IN MD NY;
`rerank.hold_frac: 0.2`, `train_sides: [val, hold]`) → one owner per record → **anchor prior
correction** (`decide.prior_shift: anchor`; fires only for France's generic-word swaps) →
per-S1 expected-F0.5 (min_p chosen on val: 0.02, flat) → TSVs + official validator.
Methodology write-up: `Documentation_template.md` (up to date for v2.4).

## 6. Results

| Version | Val F0.5 (222,624 S1) | Unseen-region CV | Public LB |
|---|---|---|---|
| main (6120d59) | 0.98211 | – | 0.975 |
| v2.1 stage 2 on all features | 0.98895 | 0.98684 | 0.977 |
| v2.2 + domain normalisation | 0.98897 | – | 0.980 |
| v2.3 + anchor correction (France) | 0.98897 (no-op on val) | – | not submitted |
| **v2.4 stage 2 on val + held-out regions (+ v2.3)** | **0.98911** | **0.98776** (val-only: 0.98660) | **0.980973** |

## 7. Key findings (evidence in `progress.md`; scripts in `D:\aml\analysis\`)

1. **Test ≠ train in density** (v2 session): test US has half the S1 of train US; test has
   5.5–5.8 S2/S3 per S1 vs 4.7 → domain normalisation of size-dependent stage-2 features
   (+0.003 LB).
2. **France's generic-word swaps** (`swap_check.py`, `fr_swaps.py`, `fr_hn_mix.py`): same
   address + house number, the generic word of the name swapped ("Ski College SARL" → "Ski
   Comite SARL", "Dessin Club" → "Dessin & Fils"): 6.6 % of French same-street pairs vs
   0.1–1.5 % in US/India (mostly decoys there). The model is uninformed (mean p2 0.26–0.40).
   House-number profile (independent of the name change given the match status) says
   ~80 % are true copies; not clustered at shared addresses (not "sibling" decoys).
3. **Anchor correction** (`decide/anchor.py`; `anchor_proto.py`, `anchor_sim.py`,
   `anchor_sim2.py`, `anchor_legal.py`): per country × name-change kind, share of true
   copies from the hn_rel mixture vs the model's mean p, de-biased on val; corrected only
   beyond 2×. Validated: no false alarms under 8 label-shift scenarios; recovers 96–99 % of
   the loss of an uninformed wrong-prior score. Limit: along the legal-conflict axis the
   assumption fails (India val bias 13.8×), so it is used for name-change kinds only.
4. **Model self-estimate** (`selfest.py`): expected F0.5 under its own p2 — test 0.9913
   (France 0.9870 lowest) vs val 0.9949; minus val optimism → ≈ 0.985 predicted for test,
   so part of the val→LB gap is invisible to the model.
5. **Incomplete S1 groups** (progress "v2.4"): stage 2 only loads val (+hold) records, so
   16 % of v2.2's val pairs (181k negatives on train-side S1 whose owners were never
   loaded) looked trivially easy; at test all groups are complete, esp. for records without
   a region (21 % of test pairs, mostly France's 35 % region-less records, which search the
   whole country). A val-only stage 2 mis-scores such pairs once their groups are complete
   (E0). Training on val + hold fixes much of it; 141k val pairs still sit on S1 outside
   the loaded regions.
6. **More regions for stage 2 help unseen regions** (E0–E4 table in progress): +0.0012
   unseen-region (India +0.0020, singletons +0.0055); hold-only training transfers to val as
   well as val's own regions (0.98666 vs 0.98660).
7. Earlier (v2 session): noise catalogue (fillers, brand syllables, injected ids, prepended
   house numbers), error budget (missed pairs mostly records with empty address), label-shift
   EM unreliable, adversarial weights degenerate, candidate filter trade-off (p1 ≥ 0.001:
   1.078 pairs/record, 99.97 %; 0.003: 0.984, 99.93 %; 0.01: 0.885, 99.77 %).

## 8. Next steps (in priority order; the user wants a better score)

1. **More held-out regions for stage 2** (strongest evidence so far): try `rerank.hold_frac`
   0.3–0.4 in a **new run dir** (junctions: normalize/candidates/features → v2).
   `features-hold` recomputes all hold queries (~1 h per 20 %; the per-S1 candidate context
   step alone is 16 min — could be cached/sped up with Arrow). Then scorer + the E2/E3
   protocol (region folds, val-only vs val+hold) + E1; compare unseen-region CV with
   0.98776. Note: `hold_regions()` picks regions greedily from a seeded permutation, so a
   larger frac mostly (not strictly) extends the 0.2 set. Stage 1 loses those regions too
   (at 0.2: val AUC 0.999938 vs 0.99994 — negligible).
2. **Complete the remaining S1 groups ("fix B")**: 141k val pairs still sit on S1 outside
   the loaded regions (empty groups). Option: exclude such pairs from stage-2 *training*
   (flag via the split's `s1.parquet` geo + hold regions), keep them for scoring; compare
   E1 vs E1b on val/unseen-region CV. Cheap (rerank + decide ~20 min).
3. **D1 diagnostic** (`D:\aml\analysis\d1_v22_complete_groups.py`, written, not run):
   v2.2's stage-2 models on val with fuller groups → quantifies how much v2.2 suffered
   from the empty-group artefact (documentation value; ~10 min, ~3 GB).
4. Candidate-set size: `rerank.min_p1` 0.003 (0.984 vs 1.078 pairs/record, −0.04 % of
   retrieved true pairs) for the final-ranking criterion; check val cost first.
5. Stage-1 domain normalisation (size-dependent features are ~1.3 % of stage-1 gain: small).
6. Bigger, only if time: France pseudo-labelling; a transformer cross-encoder on the
   uncertain band (downloads ~3.6 GB need the user's OK).

## 9. Before the deadline (leave ≥ 3 h)

Pick the final file on validation/robustness evidence (LB only as a regression check), put
it in `output\`, make sure `configs/default.yaml` reproduces it, update
`Documentation_template.md` + `progress.md`, rebuild the zip with
`python code/business_entity_resolution/tools/make_submission_zip.py --team pootato`
(checks CRC), run the official validator on both TSVs (`utils/validate_submission.py
--matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir
dataset/test --check-ids`), push `improve/v2`, tell the user which file to upload and where
the zip is, and ask about a PR / merge into main.

## 10. Gotchas

- New corpus statistics must be learned identically for train and test.
- `data.phase=test` merges the stage's previous manifest; a train pass must exist first.
  `ber decide --set data.phase=test` reads `decision_params.json` from the train pass: `--set
  decide.min_p` on a test-only pass has no effect.
- `run_v24.sh` overwrites `v24\rerank` per experiment (the last one, E1, is the final); save
  models per experiment (`exp\`) or use another run.
- `hold_frac > 0` needs `ber features-hold` before `ber scorer` (the scorer excludes the hold
  queries from training and scores them; `dev.yaml` / `france_proxy.yaml` set it to 0).
- Filter by id with Arrow (`pc.is_in` on a prebuilt array), not `Series.isin(python_set)`:
  the latter took 19 s per 1 M-row part (was the cause of a 10-min stall).
- Config-hash warnings between stages are harmless. LightGBM gets float32 matrices.
- The official validator is also in `code/business_entity_resolution/tools/`; `submit`
  checks `candidate_pairs.tsv` too.
- Windows: absolute D: paths; Git Bash heredocs with nested quotes can break — write scripts
  to files; `sleep` loops are blocked in the harness (use background jobs / `until` loops).
