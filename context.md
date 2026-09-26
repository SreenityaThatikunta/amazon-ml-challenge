# Handoff context: Amazon ML Challenge 2026 (Business Entity Resolution), team "pootato"

Written 2026-09-26 ~08:30 IST at the end of a long Claude Code session. Read this first, then
`progress.md` (the detailed log: every change, its reasoning, every experiment and number).
The problem statement PDF is supplied separately (its text is also extracted at
`D:\aml\problem_pdf.txt`). This file does not repeat it.

---

## 1. Situation in one paragraph

We are on the public leaderboard at **0.980** (best so far) with the file produced by
branch `improve/v2` (commit `8811d17`). The **final ranking uses the private part of the
test set** (the rest of the test S1 entities), and `candidate_pairs.tsv` plus the code are
reviewed for the final ranking (smaller candidate set per S1 ranks higher). The team's explicit
priority for the remaining time is **a well-generalising, defensible approach over
squeezing the public score**: a change that is principled and robust is preferred even if the
public score dips below 0.980. Deadline: Unstop timer ends ≈ **Sun 27 Sep 2026 23:59 IST**
(verify on the page). **5 leaderboard submissions per team per day**; 2 were used on
2026-09-26 (v2.1 → 0.977, v2.2 → 0.980).

## 2. People, repo, rules for working

- Team: Sreenitya Thatikunta (wrote `main`), Rudraksha Dev Sharma (the user in these
  sessions; email rudrakshash@gmail.com, git user `RudrakshaSh`).
- GitHub: https://github.com/SreenityaThatikunta/amazon-ml-challenge
  - `main` = Sreenitya's pipeline (val 0.98211, public LB 0.975). **Never commit to main.**
  - `improve/v2` = all our work (15 commits ahead of main). Push works from the laptop
    (Git Credential Manager). Commit messages end with
    `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  - `Rudy_commit_1` = an unrelated earlier pipeline of Rudraksha's (ignore).
- The user wants a `progress.md` that explains the **logic** behind every change (like
  main's `plan.md`/`progress.md`). Keep appending to the "v2" section at its top.
- The user is sensitive to **RAM use** (16 GB laptop, often ~4–7 GB free): never run
  heavy analyses at the same time as a pipeline stage; keep stages ≤ ~10 GB peak.
- Downloads (pip packages, models) need the user's explicit OK first (name, source, size).

## 3. Machine and folders

- Windows 11, 16 threads, 16 GB RAM, **RTX 4050 Laptop 6 GB** (CUDA 13.1 toolkit installed).
  **C: has only ~12 GB free → everything lives on D:.**
- `D:\aml\amazon-ml-challenge\` = the git clone (the session's project dir)
  - `dataset\` (gitignored) = the organisers' TSVs, extracted from
    `C:\Users\Rudraksha\OneDrive\Desktop\Amazon ML\6ab10eb3b23ba_student_resource.zip`
  - `.venv\` (gitignored) = Python 3.13 (miniconda base) with `requirements.txt` +
    `cupy-cuda13x` (GPU top-k, MIT) + editable `ber` package. CLI: `.venv\Scripts\ber.exe`.
  - `artifacts\<run>\<stage>\` (gitignored): `v2` = the full production run (all stages,
    train+test); `v2x` = experiment run whose normalize/candidates/features/scorer dirs are
    **directory junctions to v2** (only rerank/decide/submit are its own; don't delete
    v2's dirs); `dev2` = dev-slice run; `main` = partial old run (split used by everything:
    `data.split_run=main`).
  - `output\` = files for upload/zip (currently **v2.2**). `pootato_submission.zip` = final
    package built from `output\` + code + `Documentation_template.md` (currently v2.2).
- `D:\aml\submissions\{v2, v2.1, v2.2_domainnorm, probe_stage1, probe_france_empty}\` =
  every candidate file we produced (v2.2 = LB 0.980, v2.1 = 0.977; the two probes were not
  submitted).
- `D:\aml\logs\` = stage logs; `D:\aml\analysis\` = analysis scripts and their outputs
  (`shift.py/.txt`, `adversarial.py/.txt`, `errors_v2.py/.txt`, `france_inspect.txt`,
  `lowp_inspect.txt`); `D:\aml\run_*.sh` = helper runners (bash).
- `C:\Users\Rudraksha\OneDrive\Desktop\Amazon ML\` holds the user's *other* older pipeline
  (its own CLAUDE.md, `submission\`, and an old `pootato_submission.zip`): **not the one we
  use; do not upload that zip.**

## 4. How to run (details in `code/business_entity_resolution/README.md`)

Run from `D:\aml\amazon-ml-challenge` in Git Bash with `export PYTHONUTF8=1`.
Stages: `split normalize candidates features scorer rerank decide submit`, each
`ber <stage> --run <run> [--set key=value ...]`, artifacts + `manifest.json` per stage.
`--set data.phase=train` runs only the train/val side (experiments), `data.phase=test`
only the test side, reusing the train pass's models/params. Helpers:
`D:\aml\run_train_phase.sh <run> <from-stage>`, `run_test_phase.sh`, `run_rerank_refresh.sh`.
Tests: `cd code/business_entity_resolution && ../../.venv/Scripts/python.exe -m pytest -q
-p no:cacheprovider --basetemp=/d/aml/pytest_tmp` (all pass).

Timings / peak RAM on this laptop (full data):

| Stage | Train/val side | Test side | Peak RAM |
|---|---|---|---|
| normalize | 7 min (both) | – | 4 GB |
| candidates | 145 min CPU (≈60–80 min with GPU) | 78 min (GPU) | 8.2 GB |
| features | 64 min | 115 min | 9.9 GB |
| scorer | 4 min (+val scoring) | 12 min | 6.5 GB |
| rerank | 3 min | 8 min | 7.5–9.3 GB |
| decide + submit | 5 min | 2 min | ≤ 4 GB |

Cheap iteration loop: anything after `features` (scorer/rerank/decide) can be re-run in
`v2x` in minutes; changing normalisation/candidates/features costs a full re-run (~6–8 h).

## 5. What the pipeline is now (v2.2)

Normalise → region-blocked char-3gram TF-IDF top-k (3 indexes, ~24 S1 per record, 99.05 %
of true pairs; GPU for large blocks) → 102 pair features → **stage 1** LightGBM (p1),
keep p1 ≥ 0.001 (**= `candidate_pairs.tsv`, 5.57 per S1, 99.97 % of retrieved true pairs
kept**) → **stage 2** re-ranker LightGBM on p1 + S1/record group context + all 102 stage-1
features, trained cross-fitted on the val side, with **domain normalisation** of 20
size-dependent features → one owner per record → per-S1 expected-F0.5 (min_p 0.2) → TSVs +
official validator. Full description: `Documentation_template.md` (methodology, filled) and
`progress.md`.

## 6. Results so far

| Version | Val macro F0.5 (geographic split, 222,624 S1) | Public LB |
|---|---|---|
| main (6120d59) | 0.98211 | 0.975 |
| v2 stage 1 only (new features) | 0.98337 | (probe file ready, not submitted) |
| v2.1 stage 2, 25 → all features | 0.98748 → 0.98895 (region-held-out CV 0.98684) | 0.977 |
| **v2.2 + domain normalisation** | 0.98897 | **0.980** |

Val ↔ LB gap: 0.007 → 0.012 → 0.009. Val is optimistic for test; treat val as a
*relative* signal and prefer robustness checks (section 8).

## 7. Key findings (evidence in `progress.md` and `D:\aml\analysis\`)

1. **Test ≠ train in density.** Test US has **half** the S1 of train US (663k vs 1.32M);
   test has 5.5–5.8 S2/S3 records per S1 vs 4.7 in train (~40 % unmatched vs 26 %).
   Adversarial validation on stage-2 inputs: AUC 0.986 (US), 0.990 (India), 0.9997 (France).
   Top drifting features are size-dependent: per-S1 counts (`s_n`, `s_n_top1`, `s_n_cands`,
   `c_expected`), `s_name_freq` (S1 sharing the core name: p90 31 on test vs 61 on val for
   certain matches), IDF of unmatched tokens (lower by log 2 on test US).
   → Domain normalisation in stage 2 gave **+0.003 LB**. Stage 1 still sees raw values.
2. Test has ~2× the "same name, same street, house number shifted" lookalikes per S1, but
   the model already rejects them: doubling them on val costs only −0.0003.
3. Label-shift EM per segment (Saerens) is unreliable here (feature shift within segments);
   left as `decide.prior_shift` (off). Adversarial importance weights are degenerate
   (effective sample share 11 %); `rerank.weights_file` exists but unused.
4. France (15 % of test, no labels): predictions look correct on inspection; link counts
   per S1 match US/India. The stage-2 model trained on US-only still helped India (best
   proxy for France transfer).
5. Val error budget (v2): missed pairs 2.65 % of true pairs, **more than half from the 4 %
   of records with an empty address** (they lose 32.5 % of their pairs; names are shared by
   many S1, often unrecoverable); false links 0.37 % of links (lookalikes).
6. Noise catalogue learned from data (see `progress.md`): injected ids/phones in names,
   filler words per country, made-up brand names (syllables jax/tavo/zeph…), prepended fake
   house numbers in India, doubled words, legal-word typos.
7. Candidate-set size matters for the final ranking. Trade-off at the stage-1 filter
   (val): p1 ≥ 0.001 → 1.08 pairs/record, 99.97 % kept; 0.003 → 0.98, 99.93 %; 0.01 → 0.88,
   99.78 %.

## 8. Strategy for the remaining time: generalise, don't chase the public LB

The public LB is one random subset of test S1 (~sampling noise is tiny at this size, so a
0.003 change is real), but **choosing among many variants by public score overfits the
selection**. Rules we agreed on:

- A change ships only if (a) it has a mechanism we can explain from the data, (b) it does not
  hurt val, (c) it holds under **robustness checks**: region-held-out CV
  (`rerank.fold_by=region`), cross-country transfer (train on US val only → India),
  per-country test prediction stats (links/S1, empty rate, share of uncertain p), and
  adversarial AUC going *down*. The LB is a coarse sanity check, not the objective.
- Prefer **invariant / scale-free features and domain normalisation** over tuned thresholds;
  prefer ensembles/averaging over single tuned models; keep precision-first decisions
  (expected F0.5, flat around min_p 0.02–0.5, so no threshold tuning on LB).
- Keep the candidate set small (final-ranking criterion) but never trade real recall for it.
- Use at most 1–2 LB submissions per idea, and keep a written record (in `progress.md`) of
  every submission and why it was made.

## 9. Next steps, in priority order

1. **Domain-normalise stage 1 too** (most principled continuation of what worked): the same
   per-country-per-split scaling for the size-dependent stage-1 features (`s_name_freq`,
   `s_n_cands`, `s_n_top1`, IDF-based `k_idf_*`, `a_idf_*`, `q_vocab_oov`, rank features if
   justified), applied when the scorer loads features (no feature re-run needed): retrain
   scorer in `v2x`, then rerank/decide (val + region CV), then the test scoring
   (`scorer`/`rerank`/`decide`/`submit` with `data.phase=test`, ~30 min). Check adversarial
   AUC before/after (`D:\aml\analysis\adversarial.py`).
2. Consider a more principled normalisation than "divide by the mean": e.g. corpus statistics
   (IDF, vocabulary, name frequency) computed on the **union of train+test S1 per country**,
   or rank/quantile normalisation per domain. Document the choice.
3. Stage-2 robustness: average the S1-fold and region-fold stage-2 models, or drop stage-2
   features that remain strongly drifting after normalisation.
4. Optional, bigger: a transformer cross-encoder (Ditto-style, e.g. `xlm-roberta-base`, MIT,
   0.28B params) on the uncertain band, trained on the GPU; needs PyTorch + transformers
   (~3.6 GB downloads, ask first); ~4–5 h. Only worth it if the checks point to capacity rather
   than distribution shift.
5. Candidate-set size: consider `rerank.min_p1` 0.003 (0.98 pairs/record, −0.04 % recall).
6. **Before the deadline** (leave ≥ 3 h): pick the final file on robustness grounds, put it in
   `output\`, update `Documentation_template.md` + `progress.md`, rebuild the zip with
   `python code/business_entity_resolution/tools/make_submission_zip.py --team pootato`
   (checks CRC), confirm the official validator PASS on both TSVs, push `improve/v2`, and
   tell the user which file to upload and where the zip is. Ask whether to open a PR / merge
   into main (not done so far).

## 10. Gotchas

- Features of test and train are computed per split; any new corpus statistic must be learned
  identically for both (the features stage already learns filler words from both splits).
- `data.phase=test` merges the stage's previous manifest; a train pass must exist first.
- Config-hash warnings between stages are harmless.
- LightGBM gets float32 matrices (a mixed-dtype DataFrame silently becomes float64 = 2× RAM).
- The official validator is also copied to `code/business_entity_resolution/tools/` so the
  code folder is self-contained; `submit` checks `candidate_pairs.tsv` too.
- Windows: use absolute D: paths; long temp paths break git clone; `sleep` loops are blocked
  in the harness (use background jobs / Monitor).
