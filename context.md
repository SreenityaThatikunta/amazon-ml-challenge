# Handoff context: Amazon ML Challenge 2026 (Business Entity Resolution), team "pootato"

Updated 2026-09-27 ~16:30 IST in the fourth Claude Code session (earlier handoffs: git
history, commits e83a912, 2d9cbc0, b205e14). Read this first, then `progress.md` (the
detailed log with the logic and every number; newest sections at the end: "Exploration
branch `explore/cross-encoder`" incl. "Final packaging", "v2.6 → next", then "v2.5").
Problem statement: `D:\aml\problem_pdf.txt` (the organisers' brief with the
candidate_pairs update: the user's copy `Downloads\6ab674645103d_emails_comms_amazon_ml_challenge_2026.pdf`).

**The working tree is on branch `explore/cross-encoder`** (this file, progress.md and the
cross-encoder code live there). `improve/v2` = v2.6, untouched since the user asked that new
architectures go on separate branches.

---

## 0. Where we left off (do this first)

**Update 2026-09-28 ~22:30 IST: the team's final zip was reviewed and synced (e1d5796),
writeups = team text + 5 fixes (60d1b8d; the Gen-Z voice was dropped), zip rebuilt (73 files, identical to
the team's except README / documentation / prototype README).** Organisers' email: code +
methodology submission for the Top-1000 teams due **29 Sep 10:00 AM**. See progress.md
"Team's final zip reviewed".

**Update ~19:55 IST: package layout fixed to the brief (commit 5889cc5).** The zip's
`code/business_entity_resolution/` now holds exactly `src/`, `README.md`, `requirements.txt`
(configs in `src/ber/configs`, tests in `src/tests`, tools in `src/tools`; run as `python -m
ber <stage>` with `PYTHONPATH=code/business_entity_resolution/src`; zip builder
`src/tools/make_submission_zip.py`). Zip rebuilt (73 files), still v2.7_crossenc, all
structure checks pass. HARD RULE: verify any zip against the brief's structure first.

**Update ~19:45 IST: package trimmed + README rewritten (commit dcbaee8, pushed).** Deleted
never-used parts (kaggle/, LLM stage, synthetic-France proxy); README rewritten for reviewers
(details moved to Documentation_template.md Appendix C). `pootato_submission.zip` rebuilt: 76
files, still v2.7_crossenc (sha256 faf49d17…), verified. See progress.md "Package trimmed".

**Update ~18:40 IST: FINAL FILE = v2.7_crossenc (the first v2.7 run, public LB 0.98316),
not v2.7_ber.** The user uploaded v2.7_ber: LB 0.982681. Same method, two draws of the
cross-encoder fine-tuning; the first run is better on labelled val too (paired z 3.9) and
the files differ on 1.35 % of test S1, so the LB gap is not public-subset noise (progress.md
"Final file switched"). `output\` = v2.7_crossenc (validator `--check-ids` PASS);
`pootato_submission.zip` rebuilt from 19b564b and verified (TSVs = v2.7_crossenc, sha256
faf49d17…; code = HEAD; `tools/crossenc_prototype/` = the exact scripts behind its
cross-encoder); docs describe the submitted run. Backups of the older zips in
`D:\aml\submissions\`. Uploads 2026-09-27: 3 of 5; the user re-uploads
`output\matching_results.tsv` (= v2.7_crossenc) as the last upload and submits the zip.
PR / merge still to be asked. The 16:30 update below is superseded where it says v2.7_ber.

**Update ~16:30 IST: final packaging is DONE (section 9)** on `explore/cross-encoder`
(commits 9f5caf2, 247a035 + notes, pushed): the default config reproduces v2.7_ber (config hashes
equal to v26's), `output\` = v2.7_ber (validator `--check-ids` PASS),
Documentation_template.md covers the cross-encoder, `pootato_submission.zip` rebuilt and
verified (105.0 MB; TSVs byte-identical to v2.7_ber; code = HEAD); the v2.6 zip is kept in
`D:\aml\submissions\pootato_submission_v2.6.zip`. Details: progress.md "Final packaging".
What remains is the user's: upload `output\matching_results.tsv` (= v2.7_ber) to the
portal as the last LB upload (so the uploaded file = the zip's file, as the rules ask;
uploads used 2026-09-27: 2 of 5 before it), submit the zip, decide on PR / merge (not
asked yet → ask). **The user stressed that the final ranking uses the private part of the
test set: no public-LB tuning; generalisation first.** Any further experiment must be
judged on validation + the unseen-region check and must not endanger the packaged state
(don't touch `output\` or the zip unless a new file is clearly better on val AND unseen
regions and there are ≥ 3 h left).

**Update 16:10 IST: the v26 run finished OK** (`D:\aml\logs\v26_run.log`, exit 0).
`ber crossenc` rebuilt the same 982,265 training pairs; its cross-encoder has val AUC
0.97417 (prototype 0.97485); stage 2 OOF AUC 0.999059 / logloss 0.0320; **val F0.5 0.99060**
(India 0.98936, US 0.99138; prototype v2.7: 0.99078). Test file
**`D:\aml\submissions\v2.7_ber\`**: validator PASS, 5,895,646 links, candidate_pairs.tsv
unchanged (9,582,745). vs the uploaded prototype v2.7 (LB 0.98316): S1 changed France 3.7 %,
India 1.0 %, US 0.8 %; links per S1 France 3.522 / India 3.371 / US 3.395 (prototype 3.500 /
3.368 / 3.393): retraining noise, as expected. **Steps 1–2 below are done: v2.7_ber is the
final file** (its code is what goes in the zip); continue with step 3 (packaging, section 9).
Optionally the user uploads v2.7_ber once as a check.


1. **A pipeline-native reproduction of v2.7 is running** (started 13:45 in the background of
   the old chat: `bash D:\aml\run_v26.sh` → log `D:\aml\logs\v26_run.log`, stage logs
   `artifacts\v26\<stage>\log.txt`). It runs `ber crossenc` (fine-tune done; scoring val/hold,
   then test) → `ber rerank` → `ber decide` → `ber submit` into
   **`D:\aml\submissions\v2.7_ber\`**. Expected finish ~16:25 IST. Check: `tail
   D:\aml\logs\v26_run.log` (an `exit 0` line at the end = done; the script prints the val
   F0.5 after `decide train`, expected ≈ 0.9908). If the old chat's process died with the
   chat, re-run only the missing steps (see `run_v26.sh`; each `ber` step is resumable by
   phase: e.g. `ber crossenc --run v26 --set data.phase=test ...`).
2. When it is done: compare it with the prototype v2.7 file
   (`.venv\Scripts\python.exe D:\aml\analysis\compare_files.py
   v27=D:\aml\submissions\v2.7_crossenc\matching_results.tsv
   v27ber=D:\aml\submissions\v2.7_ber\matching_results.tsv`): expect a few % of S1 to differ
   (the cross-encoder was retrained; same data, seed and settings, GPU nondeterminism) and
   val F0.5 ≈ 0.9906–0.9910. If it looks sane, **v2.7_ber is the final file** (its code is
   exactly what is in the zip). Optionally the user uploads it once as a check (uploads left
   2026-09-27: 3 of 5). If v26 fails and there is no time, fall back to
   `v2.7_crossenc` (prototype scripts `D:\aml\ce\*.py`, same logic) or to v2.6.
3. Final packaging on `explore/cross-encoder` (section 9), leaving ≥ 3 h before 23:59 IST.

## 1. Situation

- **Best public LB: v2.7 = v2.6 + transformer cross-encoder features in stage 2, LB
  0.98316** (v2.6 0.982344; v2.5 0.981708; v2.4 0.980973; v2.2 0.980; v2.1 0.977; main 0.975).
  Prototype file `D:\aml\submissions\v2.7_crossenc\` (made with `D:\aml\ce\*.py` + run
  `v25f`); pipeline-native file coming in `D:\aml\submissions\v2.7_ber\` (run `v26`).
- **Superseded 18:40: they hold v2.7_crossenc (zip from 19b564b), see section 0.** Before:
  `output\` and `pootato_submission.zip` held **v2.7_ber** (zip built from
  `explore/cross-encoder` 247a035, 2026-09-27 ~16:30); the v2.6 zip is in `submissions\`.
- **Which upload counts:** check on the Unstop page whether the private leaderboard uses
  the last or the best upload; make sure the final file is the one that counts.
- The public LB shows 6 decimals (resolves ~0.0005); it is a subset of test S1, the
  private LB (final ranking) the rest. `candidate_pairs.tsv` and the code are reviewed too
  (smaller candidate set per S1 ranks higher; ours: 5.53 per S1, unchanged since v2.5).
- **Deadline:** ≈ Sun 27 Sep 2026 23:59 IST. **5 LB uploads per team per day**; used on
  2026-09-27: v2.6, v2.7 (2 of 5).
- Team priority (explicit): a well-generalising, defensible approach over squeezing the
  public score. The user uploads files themselves; we prepare them and say which one.

## 2. People, repo, rules for working

- Team: Sreenitya Thatikunta (wrote `main`), Rudraksha Dev Sharma (the user; email
  rudrakshash@gmail.com, git user `RudrakshaSh`).
- GitHub: https://github.com/SreenityaThatikunta/amazon-ml-challenge
  - `main` = Sreenitya's pipeline (val 0.98211, LB 0.975). **Never commit to main.**
  - `improve/v2` = our GBM work up to **v2.6** (pushed, 2d9cbc0). **Do not push the new
    architecture onto it** (user, 2026-09-27).
  - `explore/cross-encoder` = v2.7 (cross-encoder stage, XGBoost option, notes), pushed
    (last commit d105ce2+). New experiments → new branches.
  - Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  - Ask the user before opening a PR / merging into main (not done so far).
- Keep `progress.md` updated with the **logic** behind every change; record every LB
  submission and why.
- **RAM:** 16 GB laptop shared with the user's apps (often 2–6 GB free). One heavy stage at
  a time; no analyses next to a stage. Don't scale runs beyond ~8 GB peak (the user declined
  hold_frac 0.6 at 9–11 GB). Overnight runs are fine but must end by 05:30–06:00.
- Downloads need the user's explicit OK (name, source, size). Approved and installed on
  2026-09-27: torch 2.14.0+cu126 (PyTorch index), transformers 5.17.0, tokenizers,
  safetensors, huggingface_hub, xgboost 3.4.1; model `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- The user likes short status updates while long jobs run, and honest timing estimates.
- GPU note: Task Manager's GPU page does not show CUDA work by default (only 3D/Copy/Video);
  `nvidia-smi` shows it (the user asked once whether the NVIDIA GPU is used: it is).

## 3. Machine and folders

- Windows 11, 16 threads, 16 GB RAM, **RTX 4050 Laptop 6 GB** (CUDA 13.1, driver 592; also
  an AMD 780M iGPU that drives the display). C: ~10 GB free → everything on D: (pip cache and
  temp: `TMP=TEMP=D:\aml\tmp`, `--no-cache-dir`; `HF_HOME=D:\aml\hf_home`).
- `D:\aml\amazon-ml-challenge\` = the git clone (`dataset\` gitignored, `.venv\` Python 3.13,
  CLI `.venv\Scripts\ber.exe`).
- `artifacts\<run>\<stage>\` (gitignored). Runs that matter:
  - `v2` = production upstream (normalize, candidates, features: train + test). Don't delete.
  - `v25` = stage 1 (scorer, trained without the 40 % hold regions) + `features_hold` (hold
    0.4, owner rule); all later runs link to it. `v25\exp\` = v2.5 experiments.
  - `v25b` = **v2.6** stage 2 (`v25b\rerank` + `exp\E1b_nc`), its test pass and submit.
  - `v25c` = bagging + v2.6 unseen-region experiments; `v25d` = cross-encoder / XGBoost
    stage-2 experiments (`v25d\exp\E1b_nc_ce` = v2.7's stage-2 models, `E3b_nc_ce` = its
    unseen-region check); `v25e` = unused; **`v25f` = v2.7 prototype test pass** (models
    restored from `v25d\exp\E1b_nc_ce`, manifest rebuilt from v25b's).
  - **`v26` = v2.7 reproduced by the pipeline's own stages** (running; section 0).
  - Older: `v24` (v2.4), `v25chk`, `v2x`, `v23`, `v2m`, `v24m`, `dev2`, `main`.
- `D:\aml\ce\` = cross-encoder prototype: `prep.py` (training pairs + texts → `data\`),
  `train.py` (fine-tune → `D:\aml\models\ce_v1\`), `score.py` (→ `scores_v1\`: val, hold,
  test_<country>.parquet with q_id, s1_id, ce), `alone_eval.py` (unused). The package stage
  `ber crossenc` (`src/ber/crossenc/stage.py`) is the same logic.
- `D:\aml\models\ms-marco-MiniLM-L-6-v2\` (base model, Apache-2.0), `ce_v1\` (prototype
  fine-tuned model); `v26`'s own model: `artifacts\v26\crossenc\model\`.
- Scripts: `D:\aml\run_v25.sh [--skip-scorer] <case...>` (named stage-2 experiments; `RUN=`
  picks the run dir), `D:\aml\run_v25_test.sh <out name> [--set ...]` (decide sweep + test
  pass + submit; `RUN=`, `SKIP_SCORER=1`), `D:\aml\run_v26.sh` (pipeline-native v2.7),
  `D:\aml\analysis\clean_eval.py name=dir ...` (val F0.5 on all / "clean" / affected S1),
  `compare_files.py` (per-country links, empty rate, S1 changed between submission files).
- `D:\aml\submissions\` = every candidate file (v2 … v2.6_normcomplete, v2.7_crossenc,
  probes); `D:\aml\logs\` stage logs; `D:\aml\analysis\` analysis scripts + outputs.

## 4. How to run (details: `code/business_entity_resolution/README.md`)

Git Bash, from `D:\aml\amazon-ml-challenge`, `export PYTHONUTF8=1`. Stages:
`split normalize candidates features features-hold scorer crossenc rerank decide submit`;
each `ber <stage> --run <run> [--set key=value ...]`; `--set data.phase=train|test` runs
one side (a test pass reuses the train pass's models/params). v2.7 is the default
(`rerank.extra_scores: crossenc` in `configs/default.yaml` since 9f5caf2; `null` = v2.6);
`crossenc.base_model` = HF id (default) or `D:/aml/models/ms-marco-MiniLM-L-6-v2`
(offline; `HF_HUB_OFFLINE=1`). Tests: `cd code/business_entity_resolution &&
../../.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --basetemp=/d/aml/pytest_tmp`
(147 pass).

| Stage | Time | Peak RAM |
|---|---|---|
| normalize | 7 min | 4 GB |
| candidates | ~2.5 h train (GPU) + 78 min test | 8 GB |
| features | 64 min train + 115 min test | 10 GB |
| features-hold (hold 0.4) | 79 min | 8.9 GB |
| scorer | 10 min train + 12 min test | 9.3 GB |
| **crossenc** (GPU) | pairs 2.5 min + fine-tune 18 min + score val 5 / hold 28 min; test ~60 min (India slowest) | ~3 GB, GPU 2.3–5.9 GB |
| rerank (5.4 M pairs) | ~22 min train + 9–13 min test | 8 GB / 5 GB |
| decide / submit | 2–5 min / 1 min | 4.6 GB |

## 5. What the pipeline is now (v2.7)

Normalise → region-blocked char-3gram TF-IDF top-k (~24.6 S1 per record, 99.05 % of true
pairs) → 102 pair features → **stage 1** LightGBM (p1), trained on train regions outside
the 40 % hold regions; keep p1 ≥ 0.001 (**= `candidate_pairs.tsv`, 5.53 per S1**) →
**cross-encoder**: MiniLM (`cross-encoder/ms-marco-MiniLM-L-6-v2`, Apache-2.0, 22 M params)
fine-tuned on stage 1's own out-of-sample pairs (train-side records outside the hold regions
that stage 1 did not train on, p1 ≥ 0.001: 982,265 pairs), text = "name core + legal ;
address + region"; scores every stage-2 pair → **stage 2** re-ranker LightGBM on p1 + group
context + 102 stage-1 features + **cross-encoder logit and its margin/rank within the
record's candidates and the S1's group** (`rerank.extra_scores`), domain normalisation
(S1-group count scales from complete groups), cross-fitted (4 folds by top S1) on val + 40 %
hold regions (owner rule), trained on complete groups only → one owner per record → anchor
correction (France's generic swaps, ×2.7 with v2.7) → per-S1 expected F0.5 (min_p 0.02).

## 6. Results

| Version | Val F0.5 | Unseen-region CV | Public LB |
|---|---|---|---|
| main (6120d59) | 0.98211 | – | 0.975 |
| v2.1 stage 2 on all features | 0.98895 | 0.98684 | 0.977 |
| v2.2 + domain normalisation | 0.98897 | – | 0.980 |
| v2.4 stage 2 on val + 20 % held-out regions (+ anchor) | 0.98911 | 0.98776 | 0.980973 |
| v2.5 40 % hold regions by owner + complete groups | 0.98923 | 0.98822 | 0.981708 |
| v2.6 + normalisation scale from complete groups | 0.98918 | 0.98817 | 0.982344 |
| **v2.7 + cross-encoder features** | **0.99078** | **0.98988** | **0.98316** |

Clean subset (val S1 with complete groups in every variant): v2.6 0.99263 → v2.7 0.99400;
unseen regions 0.99163 → 0.99309; singletons 0.98892 → 0.99385.

## 7. Key findings (evidence in `progress.md`; scripts in `D:\aml\analysis\`, `D:\aml\ce\`)

1. Test ≠ train in density → domain normalisation of size-dependent stage-2 features (+0.003).
2. France's generic-word swaps (same address + house number, generic word swapped): the
   house-number anchor says ~80 % are true copies → anchor correction per name-change kind.
3. Incomplete S1 groups in stage-2 training (records of unloaded regions) → hold regions,
   owner rule, training on complete groups only, normalisation scale from complete groups:
   each aligned training with test and gained more on the LB than val showed.
4. Val calibration is near-perfect per segment; a label-free anchor check (house-number
   profiles from labelled val) shows the model calibrated on the US's largest test segment
   too (0.756 vs 0.762). ~60 % of missed val pairs are empty-address records (unresolvable).
5. **Cross-encoder alone is weak** (val AUC 0.975 on stage-2 pairs vs 0.9989 for stage 2;
   train loss 0.06 vs val 0.31: it generalises poorly across regions, and sees house numbers
   only as word pieces), **but its margins over competing candidates are strong stage-2
   evidence** (features ranked 4/5/9 of 136): val +0.0016, unseen regions +0.0017, LB +0.0008.
6. Tried and rejected (val-flat or worse): 3,500-tree cap, second stage-2 round (India
   −0.0011 unseen), stronger regularisation, bagging (2 fold splits), min_p1 0.003 (small
   cost), stage-1 domain normalisation (stage 1 barely uses those features), label-shift EM,
   adversarial weights. XGBoost ensemble (`rerank.learner=both`) coded but never evaluated
   (stopped for the cross-encoder).

## 8. Next steps

1. Done: v26 finished and packaged; after its LB (0.982681) the first run v2.7_crossenc
   (0.98316, better on val, z 3.9) became the final file and is packaged (section 0).
2. The user: upload `output\matching_results.tsv` once (final LB file = the zip's file;
   regression check only, expected ≈ 0.983 ± public-subset noise), submit the zip.
3. Optional, only with the user's OK: evaluate `rerank.learner=both` (XGBoost + LightGBM)
   on top of v2.7 (`RUN=v25d bash D:\aml\run_v25.sh --skip-scorer E1b_nc_ce_xgb` and
   `E3b_nc_ce_xgb`; ~30 min each, GPU). **Stage-2 training peaks at ~8 GB RAM** (v25d/v26:
   8.0–8.3 GB), above the ~5 GB cap the user set on 2026-09-27, so it needs their OK; adopt
   only if it gains on val AND on the unseen-region check (bagging with 2 fold splits gained
   nothing, so expect little), then test pass + decide + submit + re-package (~1.5 h). A
   stronger cross-encoder (more epochs, larger MiniLM) needs ~2 h of GPU scoring and the
   ~10 GB crossenc stage: not before the deadline.
4. Ask the user about PR / merges (explore → improve/v2? → main).

## 9. Before the deadline (leave ≥ 3 h) — DONE 2026-09-27 ~16:30 (progress.md "Final packaging")

On branch `explore/cross-encoder`:
- `configs/default.yaml`: add `rerank.extra_scores: crossenc` (and a `crossenc:` section
  with the dataclass defaults, documented) so the default config reproduces v2.7.
- Copy the final file (v2.7_ber, else v2.7_crossenc) to `output\` (both TSVs).
- `Documentation_template.md`: add the cross-encoder (sections 1, 2.2 diagram, 4 new
  paragraph, 5 results incl. LB 0.98316 and the "alone is weak / hybrid helps" finding,
  Appendix A run commands with `ber crossenc`, license: Apache-2.0, 22 M params ≤ 8 B).
  README already documents the stage (`requirements-crossenc.txt` pins torch etc.).
- `progress.md` final entry; rebuild the zip (`.venv/Scripts/python.exe
  code/business_entity_resolution/src/tools/make_submission_zip.py --team pootato`, checks CRC;
  it zips the working tree's code, so the branch must be checked out), run the official
  validator (`.venv/Scripts/python.exe utils/validate_submission.py --matching
  output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
  --check-ids`), commit + push the branch, tell the user which file to upload / where the zip
  is, ask about PR / merge.

## 10. Gotchas

- New corpus statistics must be learned identically for train and test.
- `data.phase=test` merges the stage's previous manifest; a train pass must exist first.
  `ber decide --set data.phase=test` reads `decision_params.json` from the train pass.
- Experiment scripts overwrite `<run>\rerank` per experiment; save per experiment (`exp\`).
  `run_v25.sh`'s save() copies rerank's then decide's `manifest.json` into the same exp
  folder, so the saved one is decide's (a restored rerank dir needs a rebuilt manifest, as
  done for v25f).
- `hold_frac > 0` needs `ber features-hold` before `ber scorer`.
- Filter by id with Arrow (`pc.is_in`), not `Series.isin(python_set)`.
- Stage `log.txt` files accumulate across runs; `cmd; cat log` hides cmd's exit code; killing
  a queue's wrapper shell loses its trailing `echo exit` line (chained waiters then hang).
- pandas 3 hands out read-only numpy arrays: no in-place `&=` on them.
- LightGBM picks row-/col-wise histograms by timing: re-runs differ by ~1e-4.
- The default config has `train_complete: true`, `norm_complete: true`; experiment cases
  without them pass `false` explicitly (run_v25.sh does).
- Cross-encoder: a CUDA out-of-memory in the epoch-end validation pass killed the first
  training run; inference now halves its batch on OOM. `rerank.extra_scores` files must be in
  stage 2's row order (checked; falls back to a join on (q_id, s1_id)).
- Windows: absolute D: paths; heredocs with nested quotes can break — write scripts to files;
  `sleep` loops are blocked in the harness (use background jobs / `until` loops).
