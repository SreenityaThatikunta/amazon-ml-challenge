# business_entity_resolution

Pipeline for the Amazon ML Challenge 2026 Business Entity Resolution task.
Design: [../../plan.md](../../plan.md) · status and metrics: [../../progress.md](../../progress.md) ·
EDA: [../../eda/FINDINGS.md](../../eda/FINDINGS.md).

## Setup

From the `student_resource/` directory (the directory that contains `dataset/`;
Python 3.11–3.13, tested with 3.12 on macOS and 3.13 on Windows 11):

```bash
python -m venv .venv
.venv/bin/pip install -r code/business_entity_resolution/requirements.txt      # Windows: .venv\Scripts\pip
.venv/bin/pip install -e code/business_entity_resolution
# optional, NVIDIA GPU + CUDA 13 toolkit: GPU top-k search in stage 1 (~2x faster end to end)
.venv/bin/pip install -r code/business_entity_resolution/requirements-gpu.txt
```

Check the environment:

```bash
cd code/business_entity_resolution && ../../.venv/bin/pytest && ../../.venv/bin/python tools/check_licenses.py
```

## Reproduce end-to-end

Every stage is one command that reads the previous stage's artifacts from
`artifacts/<run>/` and writes its own (with `manifest.json` + `log.txt`).
Run them one at a time: heavy stages in parallel swap on a 16 GB machine
(peak memory: candidates ~8 GB, features ~10 GB, features-hold ~9 GB, scorer ~9 GB,
rerank ~8 GB, decide ≤ 5 GB).

```bash
B=.venv/bin/ber            # Windows: .venv\Scripts\ber.exe (set PYTHONUTF8=1)
$B split       --run main  # geographic train/val split + dev slice            (~3 min)
$B normalize   --run main  # normalise 24M records, learn Indic dictionary     (~7 min)
$B candidates  --run main  # region-blocked TF-IDF top-k + recall report       (~1.5 h GPU / ~4 h CPU, train+test)
$B features    --run main  # 102 pair + context features                       (~4 h, train+test)
$B features-hold --run main  # all pairs of the held-out train regions (rerank.hold_frac) (~1.3 h)
$B scorer      --run main  # stage-1 LightGBM (without the held-out regions), scores val + hold + test (~25 min)
$B rerank      --run main  # stage-2 re-ranker (cross-fitted on val + held-out regions) + test (~25 min)
$B decide      --run main  # one owner per record + expected F0.5 (+ anchor prior correction); val F0.5 (~10 min)
$B submit      --run main  # output/matching_results.tsv + candidate_pairs.tsv + official validator
```

The train/val side and the test side can also run as two passes (the test pass reuses
the model, region neighbourhoods and decision parameters of the train pass):

```bash
for s in candidates features features-hold scorer rerank decide; do $B $s --run main --set data.phase=train; done
for s in candidates features scorer rerank decide submit; do $B $s --run main --set data.phase=test; done
```

Stage 2 (`rerank`) trains on the validation regions and on ~40 % more whole train-side
regions that stage 1 never sees (`rerank.hold_frac: 0.4`, `rerank.train_sides: [val,
hold]`; `hold_frac: 0` gives the v2.2 setup: validation regions only, no `features-hold`).
Held-out S2/S3 records follow their owner's region, as in the validation split
(`rerank.hold_by: owner`; v2.4 used `query_region`), and stage 2 trains only on pairs
whose S1 group is complete (`rerank.train_complete: true`), as every group is at test time;
the domain normalisation takes the scale of the S1-group counts from those pairs too
(`rerank.norm_complete: true`).
`decide.prior_shift: anchor` corrects a name-change kind's probabilities when the
house-number evidence in the test data says the model misjudges how often it is a true
copy (on current data only France's generic-word swaps; `none` switches it off).

`output/candidate_pairs.tsv` is the input of the final model: the pairs that passed the
stage-1 filter (`rerank.min_p1`), ~1 per S2/S3 record. `decide.scores_stage=scorer` gives
the stage-1-only pipeline (then candidate_pairs.tsv holds every TF-IDF candidate, ~3 GB).

LLM stage (optional, on Kaggle; see [kaggle/README.md](kaggle/README.md)):

```bash
$B llm-export  --run main   # routed prompts -> Kaggle
# ... train + infer on Kaggle, put llm_scores_{val,test}.parquet in artifacts/llm_scores/
$B llm-import  --run main   # blend LLM + scorer, weight chosen on val
$B decide      --run main --set decide.scores_stage=llm-import
$B submit      --run main
```

Fast iteration on a small slice of states (whole pipeline in minutes):

```bash
$B normalize -c code/business_entity_resolution/configs/dev.yaml   # likewise for every stage
```

France validation proxy (synthetic labelled French data, scored with the main model):

```bash
$B synth-france --run main
P=code/business_entity_resolution/configs/france_proxy.yaml
for s in split normalize candidates features scorer decide; do $B $s -c $P; done
```

Debugging: `$B explain <entity_id> --run main` traces one record through every
stage. `$B config` prints the resolved configuration.

## Layout

| Path | Contents |
|---|---|
| `src/ber/normalize/` | Stage 0: text/address normalisation, region detection, Indic transliteration, lookup tables (`tables/*.tsv`) |
| `src/ber/candidates/` | Stage 1: per-country, region-blocked char-TF-IDF top-k candidates (CPU or GPU) |
| `src/ber/features/` | Stage 2: pair + context features |
| `src/ber/scorer/` | Stage 2: stage-1 LightGBM scorer (also the learned candidate filter) |
| `src/ber/rerank/` | Stage 2b: re-ranker with S1/record group context, cross-fitted on val |
| `src/ber/llm/` | Stage 3: routing, select-prompts, LLM score import/blend |
| `src/ber/decide/` | Stage 4: one-owner assignment, expected-F0.5 selection, val metrics |
| `src/ber/france/` | Synthetic French labelled data (validation proxy / LLM training) |
| `src/ber/eval/` | Metrics (same as `utils/score.py`), geographic split |
| `src/ber/{config,artifacts,io,log,cli,submit,explain}.py` | Infrastructure |
| `configs/` | `default.yaml` (every key documented), `dev.yaml`, `france_proxy.yaml` |
| `kaggle/` | LoRA training / inference scripts for Kaggle GPUs |
| `tools/` | License check |
| `tests/` | pytest suite |
