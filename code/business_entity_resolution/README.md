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
(peak ~10 GB per stage).

```bash
B=.venv/bin/ber            # Windows: .venv\Scripts\ber.exe (set PYTHONUTF8=1)
$B split       --run v3    # geographic train/val split + dev slice            (~3 min)
$B normalize   --run v3    # normalise 24M records, learn Indic dictionaries   (~7 min)
$B candidates  --run v3    # region-blocked TF-IDF top-k + recall report       (~1.5 h GPU / ~4 h CPU, train+test)
$B features    --run v3    # pair + context + key + lookalike features         (~4 h, train+test)
$B scorer      --run v3    # stage-1 LightGBM, scores val + test pairs         (~20 min)
$B rerank      --run v3    # stage-2 re-ranker (cross-fitted on val) + test    (~15 min)
$B decide      --run v3    # one owner per record + expected F0.5; val F0.5     (~10 min)
$B submit      --run v3    # output/matching_results.tsv + candidate_pairs.tsv + official validator
```

The train/val side and the test side can also run as two passes (the test pass reuses
the model, region neighbourhoods and decision parameters of the train pass):

```bash
for s in candidates features scorer rerank decide; do $B $s --run v3 --set data.phase=train; done
for s in candidates features scorer rerank decide submit; do $B $s --run v3 --set data.phase=test; done
```

`output/candidate_pairs.tsv` is the input of the final model: the pairs that passed the
stage-1 filter (`rerank.min_p1`), ~1 per S2/S3 record. `decide.scores_stage=scorer` gives
the stage-1-only pipeline (then candidate_pairs.tsv holds every TF-IDF candidate, ~3 GB).
`ber rescore` + `decide.scores_stage=rescore` is main's alternative stage 2 (M8).
What each change adds, with the measured gains, is logged in `../../experiments/README.md`.

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

France validation proxy (synthetic labelled French data, scored with the main model's
unseen-country model; every test S1 France is indexed, 20k of them are scored):

```bash
$B synth-france --run main
P=code/business_entity_resolution/configs/france_proxy.yaml
for s in split normalize candidates features; do $B $s -c $P; done
$B scorer -c $P -s scorer.model_from_run=main && $B decide -c $P
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
| `src/ber/cluster/` | Stage 2b alternative (main's M8): rescore with cluster context, cross-fitted on val |
| `src/ber/llm/` | Stage 3: routing, select-prompts, LLM score import/blend |
| `src/ber/decide/` | Stage 4: one-owner assignment, expected-F0.5 selection, val metrics |
| `src/ber/france/` | Synthetic French labelled data (validation proxy; training pairs for the unseen-country model) |
| `src/ber/eval/` | Metrics (same as `utils/score.py`), geographic split |
| `src/ber/{config,artifacts,io,log,cli,submit,explain}.py` | Infrastructure |
| `configs/` | `default.yaml` (every key documented), `dev.yaml`, `france_proxy.yaml`, `france_train.yaml` |
| `kaggle/` | LoRA training / inference scripts for Kaggle GPUs |
| `tools/` | License check |
| `tests/` | pytest suite |
