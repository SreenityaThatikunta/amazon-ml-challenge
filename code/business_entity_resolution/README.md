# business_entity_resolution

Pipeline for the Amazon ML Challenge 2026 Business Entity Resolution task.
Design: [../../plan.md](../../plan.md) · status and metrics: [../../progress.md](../../progress.md) ·
EDA: [../../eda/FINDINGS.md](../../eda/FINDINGS.md).

## Setup

From the `student_resource/` directory (Python 3.12; the data in `dataset/`):

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r code/business_entity_resolution/requirements.txt
.venv/bin/pip install -e code/business_entity_resolution
```

Check the environment:

```bash
cd code/business_entity_resolution && ../../.venv/bin/pytest && ../../.venv/bin/python tools/check_licenses.py
```

## Reproduce end-to-end

Every stage is one command that reads the previous stage's artifacts from
`artifacts/<run>/` and writes its own (with `manifest.json` + `log.txt`).
Run them one at a time: several heavy stages in parallel will swap on a 16 GB machine.

```bash
B=.venv/bin/ber
F=code/business_entity_resolution/configs/france_train.yaml
$B split       --run main   # geographic train/val split + dev slice          (~2 min)
$B normalize   --run main   # normalise 24M records, learn Indic dictionaries (~4 min)
$B candidates  --run main   # region-blocked TF-IDF top-k + recall report      (~1.5 h)
$B features    --run main   # pair + context + lookalike features, 4 workers   (~2 h)

# synthetic French training pairs for the unseen-country model (France is test-only)
$B synth-france --run main  # (writes the proxy's eval_s1.tsv, which frtrain excludes)
$B synth-france --run frtrain -s seed=7 -s france.n_entities=60000 \
    -s france.exclude=artifacts/main/synth-france/data/eval_s1.tsv
for s in split normalize candidates features; do $B $s -c $F; done   # (~8 min)

$B scorer      --run main -s scorer.oof_folds=1 -s 'scorer.unseen_extra_train=[frtrain]'
$B rescore     --run main   # M8: cluster-context second stage (val cross-fit)
$B decide      --run main -s decide.scores_stage=rescore   # one owner + expected-F0.5; val F0.5
$B submit      --run main   # output/matching_results.tsv + candidate_pairs.tsv + validator
```

The same sequence (from `features` on) is scripted in `../../experiments/run_main_v2.sh`.
What each stage adds, with the measured gains, is logged in `../../experiments/README.md`.

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
| `src/ber/candidates/` | Stage 1: per-country, region-blocked char-TF-IDF top-k candidates |
| `src/ber/features/` | Stage 2: pair + context features |
| `src/ber/scorer/` | Stage 2: LightGBM scorer (+ out-of-fold scores); unseen-country model for S1 countries absent from training |
| `src/ber/cluster/` | Stage 2b (M8): second-stage rescore with cluster context, trained on val (cross-fit) |
| `src/ber/llm/` | Stage 3: routing, select-prompts, LLM score import/blend |
| `src/ber/decide/` | Stage 4: one-owner assignment, expected-F0.5 selection, val metrics |
| `src/ber/france/` | Synthetic French labelled data (validation proxy; training pairs for the unseen-country model) |
| `src/ber/eval/` | Metrics (same as `utils/score.py`), geographic split |
| `src/ber/{config,artifacts,io,log,cli,submit,explain}.py` | Infrastructure |
| `configs/` | `default.yaml` (every key documented), `dev.yaml`, `france_proxy.yaml`, `france_train.yaml` |
| `kaggle/` | LoRA training / inference scripts for Kaggle GPUs |
| `tools/` | License check |
| `tests/` | pytest suite |
