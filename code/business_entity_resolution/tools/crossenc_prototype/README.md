# Cross-encoder prototype scripts (provenance of the submitted file)

These three scripts are **exact, unmodified copies** of the scripts that fine-tuned and
scored the cross-encoder behind the submitted `output/matching_results.tsv` (v2.7, first
run, 2026-09-27; public leaderboard 0.98316). Stages 2–4 of that file (re-ranker, decision,
submission) ran with this package's own code (`ber rerank / decide / submit`).

| Script | What it does |
|---|---|
| `prep.py` | training pairs: train-side records outside the hold regions that stage 1 did not train on (`stable_unit(q_id, seed + 4) >= 0.14`), kept at p1 ≥ 0.001 (982,265 pairs, 68 % true); record texts `name core + legal ; address + region` for the training pairs and for every stage-2 pair (val, hold, test), in stage 2's row order |
| `train.py` | fine-tunes `cross-encoder/ms-marco-MiniLM-L-6-v2` (Apache-2.0, 22.7 M parameters): 2 epochs, batch 256, AdamW lr 5e-5, weight decay 0.01, 5 % linear warm-up, fp16, max 64 tokens, BCE on the match label |
| `score.py` | scores every stage-2 pair → `<out>/{val,hold,test_France,test_India,test_US}.parquet` (q_id, s1_id, ce) |

The paths are those of our machine (`D:\aml\...`): artifacts of the runs `v2` (normalize,
features), `v25` (scorer, features-hold) and `v25b` (stage-2 row order). Adapt `A`, `OUT`,
`BASE`, `DATA`, `MODEL` to your run. The submitted file was produced by

```bash
python prep.py && python train.py 2 && python score.py      # -> D:\aml\ce\scores_v1
ber rerank --run <run> --set rerank.extra_scores=D:/aml/ce/scores_v1   # reads <dir>/{val,hold,test_<country>}.parquet
ber decide --run <run> --set rerank.extra_scores=D:/aml/ce/scores_v1
ber submit --run <run> --set rerank.extra_scores=D:/aml/ce/scores_v1
```

**The pipeline stage `ber crossenc` (`src/ber/crossenc/stage.py`) is the same logic**
packaged as a stage: the same pair selection rule (both runs: 982,265 training pairs,
68.0 % true), text, model and hyper-parameters; the default config runs it
(`rerank.extra_scores: crossenc`).
The only difference: `train.py` draws a 200,000-pair validation sample from its random
generator before training (for an epoch-end AUC printout), so its batch order differs.
GPU fine-tuning is not bit-for-bit reproducible either, so every run is a slightly
different draw: a re-run with `ber crossenc` gave validation macro F0.5 0.99060 (public
leaderboard 0.982681) against 0.99078 (0.98316) for this run. We submit this run: it is
better on the labelled validation set (paired per-entity difference +0.00018, z ≈ 3.9)
and the public leaderboard agrees.
