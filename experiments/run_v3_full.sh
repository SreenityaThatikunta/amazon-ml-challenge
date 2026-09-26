#!/usr/bin/env bash
# Full v3 run (plan.md §0, task 1.1), run id v3, two passes because features.prune_test
# needs the stage-1 model before the test features. Log: experiments/logs/v3_full.log
set -euo pipefail
B=.venv/bin/ber
R="-r v3"
$B normalize $R                                   # train + test records
for s in candidates features scorer rerank decide; do
  $B $s $R --set data.phase=train                 # train/val side: models + decision params
done
for s in candidates features scorer rerank decide submit; do
  $B $s $R --set data.phase=test                  # test side, pruned features, submission
done
