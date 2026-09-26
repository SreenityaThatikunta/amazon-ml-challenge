#!/usr/bin/env bash
# N1 (plan.md §0.1b): v3's models on (a) the val side rebuilt val-only, no reshape (run v3vo,
# control: should reproduce v3's val 0.98954) and (b) test-shaped val (run v3ts).
# Log: experiments/logs/val_ts.log
# STAGES overrides the stage list (e.g. STAGES="features scorer rerank decide").
set -euo pipefail
STAGES=${STAGES:-"normalize candidates features scorer rerank decide"}
B=.venv/bin/ber
C=code/business_entity_resolution/configs/val_ts.yaml
for s in $STAGES; do
  $B $s -c $C -r v3vo -s 'reshape.drop_s1={}'
done
for s in $STAGES; do
  $B $s -c $C -r v3ts
done
