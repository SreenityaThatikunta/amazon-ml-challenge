#!/usr/bin/env bash
# N3 (plan.md §0.1b): train pass on test-shaped data (reshape on train + val), run v4.
# Its decide F0.5 is a val_ts score. Log: experiments/logs/n3.log
set -euo pipefail
B=.venv/bin/ber
C=code/business_entity_resolution/configs/train_ts.yaml
for s in normalize candidates features scorer rerank decide; do $B $s -c $C; done
