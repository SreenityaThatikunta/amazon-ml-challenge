#!/usr/bin/env bash
# v3 dev-slice smoke run (plan.md §0, A2): the merged pipeline end to end on the
# dev slice, run id devv3 (artifacts/dev untouched). Log: experiments/logs/dev_v3.log
set -euo pipefail
B=.venv/bin/ber
C=code/business_entity_resolution/configs/dev.yaml
for s in normalize candidates features scorer rerank decide; do
  $B $s -c $C -r devv3
done
