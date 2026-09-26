#!/usr/bin/env bash
# N1b (plan.md §0.1b): stage 2 retrained (cross-fitted) on val_ts, v3's stage 1 unchanged:
# v3ts_r without, v3ts_rc with the N4 copy-group features. Upstream stages are symlinks
# to v3ts (experiments/fork_run.py). Log: experiments/logs/n1b.log
set -euo pipefail
B=.venv/bin/ber
C=code/business_entity_resolution/configs/val_ts.yaml
for s in rerank decide; do $B $s -c $C -r v3ts_r  -s rerank.model_from_run=null; done
for s in rerank decide; do $B $s -c $C -r v3ts_rc -s rerank.model_from_run=null -s rerank.copy_features=true; done
