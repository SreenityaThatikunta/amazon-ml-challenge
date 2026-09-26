#!/usr/bin/env bash
# E30: France rehearsal on val_ts (v3's stage 1). Stage 2 trained on one country, applied to
# the other (leave one country out); then + 1 round of pseudo-label self-training on the
# held-out country. Log: experiments/logs/loco.log
set -euo pipefail
B=.venv/bin/ber
C=code/business_entity_resolution/configs/val_ts.yaml
O="-s rerank.model_from_run=null -s rerank.fold_by=country -s rerank.copy_features=true"
for s in rerank decide; do $B $s -c $C -r v3ts_loco $O; done
for s in rerank decide; do $B $s -c $C -r v3ts_loco_pl $O -s rerank.pseudo_rounds=1; done
