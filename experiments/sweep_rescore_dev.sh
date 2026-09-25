#!/usr/bin/env bash
# M8 rescore sweep on the dev slice (2026-09-25): top-k and cluster-confidence threshold.
# Needs the dev scorer (B1 baseline). Log: experiments/logs/sweep_rescore_dev.log
set -euo pipefail
cd "$(dirname "$0")/.."
B=.venv/bin/ber; D=code/business_entity_resolution/configs/dev.yaml
for k in 3 5; do for conf in 0.3 0.5 0.7; do
  $B rescore -c $D -s rescore.k=$k -s rescore.conf=$conf > /dev/null 2>&1
  $B decide -c $D -s decide.scores_stage=rescore > /dev/null 2>&1
  python3 -c "
import json; m=json.load(open('artifacts/dev/decide/manifest.json'))['metrics']; r=json.load(open('artifacts/dev/rescore/manifest.json'))['metrics']
print('k=$k conf=$conf', 'val_f05', m['val_f05'], {c:v['f05'] for c,v in m['val_by_country'].items()}, 'auc', r['val_auc_rescore'], 'P/R', m['val_overall']['pair_p'], m['val_overall']['pair_r'])"
done; done
