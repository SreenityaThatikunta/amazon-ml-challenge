#!/usr/bin/env bash
# Full-scale v2 (2026-09-25):
#   lookalike features (built by `ber features --run main`, E8)
#   -> scorer + unseen-country model trained with synthetic French pairs (run frtrain, E6)
#   -> rescore (M8 cluster context, seen countries only, E9)
#   -> decide on rescored scores -> submit (matching_results + candidate_pairs + validator)
#   -> France proxy scored with the new stage-1 models (France is not rescored).
# Set FEATURES_PID to wait for a running `ber features` first (pgrep -f would also match
# wrapper shells). Log: experiments/logs/run_main_v2.log
set -euo pipefail
cd "$(dirname "$0")/.."   # student_resource/
B=.venv/bin/ber
P=code/business_entity_resolution/configs/france_proxy.yaml
if [[ -n "${FEATURES_PID:-}" ]]; then while kill -0 "$FEATURES_PID" 2>/dev/null; do sleep 30; done; fi
python3 -c "import json,sys; m=json.load(open('artifacts/main/features/manifest.json')); sys.exit(m['status']!='ok')" \
  || { echo "features failed"; exit 1; }
# oof_folds=1: out-of-fold train scores are only needed by the LLM stage (not used now)
$B scorer --run main -s scorer.oof_folds=1 -s 'scorer.unseen_extra_train=[frtrain]'
$B decide --run main                                   # stage-1 only, for comparison
cp artifacts/main/decide/manifest.json experiments/logs/main_v2_decide_stage1.json
$B rescore --run main
$B decide --run main -s decide.scores_stage=rescore
$B submit --run main -s decide.scores_stage=rescore
$B scorer -c $P -s scorer.model_from_run=main
$B decide -c $P
echo "v2 done"
