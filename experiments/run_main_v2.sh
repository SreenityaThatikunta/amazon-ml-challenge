#!/usr/bin/env bash
# Full-scale v2 (2026-09-25):
#   lookalike features (`ber features --run main`, E8; parallel since E12)
#   -> scorer + unseen-country model trained with synthetic French pairs (run frtrain, E6)
#   -> rescore (M8 cluster context, seen countries only, E9)
#   -> decide on rescored scores -> submit (matching_results + candidate_pairs + validator)
#   -> France proxy scored with the new stage-1 models (France is not rescored).
# Every stage runs in sequence here (set -e): a stage only starts after the previous one
# succeeded. SKIP_FEATURES=1 reuses existing features. Log: experiments/logs/run_main_v2.log
set -euo pipefail
cd "$(dirname "$0")/.."   # student_resource/
B=.venv/bin/ber
P=code/business_entity_resolution/configs/france_proxy.yaml
[[ "${SKIP_FEATURES:-0}" == 1 ]] || $B features --run main   # parallel build_features (4 workers)
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
