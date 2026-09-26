# main-only (pre-v3): uses scorer.extra_train / unseen_extra_train, which the v3 scorer
# does not have (synthetic-France training dropped, plan.md §0 A3). Kept as a record of E6/E11.
set -e
# Dev A/B for the France work (2026-09-25): B1 = new lookalike features, B2 = B1 + synthetic
# French training pairs (run frtrain). Each variant is scored on dev val (US/India) and on the
# France proxy (run synthfr, model_from_run=dev). Log: experiments/logs/ab_dev_france.log
cd "$(dirname "$0")/.."   # student_resource/
B=.venv/bin/ber; D=code/business_entity_resolution/configs/dev.yaml; P=code/business_entity_resolution/configs/france_proxy.yaml
res() { python3 -c "
import json
d=json.load(open('artifacts/dev/decide/manifest.json'))['metrics']; p=json.load(open('artifacts/synthfr/decide/manifest.json'))['metrics']
print('$1', 'dev', d['val_f05'], {k:v['f05'] for k,v in d['val_by_country'].items()}, 'proxyFR', p['val_f05'], 'proxy P/R', p['val_overall']['pair_p'], p['val_overall']['pair_r'], 'singl', p['val_by_segment']['singleton']['f05'])"; }
rm -rf artifacts/dev/_scorer_B0 && cp -r artifacts/dev/scorer artifacts/dev/_scorer_B0
echo "B0 dev 0.97655 (old) proxyFR 0.88666"
$B features -c $D
$B scorer -c $D -s scorer.oof_folds=1
$B decide -c $D
$B scorer -c $P -s scorer.model_from_run=dev && $B decide -c $P
res B1_newfeat
rm -rf artifacts/dev/_scorer_B1 && cp -r artifacts/dev/scorer artifacts/dev/_scorer_B1
$B scorer -c $D -s scorer.oof_folds=1 -s 'scorer.extra_train=[frtrain]' -s scorer.extra_train_frac=0.39
$B decide -c $D
$B scorer -c $P -s scorer.model_from_run=dev && $B decide -c $P
res B2_newfeat+frtrain
