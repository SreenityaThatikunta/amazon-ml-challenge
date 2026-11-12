# Experiment branch: v1 LightGBM baseline and the "main v2" follow-ups (public LB 0.975 / 0.972)

> **Archived experiment.** The final solution is on [`main`](https://github.com/SreenityaThatikunta/amazon-ml-challenge/tree/main) (public LB **0.98316**, rank 789).
> The challenge problem statement is in [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md).

This branch holds the first full-scale version of the `ber` pipeline (`code/business_entity_resolution/`),
the EDA (`eda/`) and the experiment log E1–E18 (`experiments/README.md`, `progress.md`, `plan.md`).

## What it built (v1, tag `v1-lb0.975`)
- Geographic train/val split (whole states held out) and a local scorer identical to the official metric.
- Text and address normalisation, including a learned Indic-script → Latin dictionary (95.8 % exact-token match on val).
- Region-blocked character-trigram TF-IDF retrieval: **99.04 % pair recall** at ~24 candidates per record.
- ~70 pair features → LightGBM pair scorer → expected-F0.5 subset selection per S1 entity.
- **Val macro F0.5 0.98211, public LB 0.975.**

## What it tried next ("main v2", E10–E18)
- Lookalike features (house number shifted by a few units, same street), French legal-form conflicts.
- A synthetic-French training set and an "unseen-country" scorer for France (no French labels exist).
- A cluster-context rescore (second LightGBM over each entity's confident cluster).
- **Val rose to 0.98906, but the public LB fell to 0.972.**

## What we learned
The test set is denser than training (≈ 5.6 vs 4.7 S2/S3 records per S1, so many more unmatched records),
and validation could not see that shift. Gains that ignored it did not transfer to the leaderboard. This finding
shaped [`experiment/v2-group-reranker`](https://github.com/SreenityaThatikunta/amazon-ml-challenge/tree/experiment/v2-group-reranker), which normalised
density-dependent features per country and split.
