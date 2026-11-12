# Experiment branch: initial prototype (public LB 0.961)

> **Archived experiment.** The final solution is on [`main`](https://github.com/SreenityaThatikunta/amazon-ml-challenge/tree/main) (public LB **0.98316**, rank 789).
> The challenge problem statement is in [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md).

This is the first end-to-end pipeline, written by Rudraksha Dev Sharma as a quick, independent baseline.
The modules are at the repository root:

| File | Role |
|---|---|
| `prepare.py`, `normalize.py`, `resources.py` | load the TSVs, normalise names and addresses |
| `blocking.py` | candidate generation (blocking keys) |
| `features.py`, `model.py` | pairwise string features and a gradient-boosted pair classifier |
| `pipeline.py`, `metrics.py` | end-to-end run and the macro F0.5 metric |

**Result:** public leaderboard F0.5 **0.961**.

**Why it was superseded:** blocking recall capped the score and pairs were judged in isolation.
The [`experiment/v1-lightgbm-baseline`](https://github.com/SreenityaThatikunta/amazon-ml-challenge/tree/experiment/v1-lightgbm-baseline) pipeline
replaced it with region-blocked TF-IDF retrieval (99 % pair recall) and an expected-F0.5 decision layer (LB 0.975).
