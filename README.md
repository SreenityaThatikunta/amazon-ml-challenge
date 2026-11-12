# Experiment branch: v2 group-aware re-ranker (public LB 0.977 → 0.982344)

> **Archived experiment.** The final solution is on [`main`](https://github.com/SreenityaThatikunta/amazon-ml-challenge/tree/main) (public LB **0.98316**, rank 789).
> The challenge problem statement is in [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md).

v2 rebuilt the pipeline for bounded memory (it ran on a 16 GB laptop) and added a **second-stage re-ranker**
that sees every record competing for the same entity. Full notes: `progress.md`, `Documentation_template.md`.

| Version | Change | Val F0.5 | Public LB | Tag |
|---|---|---:|---:|---|
| v2.1 | stage-2 LightGBM over S1-group context (competing claims, rival scores, source), all 102 stage-1 features | 0.98895 | 0.977 | `v2.1-lb0.977` |
| v2.2 | per-country, per-split rescaling of size-dependent stage-2 features (test is denser than train) | 0.98897 | 0.980 | `v2.2-lb0.980` |
| v2.3 | label-free anchor correction for France's generic-word name swaps, estimated from house numbers | 0.98897 | — | |
| v2.4 | stage 2 trained on val + 20 % held-out train regions | 0.98911 | 0.980973 | `v2.4-lb0.980973` |
| v2.5 | 40 % held-out regions by owner, stage 2 trained on complete groups only | 0.98923 | 0.981708 | `v2.5-lb0.981708` |
| v2.6 | normalisation scale of S1-group counts taken from complete groups | 0.98918 | **0.982344** | `v2.6-lb0.982344` |

Other v2 work: filler-free name keys and IDF overlap, house-number relation features, a made-up-brand detector,
exact expected-F0.5 via Poisson-binomial, optional CuPy GPU top-k retrieval (27.9 s → 2.6 s on a large block).

**Key lesson:** v2.2 changed val by +0.00002 but the leaderboard by +0.003. Handling the train/test density shift
mattered more than further validation gains.

**What came next:** [`experiment/v2.7-cross-encoder`](https://github.com/SreenityaThatikunta/amazon-ml-challenge/tree/experiment/v2.7-cross-encoder) added a
fine-tuned MiniLM cross-encoder on top of v2.6 and became the final submission (LB 0.98316).
