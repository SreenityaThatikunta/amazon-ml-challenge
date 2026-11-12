# Experiment branch: v2.7 cross-encoder, full development history (public LB 0.98316, final)

> This is the **raw development history** of the final submission, including working notes
> (`progress.md`, `context.md`, `plan.md`). The cleaned-up version of the same code is on
> [`main`](https://github.com/SreenityaThatikunta/amazon-ml-challenge/tree/main). The challenge problem statement is in [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md).

v2.7 = v2.6 ([`experiment/v2-group-reranker`](https://github.com/SreenityaThatikunta/amazon-ml-challenge/tree/experiment/v2-group-reranker)) plus a
**MiniLM-L6 cross-encoder** (22.7 M parameters, Apache-2.0) fine-tuned only on pairs the stage-1 scorer never
trained on. On its own it is a weaker matcher than LightGBM, but its logit margin and rank over the competing
candidates are strong evidence for the stage-2 re-ranker.

| | Val F0.5 | Public LB |
|---|---:|---:|
| v2.6 (no cross-encoder) | 0.98918 | 0.982344 |
| **v2.7 (cross-encoder margins → stage 2)** | **0.99078** | **0.98316** (rank 789) |

The later commits on this branch trim the code into the organisers' submission-package layout and sync it with
the zip the team submitted (tag `v2.7-lb0.98316`).
