# Experiment branch: v3 test-shaped validation and France rehearsal (public LB 0.9805)

> **Archived experiment.** The final solution is on [`main`](https://github.com/SreenityaThatikunta/amazon-ml-challenge/tree/main) (public LB **0.98316**, rank 789).
> The challenge problem statement is in [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md).

v3 ran in parallel with the v2.4–v2.7 work. It merged v2.2 (stage-2 re-ranker, LB 0.980) with the v1 branch's
lookalike features, then went after the train/test shift directly. Every run is logged in
`experiments/README.md` (E19–E32), with scripts and logs under `experiments/`.

## What it tried
- **Merged feature set and test-side pruning** (E19–E25): only test pairs that survive the stage-1 filter are
  written (10.2 M pairs), parallel per-pair features (dev 288 s → 122 s). Full run: **val 0.98954, public LB 0.9805**.
- **The H-test** (E26): unmatched test records turned out to be mostly *copy groups of entities with no S1 row*
  (≈ 66 % in the US). These orphan copies look like real matches to a pairwise model.
- **Test-shaped validation, `val_ts`** (E27–E28): drop S1 rows from val so it has the test's density and orphan
  copies. v3 lost −0.0016 on it (orphan false links); retraining stage 2 on `val_ts` recovered +0.0016 and
  copy-group features added +0.0003.
- **France rehearsal by leave-one-country-out** (E30): pseudo-labels gave no gain and calibration held. The US LOCO
  loss (−0.0023) is the closest analogue to France.
- **France diagnosis without labels** (E31): France swaps a generic word of the name at the same address. A
  copy-slot test says about half of those are true copies, so the model's hesitation there is roughly right.

## Outcome
The plan was stopped on 2026-09-26 (E32) when v2.7 (LB 0.98316) clearly beat v3 (LB 0.9805). The orphan-copy
analysis is the main unexplored lead left over.
