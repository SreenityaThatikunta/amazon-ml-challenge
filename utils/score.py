#!/usr/bin/env python3
"""
ML Challenge 2026 — Local F_0.5 Scorer

Computes the leaderboard metric: F_beta (beta = 0.5), calculated per Source 1
entity and macro-averaged over every Source 1 entity in the evaluation set.

Per-entity rules (as described in README.md):
  * truth empty,  prediction empty      -> 1.0
  * truth empty,  prediction non-empty  -> 0.0
  * truth non-empty, prediction empty   -> 0.0
  * otherwise F_0.5 = (1.25 * P * R) / (0.25 * P + R)   (0.0 when P = R = 0)

Stdlib only, Python 3.8+. Run from the ``student_resource/`` directory::

    python3 utils/score.py \
        --pred output/val_matching_results.tsv \
        --truth dataset/train/train_ground_truth.tsv

By default the evaluation set is every S1 entity in ``--truth``; S1 entities
missing from ``--pred`` count as empty predictions. For a validation split,
either pass a truth file containing only the held-out S1 rows, or pass
``--only-predicted`` to score just the S1 entities that appear in ``--pred``.

Pass ``--source1`` (the matching *_source1.tsv) to also get a per-country
breakdown.
"""

import argparse
import sys
from collections import defaultdict

DELIM = "\t"
BETA2 = 0.25  # beta = 0.5


def read_id_lists(path):
    """Return {source1_entity_id: set(matched ids)} from a 2-column TSV."""
    out = {}
    with open(path, encoding="utf-8") as f:
        next(f, None)  # skip header
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            s1, _, ids = line.partition(DELIM)
            out[s1.strip()] = {i.strip() for i in ids.split(",") if i.strip()}
    return out


def read_countries(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.rstrip("\n").split(DELIM)
            if len(parts) >= 4:
                out[parts[0].strip()] = parts[3].strip()
    return out


def f_half(truth, pred):
    """Return (f05, precision, recall) for one Source 1 entity."""
    if not truth:
        return (1.0, 1.0, 1.0) if not pred else (0.0, 0.0, 1.0)
    if not pred:
        return 0.0, 0.0, 0.0
    tp = len(truth & pred)
    if tp == 0:
        return 0.0, 0.0, 0.0
    p = tp / len(pred)
    r = tp / len(truth)
    return (1 + BETA2) * p * r / (BETA2 * p + r), p, r


def score(truth, pred, only_predicted=False, countries=None):
    ids = [s for s in truth if s in pred] if only_predicted else list(truth)
    if not ids:
        sys.exit("No Source 1 entities to score (check --pred / --truth).")

    groups = defaultdict(lambda: [0.0, 0.0, 0.0, 0])  # f, p, r, n
    tp = n_pred = n_true = 0
    single_n = single_ok = 0
    for s1 in ids:
        t = truth[s1]
        p = pred.get(s1, set())
        f, prec, rec = f_half(t, p)
        keys = ["ALL"]
        if countries is not None:
            keys.append(countries.get(s1, "UNKNOWN"))
        keys.append("singletons" if not t else "non-singletons")
        for k in keys:
            g = groups[k]
            g[0] += f
            g[1] += prec
            g[2] += rec
            g[3] += 1
        tp += len(t & p)
        n_pred += len(p)
        n_true += len(t)
        if not t:
            single_n += 1
            single_ok += not p

    extra = sorted(set(pred) - set(truth))
    return groups, tp, n_pred, n_true, single_n, single_ok, extra


def main():
    ap = argparse.ArgumentParser(description="Local macro F_0.5 scorer.")
    ap.add_argument("--pred", required=True, help="matching_results.tsv to score")
    ap.add_argument("--truth", required=True, help="ground-truth TSV (same format)")
    ap.add_argument("--only-predicted", action="store_true",
                    help="score only S1 entities present in --pred")
    ap.add_argument("--source1", help="optional *_source1.tsv for per-country scores")
    args = ap.parse_args()

    truth = read_id_lists(args.truth)
    pred = read_id_lists(args.pred)
    countries = read_countries(args.source1) if args.source1 else None

    groups, tp, n_pred, n_true, single_n, single_ok, extra = score(
        truth, pred, args.only_predicted, countries)

    print(f"{'group':<16}{'n':>10}{'F0.5':>9}{'P':>9}{'R':>9}")
    order = ["ALL"] + sorted(k for k in groups if k not in
                             ("ALL", "singletons", "non-singletons")) + \
        ["non-singletons", "singletons"]
    for k in order:
        if k in groups:
            f, p, r, n = groups[k]
            print(f"{k:<16}{n:>10}{f / n:>9.4f}{p / n:>9.4f}{r / n:>9.4f}")

    micro_p = tp / n_pred if n_pred else 0.0
    micro_r = tp / n_true if n_true else 0.0
    print(f"\nmicro pairs: TP={tp} pred={n_pred} true={n_true} "
          f"P={micro_p:.4f} R={micro_r:.4f}")
    if single_n:
        print(f"singletons correctly empty: {single_ok}/{single_n} "
              f"({single_ok / single_n:.4f})")
    if extra:
        print(f"warning: {len(extra)} S1 ids in --pred not in --truth "
              f"(ignored), e.g. {extra[:3]}")
    if not args.only_predicted:
        missing = sum(1 for s in truth if s not in pred)
        if missing:
            print(f"note: {missing} truth S1 ids missing from --pred "
                  f"(scored as empty predictions)")

    print(f"\nMACRO F0.5 = {groups['ALL'][0] / groups['ALL'][3]:.6f}")


if __name__ == "__main__":
    main()
