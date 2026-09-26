import numpy as np
import pandas as pd
import pytest

from ber.decide.expected_f import best_prefix, select
from ber.decide.stage import decide, one_owner


def test_best_prefix_prefers_empty_when_unlikely():
    assert best_prefix(np.array([0.1]))[0] == 0          # P(no match)=0.9 beats F of keeping
    assert best_prefix(np.array([0.95, 0.9, 0.2]))[0] == 2
    assert best_prefix(np.array([]))[0] == 0


def test_best_prefix_value_matches_certain_case():
    k, ef = best_prefix(np.array([1.0, 1.0]))
    assert k == 2 and ef == pytest.approx(1.0)


def test_one_owner_and_select():
    scores = pd.DataFrame({"q_id": ["a", "a", "b", "c"], "s1_id": ["X", "Y", "X", "Z"],
                           "p": [0.9, 0.95, 0.8, 0.05]})
    a = one_owner(scores)
    assert dict(zip(a.q_id, a.s1_id)) == {"a": "Y", "b": "X", "c": "Z"}
    kept = select(a)
    assert set(kept.q_id) == {"a", "b"}
    assert set(decide(a, "threshold", 0.5, 0, 1).q_id) == {"a", "b"}
    with pytest.raises(ValueError):
        decide(a, "magic", 0, 0, 1)


def test_tie_duplicates():
    from ber.decide.stage import dup_keys, tie_duplicates
    recs = pd.DataFrame({"entity_id": ["S2-1", "S2-2", "S2-3"], "source": [2, 2, 2],
                         "country": "US", "name_raw": ["Acme", "Acme", "Other"],
                         "addr_raw": ["1 Main St", "1 Main St", "1 Main St"]})
    a = pd.DataFrame({"q_id": ["S2-1", "S2-2", "S2-3"], "s1_id": ["X", "Y", "Z"],
                      "p": [0.4, 0.9, 0.5]})
    t = tie_duplicates(a, dup_keys(recs))
    assert t.s1_id.tolist() == ["Y", "Y", "Z"] and t.p.tolist() == [0.9, 0.9, 0.5]


def test_prior_shift_em_recovers_prior_and_rescales_segment():
    from ber.decide.prior_shift import adjust, em_ratio, segment_priors
    rng = np.random.default_rng(0)
    n = 50000
    y = rng.random(n) < 0.4
    # perfectly calibrated scores: p = P(y | bin) with two bins
    p = np.where(rng.random(n) < 0.5, 0.8, 0.2)
    y = rng.random(n) < p
    keep = ~y | (rng.random(n) < 0.5)            # test: half of the positives gone
    r, pi = em_ratio(p[keep], float(p.mean()))
    assert abs(pi - y[keep].mean()) < 0.01 and r < 1
    seg = dict(hn_rel=4, l_conflict=0, k_eq=1, a_tsr=100.0)
    val = pd.DataFrame({"p": p, **seg})
    test = pd.DataFrame({"p": p[keep], **seg})
    adj, rep = adjust(test, segment_priors(val), min_pairs=10)
    assert list(rep) == ["4|0|1"] and (adj.p <= test.p + 1e-6).all()
