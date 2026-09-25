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
