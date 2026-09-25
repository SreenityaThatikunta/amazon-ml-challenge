import numpy as np
import pandas as pd
import pytest

from ber.features.build import _best_other, build_features
from ber.features.numbers import (DIFFERENT, EQUAL, MISSING, PADDED, SHIFTED, TRUNCATED,
                                  number_features, relation)
from ber.features.stage import _query_chunks, record_extras


@pytest.mark.parametrize("a, b, rel", [
    ("7622", "7622", EQUAL), ("2877", "002877", PADDED),
    ("7622", "762", TRUNCATED), ("6665", "665", TRUNCATED), ("5123", "512", TRUNCATED),
    ("805", "812", SHIFTED), ("5835", "5838", SHIFTED), ("11911", "11913", SHIFTED),
    ("16445", "16449", SHIFTED), ("805", "3120", DIFFERENT), ("", "12", MISSING),
])
def test_house_number_relation(a, b, rel):  # examples from eda/FINDINGS.md §3
    assert relation(a, b) == rel


def test_number_features():
    rel, diff, common, q_only, s_only, jacc = number_features("41 521", "41")
    assert rel == EQUAL and diff == 0 and (common, q_only, s_only) == (1, 1, 0) and jacc == 0.5
    assert number_features("", "")[5] == -1.0


def test_best_other():
    q = np.array([0, 0, 0, 1])
    v = np.array([0.2, 0.9, 0.5, 0.3])
    best, other, rank = _best_other(q, v)
    assert best.tolist() == [0.9, 0.9, 0.9, 0.3]
    assert other[:3].tolist() == [0.9, 0.5, 0.9] and np.isnan(other[3])
    assert rank.tolist() == [2, 0, 1, 0]


def test_query_chunks_keep_queries_whole():
    part = pd.DataFrame({"q_id": ["a", "a", "b", "b", "b", "c"], "x": range(6)})
    chunks = list(_query_chunks(part, 2))
    assert [c.q_id.tolist() for c in chunks] == [["a", "a"], ["b", "b", "b"], ["c"]]


def _rec(eid, src, name_core, addr, nums, region="MD", legal="", compact=None):
    return {"entity_id": eid, "source": src, "country": "US", "name_core": name_core,
            "name_compact": compact or name_core.replace(" ", ""), "name_legal": legal,
            "name_pre_core": "", "name_alt_core": "", "name_is_domain": False,
            "name_has_alias": False, "name_indic": False, "name_n_indic_oov": 0,
            "addr_norm": addr, "addr_bag": " ".join(sorted(set(addr.split()))),
            "region": region, "addr_numbers": nums, "addr_has_box": False}


def test_build_features_smoke_and_signals():
    from ber.features.stage import join_records
    recs = pd.DataFrame([
        _rec("S1-a", 1, "verrify america", "3517 urban ave dallas", "3517", legal="llc"),
        _rec("S1-b", 1, "vision health", "7622 arbory ct laurel", "7622"),
        _rec("S2-1", 2, "verrufy america", "3517 urban ave dallas", "3517", legal="llc"),
        _rec("S2-2", 2, "vision health", "762 arbory ct laurel", "762"),
    ])
    rec = record_extras(recs).set_index("entity_id")
    cands = pd.DataFrame({"q_id": ["S2-1", "S2-1", "S2-2", "S2-2"],
                          "s1_id": ["S1-a", "S1-b", "S1-b", "S1-a"],
                          "cos_name": [.8, .1, 1., .1], "cos_addr": [1., .2, .8, .2],
                          "cos_combo": [.9, .1, .9, .1], "rank_name": [0, 1, 0, 1],
                          "rank_addr": [0, 1, 0, 1], "rank_combo": [0, 1, 0, 1]})
    ctx = pd.DataFrame({"s_n_cands": [2, 2], "s_n_top1": [1, 1]}, index=["S1-a", "S1-b"])
    d = join_records(cands, rec, ctx)
    f = build_features(d)
    assert len(f) == 4 and not f.isna().all().any()
    assert f.loc[0, "n_sub"] == 1 and f.loc[0, "a_bag_eq"] == 1       # one-letter distractor edit
    assert f.loc[2, "hn_rel"] == TRUNCATED and f.loc[2, "n_core_eq"] == 1
    assert f.loc[0, "cos_combo_margin"] == pytest.approx(0.8)
    assert f.loc[0, "l_eq"] == 1 and f.loc[2, "l_q_empty"] == 1
