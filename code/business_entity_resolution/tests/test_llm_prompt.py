import pandas as pd
import pytest

from ber.llm.export import route, top_candidates
from ber.llm.prompt import build_prompt, hints


def _feat(**kw):
    base = dict(n_core_eq=0, q_is_domain=0, n_q_is_initials=0, n_s_is_initials=0, n_lev=1,
                n_sub=1, n_transp=0, n_tsr=94.0, l_eq=1, l_conflict=0, hn_rel=1, num_q_only=0,
                box_q=0, box_s=0, r_eq=1, r_conflict=0, a_tsr=100.0, p=0.62)
    base.update(kw)
    return pd.Series(base)


def test_hints_describe_distractor_signals():
    h = hints(_feat())
    assert "1 letter(s) replaced" in h and "house no. equal" in h and "scorer 0.62" in h
    h = hints(_feat(n_core_eq=1, hn_rel=4, num_q_only=1, box_q=1))
    assert "same name" in h and "house no. shifted" in h and "PO box" in h


def test_prompt_lists_candidates_and_answer_range():
    q = pd.Series({"name_raw": "Verrufy America LLC", "addr_raw": "3517 URBAN AVE, DALLAS, TX",
                   "country": "US", "source": 2})
    c = pd.DataFrame([dict(_feat(), name_raw="Verrify America LLC",
                           addr_raw="3517 Urban Avenue, Dallas, TX"),
                      dict(_feat(n_sub=0, n_lev=8, n_tsr=40.0), name_raw="Other Co",
                           addr_raw="")])
    p = build_prompt(q, c)
    assert "1. name: Verrify America LLC" in p and "2. name: Other Co" in p
    assert "address: (empty)" in p and p.rstrip().endswith("Answer (0-2):")


def test_route_and_top_candidates():
    s = pd.DataFrame({"q_id": ["a", "a", "b", "c", "c"], "s1_id": ["X", "Y", "X", "Z", "W"],
                      "p": [0.99, 0.2, 0.5, 0.01, 0.03]})
    r = route(s, 0.05, 0.95)
    assert r.to_dict() == {"a": False, "b": True, "c": False}
    t = top_candidates(s, 1, 0)
    assert dict(zip(t.q_id, t.s1_id)) == {"a": "X", "b": "X", "c": "W"}


def test_blend_only_touches_llm_queries():
    from ber.llm.import_scores import blend
    scores = pd.DataFrame({"q_id": ["a", "a", "b"], "s1_id": ["X", "Y", "X"],
                           "p": [0.6, 0.4, 0.9]})
    llm = pd.DataFrame({"q_id": ["a"], "cands": [["Y", "X"]], "probs": [[0.1, 0.8, 0.1]]})
    out = blend(scores, llm, 0.5).set_index(["q_id", "s1_id"]).p
    assert out[("a", "Y")] == pytest.approx(0.6) and out[("a", "X")] == pytest.approx(0.35)
    assert out[("b", "X")] == pytest.approx(0.9)
