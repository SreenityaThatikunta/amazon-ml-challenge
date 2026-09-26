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


@pytest.mark.parametrize("q, s, expected", [
    ("937 33", "33", (1, 0, EQUAL, 2, -1)),        # fake number prepended (India)
    ("9310", "177 9310", (0, 1, DIFFERENT, 4, -1)),  # unit dropped + components reordered
    ("894", "394", (0, 0, DIFFERENT, 1, 2)),        # leading-digit typo (true pair)
    ("5838", "5835", (0, 0, SHIFTED, 1, 0)),        # last-digit shift (distractor)
    ("", "12", (-1, -1, MISSING, -1, -1)),
])
def test_number_features_v2(q, s, expected):
    from ber.features.numbers import number_features_v2
    assert number_features_v2(q, s) == expected


def test_fillers_and_keys():
    from ber.features.build import KeyIndex, key_features
    from ber.features.stage import learn_fillers, name_key, token_idf
    s1 = [f"name{i} dental" for i in range(20)]
    q = [f"name{i} dental" for i in range(20)] + [f"name{i} dental riverside" for i in range(10)]
    recs = pd.DataFrame({"source": [1] * 20 + [2] * 30, "country": "US", "name_core": s1 + q})
    assert learn_fillers([recs], 0.05, 2.0) == ["riverside"]
    keys = name_key(pd.Series(["dental name1 riverside", "name1 dental", "name2 plumbing"]),
                    frozenset(["riverside"]))
    assert keys.tolist() == ["dental name1", "dental name1", "name2 plumbing"]
    kidx = KeyIndex(keys, token_idf(keys[1:]))
    d = pd.DataFrame({"q_name_key": [keys[0], keys[0]], "s_name_key": [keys[1], keys[2]],
                      "q_kpos": [0, 0], "s_kpos": [1, 2]})
    f = key_features(d, kidx)
    assert f.k_eq.tolist() == [1, 0] and f.k_idf_rec.tolist() == [1.0, 0.0]
    assert f.k_idf_unmatched_s[0] == 0 and f.k_idf_unmatched_s[1] > 0


def _fr(eid, src, name_core, addr, nums, legal=""):
    r = _rec(eid, src, name_core, addr, nums, region="HDF", legal=legal)
    r["country"] = "France"
    return r


def test_lookalike_features_french_twins():
    """eda/reports/07: the query matches S1-a; S1-b is a twin at another number,
    S1-c a twin on another street, S1-d another legal form."""
    from ber.features.stage import join_records
    recs = pd.DataFrame([
        _fr("S1-a", 1, "lille amicale", "13 rue du vaisseau le vengeur lille", "13", legal="sarl"),
        _fr("S1-b", 1, "lille amicale", "3 rue du vaisseau le vengeur lille", "3", legal="sarl"),
        _fr("S1-c", 1, "lille amicale", "13 rue du chateau lille", "13", legal="sarl"),
        _fr("S1-d", 1, "lille amicale", "13 rue du vaisseau le vengeur lille", "13", legal="sas"),
        _fr("S1-e", 1, "lille ecole", "13 rue du vaisseau le vengeur lille", "13", legal="sarl"),
        _fr("S2-1", 2, "lille amicale", "13 rue du vaiseau le vengeur lille", "13", legal="sarl"),
    ])
    rec = record_extras(recs).set_index("entity_id")
    s1 = ["S1-a", "S1-b", "S1-c", "S1-d", "S1-e"]
    cands = pd.DataFrame({"q_id": ["S2-1"] * 5, "s1_id": s1, "cos_name": [1.] * 5,
                          "cos_addr": [1., .9, .6, 1., 1.], "cos_combo": [1., .9, .7, 1., .8],
                          "rank_name": range(5), "rank_addr": range(5), "rank_combo": range(5)})
    ctx = pd.DataFrame({"s_n_cands": [1] * 5, "s_n_top1": [1] * 5}, index=s1)
    f = build_features(join_records(cands, rec, ctx)).set_index(pd.Index(s1))
    assert f.loc["S1-a", "a_q_unmatched"] == 0             # "vaiseau" typo still matches
    assert f.loc["S1-c", "a_word_conflict"] == 1           # vaisseau vengeur vs chateau
    assert f.loc["S1-b", "hn_len_min"] == 1 and f.loc["S1-b", "hn_trunc_lead"] == 1
    assert f.loc["S1-b", "hn_quality_margin"] < 0          # another candidate has the exact number
    assert f.loc["S1-d", "l_quality_margin"] < 0           # another candidate has the same legal form
    assert f.loc["S1-a", "l_quality_margin"] == 0 and f.loc["S1-a", "hn_quality_margin"] == 0
    assert f.loc["S1-e", "n_nocity_ratio"] < f.loc["S1-e", "n_ratio"] - 10  # city word removed
    assert f.loc["S1-e", "n_shared_only_city"] == 1
    assert f.loc["S1-a", "n_s_city_frac"] == 0.5


def test_unmatched_word_threshold_inclusive():
    from ber.features.build import _unmatched
    assert _unmatched(["nagar"], ["naga", "urban"]) == 0      # similarity exactly 0.8 counts
    assert _unmatched(["chateau"], ["magasin"]) == 1
    assert _unmatched(["x"], []) == 1


def test_test_pruner_keeps_high_p_rows_unchanged(tmp_path):
    import lightgbm as lgb
    from ber.features.stage import TestPruner
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 2)).astype(np.float32)
    y = (x[:, 0] > 0).astype(int)
    lgb.train({"objective": "binary", "verbose": -1}, lgb.Dataset(x, y, feature_name=["a", "b"]),
              num_boost_round=5).save_model(str(tmp_path / "m.txt"))
    out = pd.DataFrame({"q_id": [f"S2-{i}" for i in range(400)], "a": x[:, 0], "b": x[:, 1]})
    prune = TestPruner(tmp_path / "m.txt", 0.5)
    kept = prune(out)
    p = prune.model.predict(x)
    assert len(kept) == int((p >= 0.5).sum()) and 0 < len(kept) < 400
    pd.testing.assert_frame_equal(kept, out[p >= 0.5].reset_index(drop=True))
    assert (prune.n_in, prune.n_out) == (400, len(kept))


def test_prune_test_refuses_single_pass():
    from ber.artifacts import StageError
    from ber.config import load_config
    from ber.features.stage import run_features
    cfg = load_config(overrides=["features.prune_test=true", "data.phase=all"])
    if not (cfg.run_test and cfg.run_train):
        pytest.skip("default config has no test split here")
    with pytest.raises(StageError, match="data.phase=train"):
        run_features(cfg)


def test_parallel_rows_match_serial():
    from concurrent.futures import ProcessPoolExecutor
    from ber.features.stage import join_records, parallel_rows
    recs = pd.DataFrame([_rec(f"S1-{i}", 1, f"name {i % 7}", f"{i} main st town", str(i))
                         for i in range(40)] +
                        [_rec(f"S2-{i}", 2, f"name {i % 5}", f"{i} main st town", str(i))
                         for i in range(40)])
    rec = record_extras(recs).set_index("entity_id")
    q = [f"S2-{i}" for i in range(40) for _ in range(3)] * 200
    s = [f"S1-{(i + j) % 40}" for i in range(40) for j in range(3)] * 200
    cands = pd.DataFrame({"q_id": q, "s1_id": s, "cos_name": 0.5, "cos_addr": 0.5,
                          "cos_combo": 0.5, "rank_name": 0, "rank_addr": 0, "rank_combo": 0})
    ctx = pd.DataFrame({"s_n_cands": 1, "s_n_top1": 1}, index=sorted(set(s)))
    d = join_records(cands, rec, ctx)
    assert len(d) >= 20_000
    with ProcessPoolExecutor(3) as pool:
        par = build_features(d, rows=parallel_rows(pool, d, 3))
    pd.testing.assert_frame_equal(build_features(d), par)
