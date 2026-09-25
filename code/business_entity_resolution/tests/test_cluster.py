import numpy as np
import pandas as pd

from ber.cluster.stage import build, cluster_features, top_k


def _pairs():
    # S1-a: confident members S2-1 (0.9), S3-1 (0.8); S2-2 is weak (0.3) with a runner-up
    s = pd.DataFrame({
        "q_id": ["S2-1", "S3-1", "S2-2", "S2-2", "S3-2"],
        "s1_id": ["S1-a", "S1-a", "S1-a", "S1-b", "S1-b"],
        "p": [0.9, 0.8, 0.3, 0.1, 0.95]})
    return top_k(s, 3)


def test_top_k_margins():
    t = _pairs().set_index(["q_id", "s1_id"])
    assert t.loc[("S2-2", "S1-a"), "rank"] == 0
    assert np.isclose(t.loc[("S2-2", "S1-a"), "p_margin"], 0.2)
    assert np.isclose(t.loc[("S2-2", "S1-b"), "p_margin"], -0.2)
    assert t.loc[("S2-1", "S1-a"), "q_p2"] == 0.0


def test_cluster_features_exclude_self():
    p = _pairs()
    f, members = cluster_features(p, conf=0.5)
    f.index = pd.MultiIndex.from_frame(p[["q_id", "s1_id"]])
    # S2-1 is itself a confident member of S1-a: its cluster is only S3-1
    assert f.loc[("S2-1", "S1-a"), "c_n"] == 1
    assert np.isclose(f.loc[("S2-1", "S1-a"), "c_sum"], 0.8)
    assert f.loc[("S2-1", "S1-a"), "c_same_src"] == 0 and f.loc[("S2-1", "S1-a"), "c_cap_room"] == 5
    # the weak S2-2 sees both confident members; one S2 already there
    assert f.loc[("S2-2", "S1-a"), "c_n"] == 2 and f.loc[("S2-2", "S1-a"), "c_same_src"] == 1
    # members assigned to S1-a scoring strictly above each pair
    assert f.loc[("S2-1", "S1-a"), "s_n_above"] == 0
    assert f.loc[("S3-1", "S1-a"), "s_n_above"] == 1
    assert f.loc[("S2-2", "S1-a"), "s_n_above"] == 2
    assert f.loc[("S2-2", "S1-b"), "s_n_above"] == 1   # S3-2 (0.95) is assigned to S1-b
    assert set(members.q_id) == {"S2-1", "S3-1", "S3-2"}


def test_build_member_similarity():
    p = _pairs()
    rec = pd.DataFrame({"name_core": ["lille club", "lille club", "lille club", "x"],
                        "addr_norm": ["8 rue gassendi lille"] * 3 + ["y"]},
                       index=["S2-1", "S3-1", "S2-2", "S3-2"])
    X = build(p, rec, conf=0.5)
    X.index = pd.MultiIndex.from_frame(p[["q_id", "s1_id"]])
    assert X.loc[("S2-2", "S1-a"), "m_dup"] == 1 and X.loc[("S2-2", "S1-a"), "m_name_max"] == 100
    assert X.loc[("S2-2", "S1-b"), "m_name_max"] < 50
    assert X.loc[("S3-2", "S1-b"), "m_n"] == 0            # no other member to compare with


def test_predict_parts_matches_direct(tmp_path):
    """Part-by-part test scoring == scoring the fully joined frame."""
    import lightgbm as lgb
    from ber.cluster.stage import predict_parts
    rng = np.random.default_rng(0)
    parts, frames = [], []
    for i in range(3):
        f = pd.DataFrame({"a": rng.random(50).astype(np.float32),
                          "b": rng.random(50).astype(np.float32)})
        f.to_parquet(tmp_path / f"part-{i}.parquet")
        parts.append(tmp_path / f"part-{i}.parquet")
        frames.append(f)
    # 40 pairs pointing at random rows of random parts, shuffled
    part = rng.integers(0, 3, 40).astype(np.int16)
    row = rng.integers(0, 50, 40).astype(np.int32)
    pairs = pd.DataFrame({"_part": part, "_row": row, "p": rng.random(40).astype(np.float32)})
    ctx = pd.DataFrame({"c": rng.random(40).astype(np.float32)})
    full = pd.concat([ctx, pd.DataFrame([frames[q].iloc[r] for q, r in zip(part, row)]
                                        ).reset_index(drop=True)], axis=1)
    y = (full.a + full.c > 1).astype(int)
    mdl = lgb.train({"objective": "binary", "verbose": -1, "min_data_in_leaf": 2},
                    lgb.Dataset(full[["c", "b", "a"]], y), num_boost_round=5)
    mask = rng.random(40) < 0.7
    got = predict_parts(mdl, pairs, ctx, parts, mask)
    want = np.where(mask, mdl.predict(full[["c", "b", "a"]]), pairs.p)
    np.testing.assert_allclose(got, want, rtol=1e-5)
