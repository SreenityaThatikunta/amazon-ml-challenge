import numpy as np

from ber.rerank.stage import group_features


def test_group_features_query_and_s1_context():
    # queries 0,1 (source 2) and 2 (source 3); S1 10 and 11
    q = np.array([0, 0, 1, 2])
    s = np.array([10, 11, 10, 10])
    src = np.array([2, 2, 2, 3])
    p = np.array([0.9, 0.2, 0.6, 0.7])
    f = group_features(q, s, src, p)
    # query context
    assert list(f.q_rank) == [0, 1, 0, 0]
    np.testing.assert_allclose(f.q_best_other, [0.2, 0.9, -1, -1], atol=1e-6)
    np.testing.assert_allclose(f.q_margin, [0.7, -0.7, 0.6, 0.7], atol=1e-6)
    # S1 10 is listed by queries 0 (0.9), 1 (0.6), 2 (0.7)
    assert list(f.s_rank) == [0, 0, 2, 1]
    np.testing.assert_allclose(f.s_sum_other, [1.3, 0.0, 1.6, 1.5], atol=1e-6)
    np.testing.assert_allclose(f.s_n_hi_other, [2, 0, 2, 2])
    # same source (2) only: queries 0 and 1 compete on S1 10; query 2 is alone in source 3
    np.testing.assert_allclose(f.ss_best_other, [0.6, -1, 0.9, -1], atol=1e-6)
    # claims: every query's best S1 is 10 -> S1 10 has 3 claims, S1 11 none
    np.testing.assert_allclose(f.c_expected, [2.2, 0.0, 2.2, 2.2], atol=1e-6)
    assert f.dtypes.map(lambda d: d == np.float32).all()
