import numpy as np

from ber.rerank.copies import COLS, copy_features, vectors


def test_orphan_group_closer_to_itself_than_to_twin():
    s1 = ["acme plumbing 12 main st springfield", "zeta dental 5 oak rd shelby"]
    q = ["acme plumbing 1400 elm st springfield",      # copies of a *missing* S1 ...
         "acme plumbng 1400 elm st springfield",       # ... both claiming the twin s1[0]
         "zeta dentl 5 oak rd shelby",                 # real copies of s1[1]
         "zeta dental 5 oak road shelbi"]
    Qv, Sv = vectors(q), vectors(s1)
    qi = np.array([0, 1, 2, 3])
    si = np.array([0, 0, 1, 1])
    p = np.array([0.6, 0.5, 0.9, 0.8])
    f = copy_features(qi, si, p, Qv, Sv)
    assert set(f) == set(COLS)
    orphan, real = f["cp_sib_gap"][:2], f["cp_sib_gap"][2:]
    assert orphan.min() > 0.1 and orphan.min() > real.max() + 0.1     # sibling >> twin
    assert f["cp_sib_p"][0] == np.float32(0.5) and f["cp_sib_top_s"][0] == 1
    assert f["cp_sib_n08"][0] >= 1


def test_single_claimant_and_low_p():
    Qv, Sv = vectors(["a b c d"]), vectors(["a b c d"])
    f = copy_features(np.array([0]), np.array([0]), np.array([0.001]), Qv, Sv)
    assert f["cp_sib_max"][0] == 0 and abs(f["cp_sib_gap"][0] + f["cp_cos_qs"][0]) < 1e-6
