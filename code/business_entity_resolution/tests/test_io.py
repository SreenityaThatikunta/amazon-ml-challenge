import pytest

from ber.io import MATCHING_HEADER, read_id_lists, read_tsv, write_id_lists


def test_write_id_lists_roundtrip(tmp_path):
    p = tmp_path / "m.tsv"
    n = write_id_lists(p, {"S1-1": ["S2-9", "S3-4", "S2-9"], "S1-3": []},
                       ["S1-1", "S1-2", "S1-3"], MATCHING_HEADER)
    assert n == 3
    assert p.read_text() == ("source1_entity_id\tmatched_entity_ids\n"
                             "S1-1\tS2-9,S3-4\nS1-2\t\nS1-3\t\n")
    assert read_id_lists(p) == {"S1-1": ["S2-9", "S3-4"], "S1-2": [], "S1-3": []}


@pytest.mark.parametrize("lists, keys, msg", [
    ({"S1-1": ["S1-2"]}, ["S1-1"], "non S2/S3"),
    ({"S1-9": ["S2-1"]}, ["S1-1"], "not in all_keys"),
    ({}, ["S1-1", "S1-1"], "duplicate key"),
])
def test_write_id_lists_rules(tmp_path, lists, keys, msg):
    with pytest.raises(ValueError, match=msg):
        write_id_lists(tmp_path / "x.tsv", lists, keys, MATCHING_HEADER)


def test_read_tsv_keeps_quotes_and_empty(tmp_path):
    p = tmp_path / "s.tsv"
    p.write_text('entity_id\tbusiness_name\tbusiness_address\tcountry\n'
                 'S1-1\t"Joe\'s" "Bar\t\tUS\n')
    df = read_tsv(p)
    assert df.business_name[0] == '"Joe\'s" "Bar' and df.business_address[0] == ""


def test_matches_not_in_candidates():
    import numpy as np
    from ber.submit import _matches_not_in_candidates
    cs, cq = np.array([0, 0, 1, 2]), np.array([0, 3, 1, 2])
    assert _matches_not_in_candidates(cs, cq, np.array([0, 1]), np.array([3, 1]), 4) == 0
    assert _matches_not_in_candidates(cs, cq, np.array([0, 1]), np.array([1, 1]), 4) == 1
