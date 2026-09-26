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


def test_global_codes_dictionary_and_plain():
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    from ber.submit import _global_codes
    index = pd.Index(["S2-1", "S2-2", "S3-9"])
    d = pa.chunked_array([pa.array(["S3-9", "S2-1", "S3-9", "S2-7"]).dictionary_encode()])
    assert _global_codes(d, index).tolist() == [2, 0, 2, -1]
    assert _global_codes(pa.chunked_array([pa.array(["S2-2"])]), index).tolist() == [1]
