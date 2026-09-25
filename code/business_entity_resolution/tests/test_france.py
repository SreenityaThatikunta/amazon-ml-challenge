import random

import numpy as np
import pandas as pd

from ber.france.synth import generate, lookalike, noisy_address, noisy_name
from ber.normalize.address import AddressNormalizer
from ber.normalize.text import NameNormalizer


def test_noisy_variants_normalise_close_to_source():
    rng = random.Random(0)
    N, A = NameNormalizer(), AddressNormalizer()
    name, addr = "Honore Loisirs SARL", "16 Cours François Bart, Dunkerque, Hauts-de-France"
    same_core = sum(N(noisy_name(name, 2, rng)).core == N(name).core for _ in range(300))
    assert same_core > 150  # most copies keep the core name after normalisation
    regions = {A(noisy_address(addr, 3, rng), "France").region for _ in range(200)}
    assert regions <= {"HDF", ""} and "HDF" in regions


def test_lookalike_differs():
    rng = random.Random(1)
    name, addr = "Honore Loisirs SARL", "16 Cours François Bart, Dunkerque, Hauts-de-France"
    for _ in range(50):
        n, a = lookalike(name, addr, 2, rng)
        assert (n, a) != (name, addr)


def test_generate_ground_truth_consistent():
    s1 = pd.DataFrame({"entity_id": ["S1-1", "S1-2"], "business_name": ["A Club SAS", "B Ecole"],
                       "business_address": ["1 Rue X, Lille, Hauts-de-France",
                                            "2 Avenue Y, Nantes, Pays de la Loire"],
                       "country": "France"})
    s1_, s2, s3, gt = generate(s1, np.array([1, 2]), np.array([0, 1]), 1.0, 0)
    ids = set(s2.entity_id) | set(s3.entity_id)
    matched = [i for x in gt.matched_entity_ids for i in x.split(",") if i]
    assert set(matched) <= ids and len(matched) == len(set(matched))
    assert len(ids) > len(matched)  # distractors exist and are unmatched
