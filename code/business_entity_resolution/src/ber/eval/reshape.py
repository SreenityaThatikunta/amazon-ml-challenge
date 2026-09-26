"""Test-shaped data (plan.md §0.1b N1/N3, experiments E26).

About 60-66% of test's no-match records are S2/S3 copies of entities whose S1 is not
in the test pool; train and val have almost none. Dropping a hashed fraction of S1
(per country, on the chosen sides) while keeping their S2/S3 copies, which become
unmatched records, reproduces that: records per S1 go 4.68 -> 5.76-5.82 at a drop
rate of ~0.19-0.20, as on test. The original owner of each orphaned copy is kept in
``owner_dropped`` so its false links can be counted."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .split import stable_unit


def dropped_s1(s1_split: pd.DataFrame, drop: dict[str, float], sides: list[str],
               seed: int) -> set[str]:
    """Ids of the S1 to drop: per country, a stable-hash fraction ``drop[country]``
    of the S1 on ``sides`` (``s1_split`` needs entity_id, country, side)."""
    if not drop:
        return set()
    u = stable_unit(s1_split.entity_id, seed + 11)
    rate = s1_split.country.map(drop).fillna(0.0).to_numpy()
    m = s1_split.side.isin(sides).to_numpy() & (u < rate)
    return set(s1_split.entity_id[m])


def orphan_queries(q_split: pd.DataFrame, dropped: set[str]) -> pd.DataFrame:
    """Queries whose owner was dropped lose it (owner -> ""); the old owner goes to
    ``owner_dropped`` ("" for every other record)."""
    q = q_split.copy()
    hit = q.owner.isin(dropped).to_numpy()
    q["owner_dropped"] = np.where(hit, q.owner, "")
    q.loc[hit, "owner"] = ""
    return q
