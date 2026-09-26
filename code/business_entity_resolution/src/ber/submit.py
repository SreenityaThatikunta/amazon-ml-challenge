"""``ber submit`` (task T3.5): write output/matching_results.tsv for the test
set and run the official validator on it.

Also (``submit.write_candidates``, on by default) output/candidate_pairs.tsv,
the full stage-1 candidate set (~2.4e8 pairs, ~3 GB; grouped with integer
codes). It is required in the final submission zip: it is the exact set the
scorer runs inference over. Every match is checked to be one of its S1's
candidates here, because the official validator needs ~30 GB of RAM for that.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd

from .artifacts import StageRun
from .config import Config
from .io import CANDIDATE_HEADER, MATCHING_HEADER
from .log import timed


def _write_grouped(path, s1_ids: np.ndarray, s1_codes: np.ndarray, q_ids: np.ndarray,
                   q_codes: np.ndarray, header: tuple[str, str]) -> int:
    """One line per S1 (all of ``s1_ids``, in order), listing its q ids."""
    order = np.lexsort((q_codes, s1_codes))
    s_sorted, q_sorted = s1_codes[order], q_codes[order]
    starts = np.searchsorted(s_sorted, np.arange(len(s1_ids)), side="left")
    ends = np.searchsorted(s_sorted, np.arange(len(s1_ids)), side="right")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(header) + "\n")
        for i, sid in enumerate(s1_ids):
            qs = q_sorted[starts[i]:ends[i]]
            f.write(f"{sid}\t{','.join(q_ids[qs])}\n")
    return len(s1_ids)


def run_submit(cfg: Config) -> None:
    if cfg.data.slice is not None:
        raise SystemExit("submit needs the full pipeline (data.slice must be null)")
    with StageRun(cfg, "submit") as st:
        recs = pd.read_parquet(st.input_path("normalize", "records_test"),
                               columns=["entity_id", "source"])
        s1_ids = recs.entity_id[recs.source == 1].to_numpy()
        q_ids = recs.entity_id[recs.source != 1].to_numpy()
        s1_cat = pd.CategoricalDtype(s1_ids)
        q_cat = pd.CategoricalDtype(q_ids)
        out = cfg.output_dir
        out.mkdir(parents=True, exist_ok=True)

        dec = st.read_input("decide", "decisions_test")
        s_codes = pd.Categorical(dec.s1_id.astype(str), dtype=s1_cat).codes
        q_codes = pd.Categorical(dec.q_id.astype(str), dtype=q_cat).codes
        if (s_codes < 0).any() or (q_codes < 0).any():
            raise ValueError("decisions reference ids that are not in the test set")
        n = _write_grouped(out / "matching_results.tsv", s1_ids, s_codes, q_ids, q_codes,
                           MATCHING_HEADER)
        st.register("matching_results", out / "matching_results.tsv", rows=n)
        st.metric("matched_pairs", len(dec))

        if cfg.submit.write_candidates:
            _write_candidates(st, out, s1_ids, q_ids, s1_cat, q_cat, s_codes, q_codes)
        else:
            st.log.info("candidate_pairs.tsv skipped (submit.write_candidates=false)")

        with timed(st.log, "official validator"):
            _validate(st, cfg, out)


def _global_codes(col, index: pd.Index) -> np.ndarray:
    """Codes into ``index`` for a (possibly dictionary-encoded) arrow id column.
    Dictionary columns map their small dictionary once and take the indices, instead
    of hashing every row (244M string ids took >35 min that way)."""
    import pyarrow as pa
    out = []
    for chunk in col.chunks:
        if pa.types.is_dictionary(chunk.type):
            lut = index.get_indexer(chunk.dictionary.to_pandas().astype(str))
            out.append(lut[chunk.indices.to_numpy(zero_copy_only=False)])
        else:
            out.append(index.get_indexer(chunk.to_pandas().astype(str)))
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


def _write_candidates(st, out, s1_ids, q_ids, s1_cat, q_cat,
                      match_s: np.ndarray, match_q: np.ndarray) -> None:
    import pyarrow.parquet as pq
    with timed(st.log, "candidate_pairs.tsv"):
        s_index, q_index = pd.Index(s1_cat.categories), pd.Index(q_cat.categories)
        sc, qc = [], []
        for f in sorted(st.input_path("candidates", "candidates_test").glob("*.parquet")):
            t = pq.read_table(f, columns=["q_id", "s1_id"])
            sc.append(_global_codes(t.column("s1_id"), s_index))
            qc.append(_global_codes(t.column("q_id"), q_index))
        sc_all, qc_all = np.concatenate(sc), np.concatenate(qc)
        if (sc_all < 0).any() or (qc_all < 0).any():
            raise ValueError("candidates reference ids that are not in the test set")
        n_missing = _matches_not_in_candidates(sc_all, qc_all, match_s, match_q, len(q_ids))
        st.metric("matches_not_in_candidates", n_missing)
        if n_missing:
            raise ValueError(f"{n_missing} matched pairs are not in the candidate set")
        n = _write_grouped(out / "candidate_pairs.tsv", s1_ids, sc_all, q_ids, qc_all,
                           CANDIDATE_HEADER)
        st.register("candidate_pairs", out / "candidate_pairs.tsv", rows=n)
        st.metric("candidate_pairs", len(sc_all))


def _matches_not_in_candidates(cand_s: np.ndarray, cand_q: np.ndarray, match_s: np.ndarray,
                               match_q: np.ndarray, n_q: int) -> int:
    """Number of matched (s1, q) pairs that are not candidate pairs."""
    cand = cand_s.astype(np.int64) * n_q + cand_q
    match = match_s.astype(np.int64) * n_q + match_q
    return int((~np.isin(match, cand)).sum())


def _validate(st, cfg: Config, out) -> None:
    """Official validator on matching_results.tsv only. It runs from utils/ on
    purpose: when --candidate is omitted the validator loads
    ./output/candidate_pairs.tsv if it exists, and at ~2.4e8 ids that needs
    ~30 GB of RAM."""
    res = subprocess.run(
        [sys.executable, "validate_submission.py",
         "--matching", str(out / "matching_results.tsv"),
         "--test-dir", str(cfg.data_dir / "test")],
        capture_output=True, text=True, cwd=cfg.root / "utils")
    st.log.info("validator output:\n%s%s", res.stdout, res.stderr)
    st.metric("validator", "PASS" if res.returncode == 0 else "FAIL")
    if res.returncode != 0:
        raise RuntimeError("submission failed validation; see log.txt")
