"""``ber submit`` (task T3.5): write output/matching_results.tsv and
output/candidate_pairs.tsv for the test set, then run the official validator.

candidate_pairs.tsv is the full stage-1 candidate set, which is exactly what the
scorer runs inference on. With ~10^8 pairs, grouping uses integer codes,
not Python strings.
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

        with timed(st.log, "candidate_pairs.tsv"):
            sc, qc = [], []
            for f in sorted(st.input_path("candidates", "candidates_test").glob("*.parquet")):
                p = pd.read_parquet(f, columns=["q_id", "s1_id"])
                sc.append(pd.Categorical(p.s1_id.astype(str), dtype=s1_cat).codes)
                qc.append(pd.Categorical(p.q_id.astype(str), dtype=q_cat).codes)
            sc_all, qc_all = np.concatenate(sc), np.concatenate(qc)
            n = _write_grouped(out / "candidate_pairs.tsv", s1_ids, sc_all, q_ids, qc_all,
                               CANDIDATE_HEADER)
            st.register("candidate_pairs", out / "candidate_pairs.tsv", rows=n)
            st.metric("candidate_pairs", len(sc_all))

        with timed(st.log, "official validator"):
            res = subprocess.run(
                [sys.executable, str(cfg.root / "utils" / "validate_submission.py"),
                 "--matching", str(out / "matching_results.tsv"),
                 "--candidate", str(out / "candidate_pairs.tsv"),
                 "--test-dir", str(cfg.data_dir / "test")],
                capture_output=True, text=True)
            st.log.info("validator output:\n%s%s", res.stdout, res.stderr)
            st.metric("validator", "PASS" if res.returncode == 0 else "FAIL")
            if res.returncode != 0:
                raise RuntimeError("submission failed validation; see log.txt")
