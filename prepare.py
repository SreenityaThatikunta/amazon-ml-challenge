"""Read the raw TSVs, normalise every source and cache as parquet."""
from __future__ import annotations

import os
import time
import zipfile

import polars as pl

from . import normalize

KEEP = [
    "entity_id", "business_name", "business_address", "ctry",
    "n_full", "n_core", "n_alt", "n_isdom", "n_hasdba", "a_full", "a_nums", "a_words",
]


class DataSource:
    """Reads the challenge TSVs either from an extracted `dataset/` folder or straight
    out of the organiser's zip (no extraction needed, saves ~2.5 GB of disk)."""

    def __init__(self, path: str):
        self.path = path
        self.zip = None
        if os.path.isfile(path) and path.lower().endswith(".zip"):
            self.zip = zipfile.ZipFile(path)
            names = [n for n in self.zip.namelist() if n.endswith(".tsv") and "__MACOSX" not in n]
            self.members = {n.split("/")[-1]: n for n in names}

    def tsv(self, split: str, fname: str) -> str | bytes:
        if self.zip is not None:
            return self.zip.read(self.members[fname])
        return os.path.join(self.path, split, fname)

    def extract_file(self, suffix: str, dest: str) -> str | None:
        """Extract one helper file (e.g. utils/validate_submission.py) from the zip."""
        if self.zip is None:
            return None
        for n in self.zip.namelist():
            if n.endswith(suffix) and "__MACOSX" not in n:
                with open(dest, "wb") as f:
                    f.write(self.zip.read(n))
                return dest
        return None


def read_tsv(path) -> pl.DataFrame:
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    return df.with_columns(
        pl.col("business_name").fill_null(""),
        pl.col("business_address").fill_null(""),
        pl.col("country").fill_null(""),
    )


def prepare_source(path, res: dict, out: str, chunk: int = 1_000_000, label: str = "") -> None:
    t = time.time()
    df = read_tsv(path)
    parts = []
    for i in range(0, df.height, chunk):
        d = normalize.normalise_frame(df.slice(i, chunk), res).select(KEEP)
        d = d.with_columns(
            pl.col("n_core").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).alias("n_toks"),
            pl.col("n_core").str.replace_all(" ", "").alias("n_cat"),
        )
        parts.append(d)
    out_df = pl.concat(parts)
    out_df.write_parquet(out)
    print(time.strftime("[%H:%M:%S] ") + f"prepared {label}: {out_df.height:,} rows in {time.time()-t:.0f}s", flush=True)


def read_ground_truth(path) -> pl.DataFrame:
    gt = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    gt = gt.rename({"source1_entity_id": "s1", "matched_entity_ids": "mid"})
    gt = gt.with_columns(pl.col("mid").fill_null("").str.split(",")).explode("mid")
    return gt.with_columns(pl.col("mid").str.strip_chars()).filter(pl.col("mid") != "")
