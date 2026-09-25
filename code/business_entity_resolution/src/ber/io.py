"""File formats: challenge TSVs in, parquet artifacts, and submission TSVs out."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd

from .config import Config

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
MATCHING_HEADER = ("source1_entity_id", "matched_entity_ids")
CANDIDATE_HEADER = ("source1_entity_id", "candidate_entity_ids")


def read_tsv(path: Path) -> pd.DataFrame:
    """Read a challenge TSV exactly as written: all strings, no NA parsing,
    no quote handling (names and addresses can contain quote characters)."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       na_filter=False, quoting=csv.QUOTE_NONE)


def source_path(cfg: Config, split: str, n: int) -> Path:
    return cfg.data_dir / split / f"{split}_source{n}.tsv"


def read_source(cfg: Config, split: str, n: int) -> pd.DataFrame:
    """One source file, with its columns and the S<n>- id prefix checked."""
    df = read_tsv(source_path(cfg, split, n))
    if list(df.columns) != SOURCE_COLUMNS:
        raise ValueError(f"{split} source{n}: unexpected columns {list(df.columns)}")
    bad = ~df.entity_id.str.startswith(f"S{n}-")
    if bad.any():
        raise ValueError(f"{split} source{n}: {bad.sum()} ids without S{n}- prefix, "
                         f"e.g. {df.entity_id[bad].iloc[0]}")
    return df


def read_ground_truth(cfg: Config) -> pd.DataFrame:
    df = read_tsv(cfg.data_dir / "train" / "train_ground_truth.tsv")
    if list(df.columns) != GT_COLUMNS:
        raise ValueError(f"ground truth: unexpected columns {list(df.columns)}")
    return df


def read_id_lists(path: Path) -> dict[str, list[str]]:
    """Read a 2-column TSV of `id <tab> comma-separated ids` (GT or submission)."""
    out: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            key, _, ids = line.partition("\t")
            out[key.strip()] = [i.strip() for i in ids.split(",") if i.strip()]
    return out


def write_id_lists(path: Path, lists: Mapping[str, Iterable[str]],
                   all_keys: Iterable[str], header: tuple[str, str]) -> int:
    """Write one row per key in ``all_keys`` (keys missing from ``lists`` get an
    empty list), dropping duplicate ids while keeping order. Enforces the
    submission rules; returns the number of rows written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    seen_keys: set[str] = set()
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(header) + "\n")
        for key in all_keys:
            if key in seen_keys:
                raise ValueError(f"duplicate key {key}")
            seen_keys.add(key)
            ids = list(dict.fromkeys(lists.get(key, ())))
            bad = [i for i in ids if not (i.startswith("S2-") or i.startswith("S3-"))]
            if bad:
                raise ValueError(f"{key}: non S2/S3 ids {bad[:3]}")
            f.write(f"{key}\t{','.join(ids)}\n")
            n += 1
    extra = set(lists) - seen_keys
    if extra:
        raise ValueError(f"{len(extra)} keys not in all_keys, e.g. {sorted(extra)[:3]}")
    return n


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)
