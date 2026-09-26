"""Rewrite feature parquet parts compactly (2026-09-25): float64 -> float32,
int64/uint32 -> int32, zstd compression. Values the models see are unchanged: the
scorer and rescore already cast every feature to float32. Row order is kept, so
scores stay aligned with feature parts. Each part is written to a temp file,
checked (same rows, columns and float32 values), then swapped in.
Usage (from student_resource/): .venv/bin/python experiments/compact_features.py DIR [DIR ...]"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def compact(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        t = out[c].dtype
        if t == np.float64:
            out[c] = out[c].astype(np.float32)
        elif t in (np.int64, np.uint32):
            out[c] = out[c].astype(np.int32)
    return out


for d in map(Path, sys.argv[1:]):
    before = after = 0
    parts = sorted(d.glob("*.parquet"))
    for i, f in enumerate(parts):
        df = pd.read_parquet(f)
        c = compact(df)
        tmp = f.with_suffix(".tmp")
        c.to_parquet(tmp, index=False, compression="zstd")
        back = pd.read_parquet(tmp)
        assert len(back) == len(df) and list(back.columns) == list(df.columns), f
        num = [k for k in df.columns if df[k].dtype.kind in "fiu"]
        assert np.array_equal(back[num].to_numpy(np.float32), df[num].to_numpy(np.float32),
                              equal_nan=True), f
        before += f.stat().st_size
        after += tmp.stat().st_size
        tmp.replace(f)
        if i % 20 == 0:
            print(f"{d.name}: {i + 1}/{len(parts)}", flush=True)
    print(f"{d}: {before / 1e9:.1f} GB -> {after / 1e9:.1f} GB", flush=True)
