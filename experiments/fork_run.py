"""Start a new run that reuses (symlinks) upstream stage folders of an existing run, so a
variant that only changes later stages (scorer, rerank, decide, ...) doesn't rebuild
normalize / candidates / features. Stages run in the new run write into its own folders.

Usage (from student_resource/):
  .venv/bin/python experiments/fork_run.py SRC DST normalize candidates features [scorer ...]
Then e.g.: .venv/bin/ber rerank -r DST -c <config> -s rerank.copy_features=true"""
import sys
from pathlib import Path

src, dst, stages = sys.argv[1], sys.argv[2], sys.argv[3:]
A = Path("artifacts")
if not (A / src).is_dir():
    sys.exit(f"no run {src!r} under artifacts/")
(A / dst).mkdir(exist_ok=True)
for s in stages:
    target, link = (A / src / s).resolve(), A / dst / s
    if not target.is_dir():
        sys.exit(f"{src} has no stage {s!r}")
    if link.is_symlink() or link.exists():
        if not link.is_symlink():
            sys.exit(f"{link} exists and is not a link; not touching it")
        link.unlink()
    link.symlink_to(target, target_is_directory=True)
    print(f"{link} -> {target}")
