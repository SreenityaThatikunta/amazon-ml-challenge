#!/usr/bin/env python3
"""List the license of every installed distribution and flag non-permissive ones.

Allowed: MIT, BSD, Apache, PSF/Python, ISC, and similar permissive licenses.
Everything else (GPL/AGPL/LGPL, MPL, unknown, CC-BY-NC, ...) is flagged for
manual review. Exit code 1 if anything is flagged.

    python tools/check_licenses.py
"""

import re
import sys
from importlib.metadata import distributions

PERMISSIVE = re.compile(
    r"\b(MIT|BSD|Apache|PSF|Python Software Foundation|ISC|Unlicense|"
    r"Zlib|0BSD|Historical Permission)\b", re.I)
COPYLEFT = re.compile(r"\b(A?GPL|LGPL|MPL|Mozilla|CC-BY-NC|Commons Clause)\b", re.I)
IGNORE = {"pip", "setuptools", "wheel", "ber"}


def license_of(dist):
    meta = dist.metadata
    expr = meta.get("License-Expression")
    if expr:
        return expr
    classifiers = [c.split("::")[-1].strip() for c in meta.get_all("Classifier") or []
                   if c.startswith("License ::")]
    if classifiers:
        return "; ".join(classifiers)
    lic = (meta.get("License") or "").strip()
    # some packages put the full license text here; keep the first line
    return lic.splitlines()[0][:80] if lic else "UNKNOWN"


def main():
    rows, flagged = [], []
    for dist in distributions():
        name = dist.metadata["Name"]
        if not name or name.lower() in IGNORE:
            continue
        lic = license_of(dist)
        ok = bool(PERMISSIVE.search(lic)) and not COPYLEFT.search(lic)
        rows.append((name, dist.version, lic, ok))
        if not ok:
            flagged.append(name)
    for name, ver, lic, ok in sorted(rows, key=lambda r: r[0].lower()):
        print(f"{'  ' if ok else '!!'} {name:<22} {ver:<14} {lic}")
    if flagged:
        print(f"\nREVIEW NEEDED: {', '.join(flagged)}")
        return 1
    print("\nall permissive")
    return 0


if __name__ == "__main__":
    sys.exit(main())
