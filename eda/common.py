"""Shared helpers for the EDA scripts: loading (with a pickle cache),
normalisation, script detection and report tee-ing."""

import csv
import pickle
import re
import sys
import time
import unicodedata
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"
CACHE = ROOT / "eda" / ".cache"
REPORTS = ROOT / "eda" / "reports"

SEED = 42

# ---------------------------------------------------------------- loading


def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       na_filter=False, quoting=csv.QUOTE_NONE)


def _cached(name, fn):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{name}.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    t = time.time()
    obj = fn()
    with open(p, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[cache] built {name} in {time.time() - t:.1f}s", file=sys.stderr)
    return obj


def load_source(split, n):
    """split in {'train','test'}, n in {1,2,3}."""
    return _cached(f"{split}_s{n}",
                   lambda: read_tsv(DATA / split / f"{split}_source{n}.tsv"))


def load_all(split):
    """All three sources of a split, with a 'source' column (1/2/3)."""
    def build():
        parts = []
        for n in (1, 2, 3):
            d = load_source(split, n).copy()
            d["source"] = n
            parts.append(d)
        return pd.concat(parts, ignore_index=True)
    return _cached(f"{split}_all", build)


def load_gt():
    return _cached("train_gt", lambda: read_tsv(
        DATA / "train" / "train_ground_truth.tsv"))


def load_pairs():
    """Exploded ground truth: one row per (s1_id, other_id) true pair."""
    def build():
        gt = load_gt()
        s = gt.set_index("source1_entity_id").matched_entity_ids
        s = s[s != ""].str.split(",").explode()
        return pd.DataFrame({"s1_id": s.index.values,
                             "other_id": s.values}).reset_index(drop=True)
    return _cached("train_pairs", build)


def pairs_with_records(n_sample=None, seed=SEED):
    """True pairs joined with both records' fields (optionally sampled)."""
    pairs = load_pairs()
    if n_sample and n_sample < len(pairs):
        pairs = pairs.sample(n_sample, random_state=seed)
    recs = load_all("train").set_index("entity_id")
    a = recs.loc[pairs.s1_id, ["business_name", "business_address", "country"]]
    b = recs.loc[pairs.other_id, ["business_name", "business_address", "country"]]
    return pd.DataFrame({
        "s1_id": pairs.s1_id.values, "other_id": pairs.other_id.values,
        "src": pairs.other_id.str[:2].values,
        "country": a.country.values, "other_country": b.country.values,
        "name1": a.business_name.values, "name2": b.business_name.values,
        "addr1": a.business_address.values, "addr2": b.business_address.values,
    })


# ---------------------------------------------------------- normalisation

_NON_ALNUM = re.compile(r"[^\w]+", re.UNICODE)
_DIGITS = re.compile(r"\d+")

LEGAL_SUFFIXES = {
    # US
    "inc", "incorporated", "llc", "l", "c", "corp", "corporation", "co",
    "company", "ltd", "limited", "lp", "llp", "pc", "p", "pllc", "plc",
    # India
    "pvt", "private", "opc",
    # France
    "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "scop",
}


def strip_accents(s):
    """Remove Latin combining diacritics only (keeps Indic vowel signs)."""
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not 0x0300 <= ord(ch) <= 0x036F)


def norm(s):
    s = strip_accents(s.lower())
    s = s.replace("&", " and ").replace("+", " and ")
    return " ".join(_NON_ALNUM.sub(" ", s).split())


def tokens(s):
    return norm(s).split()


def core_name_tokens(s):
    return [t for t in tokens(s) if t not in LEGAL_SUFFIXES]


def numbers(s):
    return _DIGITS.findall(s)


def char_ngrams(s, n=3):
    s = f" {norm(s)} "
    return {s[i:i + n] for i in range(max(len(s) - n + 1, 1))}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


_SCRIPTS = [
    (0x0900, 0x097F, "Devanagari"), (0x0980, 0x09FF, "Bengali"),
    (0x0A00, 0x0A7F, "Gurmukhi"), (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Oriya"), (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"), (0x0C80, 0x0CFF, "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam"),
]


def script_of(s):
    """Dominant script among letters: Latin / an Indic script / Other / Empty."""
    counts = {}
    for ch in s:
        if not ch.isalpha():
            continue
        o = ord(ch)
        name = "Latin" if o < 0x0250 or 0x1E00 <= o <= 0x1EFF else "Other"
        for lo, hi, sc in _SCRIPTS:
            if lo <= o <= hi:
                name = sc
                break
        counts[name] = counts.get(name, 0) + 1
    if not counts:
        return "Empty"
    return max(counts, key=counts.get)


def has_non_ascii_latin(s):
    return any(0x00C0 <= ord(c) <= 0x024F or 0x1E00 <= ord(c) <= 0x1EFF
               for c in s)


def case_style(s):
    letters = [c for c in s if c.isalpha() and c.isascii()]
    if not letters:
        return "no-latin"
    if all(c.isupper() for c in letters):
        return "UPPER"
    if all(c.islower() for c in letters):
        return "lower"
    return "Mixed"


# ------------------------------------------------------------- reporting


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


@contextmanager
def report(name):
    """Tee stdout into eda/reports/<name>.txt."""
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"{name}.txt"
    old = sys.stdout
    t = time.time()
    with open(path, "w", encoding="utf-8") as f:
        sys.stdout = _Tee(old, f)
        try:
            yield
        finally:
            print(f"\n[done in {time.time() - t:.1f}s]")
            sys.stdout = old
    print(f"[report] {path.relative_to(ROOT)}", file=sys.stderr)


def h(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def vc(series, top=15, normalize=True):
    """Compact value_counts print."""
    v = series.value_counts(normalize=normalize).head(top)
    fmt = "{:.4f}" if normalize else "{}"
    for k, x in v.items():
        print(f"  {str(k)[:60]:<60} {fmt.format(x)}")


def quantiles(series, qs=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)):
    q = np.quantile(np.asarray(series, dtype=float), qs)
    return "  ".join(f"p{int(a * 100)}={b:.3f}" for a, b in zip(qs, q))


pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 80)
