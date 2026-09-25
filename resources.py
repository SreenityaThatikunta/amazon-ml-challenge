"""Resources learned from the TRAINING ground truth only.

1. translit_tokens  – Indic-script word -> Latin word, learned by positional
   alignment of Source-1 names with their native-script Source-2/3 matches.
2. translit_phrases – native-script state names -> Latin state name, learned
   from co-occurrence with the state named in the matched Source-1 address.
3. name_canon       – Latin spelling variants that transliterate the same
   native word (sree/shree/shri/sri, laxmi/lakshmi, jai/jay ...) are collapsed
   to one canonical spelling.
"""
from __future__ import annotations

import collections
import json
import re

import polars as pl

from .normalize import IN_STATES, INDIC_RE, LEGAL_CANON

_IND = re.compile(r"[ऀ-ൿ]")
_PHRASE = re.compile(r"[ऀ-ൿ][ऀ-ൿ ]*[ऀ-ൿ]|[ऀ-ൿ]")


def learn_resources(s1: pl.DataFrame, others: pl.DataFrame, gt_pairs: pl.DataFrame) -> dict:
    pos = (
        gt_pairs.join(s1.rename({"entity_id": "s1"}), on="s1")
        .join(
            others.rename({"entity_id": "mid", "business_name": "n2", "business_address": "a2"}).drop("country"),
            on="mid",
        )
    )
    # ---- 1. token transliteration dictionary --------------------------------
    q = pos.filter(pl.col("n2").str.contains(INDIC_RE)).select("business_name", "n2")
    cnt: collections.Counter = collections.Counter()
    tot: collections.Counter = collections.Counter()
    for a, b in q.iter_rows():
        ta, tb = a.split(), b.split()
        if len(ta) != len(tb):
            continue
        for x, y in zip(ta, tb):
            if _IND.search(y):
                x = re.sub(r"[^\w]", "", x.lower())
                y = y.strip(".,()[]")
                if x:
                    cnt[(y, x)] += 1
                    tot[y] += 1
    best: dict = {}
    for (y, x), c in cnt.items():
        if c > best.get(y, ("", 0))[1]:
            best[y] = (x, c)
    tok_map = {y: x for y, (x, c) in best.items() if c >= 2}

    # ---- 3. Latin variant classes (union-find over variants of one native word)
    parent: dict = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    latin_freq: collections.Counter = collections.Counter()
    for (y, x), c in cnt.items():
        latin_freq[x] += c
    by_native = collections.defaultdict(list)
    for (y, x), c in cnt.items():
        if c >= 20 and c / tot[y] >= 0.05:
            by_native[y].append(x)
    for y, xs in by_native.items():
        for x in xs[1:]:
            ra, rb = find(xs[0]), find(x)
            if ra != rb:
                parent[ra] = rb
    classes = collections.defaultdict(list)
    for x in list(parent):
        classes[find(x)].append(x)
    name_canon = {}
    for members in classes.values():
        if len(members) < 2:
            continue
        canon = max(members, key=lambda m: latin_freq[m])
        for m in members:
            if (
                m != canon
                and m not in LEGAL_CANON and canon not in LEGAL_CANON
                and len(m) >= 3 and len(canon) >= 3
                and m[0] == canon[0]
            ):
                name_canon[m] = canon
    # apply canonicalisation to dictionary targets too
    tok_map = {k: name_canon.get(v, v) for k, v in tok_map.items()}

    # ---- 2. native state phrases --------------------------------------------
    qa = pos.filter(pl.col("a2").str.contains(INDIC_RE)).select("business_address", "a2")
    qa = qa.sample(n=min(qa.height, 300_000), seed=0)
    states = sorted(IN_STATES, key=len, reverse=True)
    pc: collections.Counter = collections.Counter()
    for a1, a2 in qa.iter_rows():
        phr = _PHRASE.findall(a2)
        if not phr:
            continue
        low = a1.lower()
        st = next((s for s in states if s in low), None)
        if st is None:
            continue
        for p in phr:
            pc[(p.strip(), st)] += 1
    pbest: dict = {}
    ptot: collections.Counter = collections.Counter()
    for (p, s), c in pc.items():
        ptot[p] += c
        if c > pbest.get(p, ("", 0))[1]:
            pbest[p] = (s, c)
    phrase_map = {p: s for p, (s, c) in pbest.items() if c >= 20 and c / ptot[p] >= 0.5}
    return {"translit_tokens": tok_map, "translit_phrases": phrase_map, "name_canon": name_canon}


def save(res: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=0)


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
