"""Synthetic French S2/S3 records with ground truth (task T5.1).

France appears only in test. Its unlabelled records show the same generator as
US/India (eda/FINDINGS.md §6), so we recreate labelled data by applying the
observed noise catalogue to test S1 France records (use of unlabelled test data
approved; disclosed in the methodology):

* true copies per S1, with cluster sizes drawn from the train distribution;
  S2 style = upper-case, abbreviated (R, BD, AV, N°), département or région;
  S3 style = mixed case (R., Av., Nº), département or région
* name noise: case, accents, legal form dropped/moved/expanded/changed,
  generic words, wrappers, scanner digits, letter drop/double/swap, website
  form, "<gibberish> aka <name>", pure gibberish, initials
* address noise: street abbreviations, reordering, dropped parts, house
  number truncated/zero-padded/"N°", placeholders, empty address
* lookalike distractors (unmatched): one-letter substitution / legal-form change
  at the same address; same name with a shifted house number (+ PMB/BP)

Realism (T7.1b; eda/reports/07_france_errors.txt): **all** test S1 France are
indexed, so the dense S1-level lookalikes of real France (13% of S1 share an
address with another S1) confuse the proxy as they confuse the real test, but
only a sample gets synthetic copies and is scored (``eval_s1.tsv``). Cluster
sizes are drawn jointly for S2 and S3 (train singleton rate), and real test
France records that the main model confidently leaves unmatched (best p below
``background_max_p``) are added as unrelated unmatched records.

Output mimics the dataset layout (source1/2/3 + ground truth TSVs), so the
normal pipeline stages can run on it.
"""

from __future__ import annotations

import csv
import random
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


REGION_DEPT = {"Hauts-de-France": ["Nord", "Pas-de-Calais"],
               "Nouvelle-Aquitaine": ["Gironde"],
               "Pays de la Loire": ["Loire-Atlantique"]}
STREET_S2 = {"rue": ["R", "R.", "RUE"], "avenue": ["AV", "AVE", "AVENUE"],
             "boulevard": ["BD", "BLVD", "BOULEVARD"], "cours": ["CRS", "COURS"],
             "route": ["RTE", "ROUTE"], "place": ["PL", "PLACE"], "allee": ["ALL", "ALLEE"],
             "impasse": ["IMP", "IMPASSE"], "chemin": ["CH", "CHEMIN"]}
STREET_S3 = {"rue": ["R", "R.", "Rue"], "avenue": ["Av.", "Ave", "Avenue"],
             "boulevard": ["Bd", "Bd.", "Boulevard"], "cours": ["Crs", "Cours"],
             "route": ["Rte.", "Route"], "place": ["Pl.", "Place"], "allee": ["Allée", "All."],
             "impasse": ["Imp.", "Impasse"], "chemin": ["Ch.", "Chemin"]}
LEGAL = ["SARL", "SAS", "EURL", "SA", "SASU", "SCI", "SNC", "EI"]
LEGAL_EXPANDED = {"SARL": "S.A.R.L.", "SAS": "S.A.S.", "SA": "S.A.", "EI": "E.I."}
GENERIC = ["Groupe", "Services", "Centre", "& Fils", "(France)", "Distribution", "Holding",
           "Compagnie"]
WRAPPERS = [(">> ", ""), ("-- ", ""), ("[", "]"), ("((", "))"), ("", " ")]
SYLLABLES = ["lyra", "brix", "yuma", "jax", "zeph", "nyla", "vera", "delta", "onyx", "xylo",
             "kor", "riza", "belo", "dova", "lum", "vio", "zeta", "tavo", "orbi", "cira"]
OCR = {"o": "0", "l": "1", "s": "5", "b": "8", "g": "6"}
PLACEHOLDERS = ["null", "N/A", "NULL", "<NULL>"]


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _add_accent(s: str, rng: random.Random) -> str:
    idx = [i for i, c in enumerate(s) if c.lower() in "aeiou"]
    if not idx:
        return s
    i = rng.choice(idx)
    acc = {"a": "à", "e": "é", "i": "î", "o": "ô", "u": "ù"}
    c = acc[s[i].lower()]
    return s[:i] + (c.upper() if s[i].isupper() else c) + s[i + 1:]


def _typo(word: str, rng: random.Random) -> str:
    if len(word) < 4:
        return word
    i = rng.randrange(1, len(word) - 1)
    op = rng.choice(["drop", "double", "swap"])
    if op == "drop":
        return word[:i] + word[i + 1:]
    if op == "double":
        return word[:i] + word[i] + word[i:]
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


def _gibberish(rng: random.Random) -> str:
    return "".join(rng.sample(SYLLABLES, rng.choice([2, 2, 3]))).capitalize()


def _split_legal(name: str) -> tuple[str, str]:
    toks = name.split()
    legal = [t for t in toks if t.upper().strip("()[]") in LEGAL]
    core = [t for t in toks if t not in legal]
    return " ".join(core), (legal[0].strip("()[]").upper() if legal else "")


def noisy_name(name: str, style: int, rng: random.Random) -> str:
    """One true-copy variant of an S1 name (style 2 = S2-like, 3 = S3-like)."""
    core, legal = _split_legal(name)
    r = rng.random()
    if r < 0.03:
        return _gibberish(rng)
    if r < 0.06:
        return re.sub(r"[^a-z0-9]", "", _strip_accents(name).lower()) + rng.choice([".fr", ".com"])
    if r < 0.08:
        return "".join(w[0] for w in core.split() if w[0].isalpha()).upper() or core
    words = core.split()
    if rng.random() < 0.12:
        words = [_typo(w, rng) if rng.random() < 0.5 else w for w in words]
    if rng.random() < 0.05:
        words = ["".join(OCR.get(c, c) if rng.random() < 0.3 else c for c in w) for w in words]
    if rng.random() < 0.06 and len(words) > 1:
        i = rng.randrange(len(words) - 1)
        words[i], words[i + 1] = words[i + 1], words[i]
    if rng.random() < 0.15:
        words.append(rng.choice(GENERIC))
    core = " ".join(words)
    if legal:
        lr = rng.random()
        if lr < 0.25:
            legal = ""
        elif lr < 0.35:
            legal = rng.choice([x for x in LEGAL if x != legal])
        elif lr < 0.45:
            legal = LEGAL_EXPANDED.get(legal, legal)
    elif rng.random() < 0.2:
        legal = rng.choice(LEGAL)
    out = f"{legal} {core}" if legal and rng.random() < 0.2 else f"{core} {legal}".strip()
    if rng.random() < 0.05:
        out = f"{_gibberish(rng)} {rng.choice(['aka', 'formerly', 'F/K/A'])} {out}"
    if rng.random() < 0.06:
        a, b = rng.choice(WRAPPERS)
        out = f"{a}{out}{b}"
    if rng.random() < 0.1:
        out = _strip_accents(out)
    elif rng.random() < 0.12:
        out = _add_accent(out, rng)
    cr = rng.random()
    if style == 2 and cr < 0.2 or style == 3 and cr < 0.05:
        out = out.upper()
    elif cr < 0.27:
        out = out.lower()
    if rng.random() < 0.05:
        out = out.replace(" ", "  ", 1)
    return out


def _parse_address(addr: str) -> tuple[str, str, str]:
    """(street part, city, region) from an S1 France address (any order)."""
    comps = [c.strip() for c in addr.split(",") if c.strip()]
    region = next((c for c in comps if c in REGION_DEPT), "")
    street = next((c for c in comps if re.match(r"^\d", c)), comps[0] if comps else "")
    city = next((c for c in comps if c not in (region, street)), "")
    return street, city, region


def _street_variant(street: str, style: int, rng: random.Random) -> str:
    m = re.match(r"^(\d+)\s*(bis|ter|[A-Z])?\s+(\S+)\s+(.*)$", street, flags=re.I)
    if not m:
        return street.upper() if style == 2 else street
    num, suffix, stype, rest = m.groups()
    table = STREET_S2 if style == 2 else STREET_S3
    key = _strip_accents(stype).lower()
    if key in table:
        stype = rng.choice(table[key])
    nr = rng.random()
    if nr < 0.04 and len(num) > 1:
        num = num[:-1] if rng.random() < 0.5 else num[1:]           # truncated
    elif nr < 0.08:
        num = num.zfill(len(num) + rng.choice([1, 2]))               # zero-padded
    elif nr < 0.12:
        num = rng.choice(["N°", "Nº", "NO. ", "#"]) + num
    elif nr < 0.15:
        num = ""                                                     # number dropped
    s = " ".join(x for x in [num + (f" {suffix}" if suffix else ""), stype, rest] if x)
    return s.upper() if style == 2 else s


def noisy_address(addr: str, style: int, rng: random.Random) -> str:
    if rng.random() < 0.03:
        return ""
    street, city, region = _parse_address(addr)
    comps = [_street_variant(street, style, rng)]
    if city and rng.random() > 0.05:
        comps.append(city.upper() if style == 2 and rng.random() < 0.7 else city)
    if region and rng.random() > 0.3:
        comps.append(rng.choice(REGION_DEPT[region]) if rng.random() < 0.5 else region)
    if rng.random() < 0.02:
        comps.insert(rng.randrange(len(comps) + 1), rng.choice(PLACEHOLDERS))
    if rng.random() < 0.25:
        rng.shuffle(comps)
    return ", ".join(comps)


def lookalike(name: str, addr: str, style: int, rng: random.Random) -> tuple[str, str]:
    """An unmatched record imitating an S1 entity (EDA 04 hard negatives)."""
    if rng.random() < 0.5:  # same address, one arbitrary letter / legal form changed
        core, legal = _split_legal(name)
        words = core.split()
        i = rng.randrange(len(words))
        w = words[i]
        if len(w) > 3:
            j = rng.randrange(1, len(w))
            words[i] = w[:j] + rng.choice("abcdefghijklmnopqrstuvwxyz") + w[j + 1:]
        else:
            legal = rng.choice([x for x in LEGAL if x != legal])
        return f"{' '.join(words)} {legal}".strip(), noisy_address(addr, style, rng)
    street, city, region = _parse_address(addr)
    m = re.match(r"^(\d+)(.*)$", street)
    if m:  # same name, house number shifted, maybe with a box number
        street = f"{int(m.group(1)) + rng.choice([-4, -3, -2, 2, 3, 4, 6, 8])}{m.group(2)}"
    box = f", BP {rng.randrange(100, 9999)}" if rng.random() < 0.3 else ""
    return noisy_name(name, style, rng), noisy_address(
        ", ".join(x for x in [street, city, region] if x), style, rng) + box


SYNTH_ID_START = 9_000_000_000  # 10 digits: never collides with real (9-digit) ids


def generate(s1: pd.DataFrame, sizes: np.ndarray, distractor_rate: float, seed: int,
             sampled: pd.Index | None = None, background: pd.DataFrame | None = None
             ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """s1: entity_id, business_name, business_address, country (France); all of
    them go to source 1. ``sampled``: entity ids that get synthetic copies and
    distractors (default: all). ``sizes``: (n, 2) empirical (n_S2, n_S3) cluster
    sizes, one row per train S1, drawn jointly. ``background``: real records
    (entity_id, business_name, business_address, country) added unmatched to
    the source given by their id prefix.
    Returns (source1, source2, source3, ground_truth) frames; S1 outside
    ``sampled`` have empty match lists (score only ``sampled``)."""
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    rows = {2: [], 3: []}
    gt = []
    next_id = {2: SYNTH_ID_START, 3: SYNTH_ID_START}
    sampled_set = set(s1.entity_id if sampled is None else sampled)
    for r in s1.itertuples(index=False):
        matched = []
        if r.entity_id not in sampled_set:
            gt.append((r.entity_id, ""))
            continue
        n_by_src = sizes[nprng.integers(len(sizes))]
        for src, n in ((2, n_by_src[0]), (3, n_by_src[1])):
            for _ in range(int(n)):
                eid = f"S{src}-{next_id[src]}"
                next_id[src] += 1
                rows[src].append((eid, noisy_name(r.business_name, src, rng),
                                  noisy_address(r.business_address, src, rng), "France"))
                matched.append(eid)
            if rng.random() < distractor_rate:
                eid = f"S{src}-{next_id[src]}"
                next_id[src] += 1
                n, a = lookalike(r.business_name, r.business_address, src, rng)
                rows[src].append((eid, n, a, "France"))
        gt.append((r.entity_id, ",".join(matched)))
    cols = ["entity_id", "business_name", "business_address", "country"]
    s2 = pd.DataFrame(rows[2], columns=cols)
    s3 = pd.DataFrame(rows[3], columns=cols)
    if background is not None and len(background):
        bg = background[cols]
        s2 = pd.concat([s2, bg[bg.entity_id.str.startswith("S2-")]], ignore_index=True)
        s3 = pd.concat([s3, bg[bg.entity_id.str.startswith("S3-")]], ignore_index=True)
    return (s1[cols], s2, s3, pd.DataFrame(gt, columns=["source1_entity_id",
                                                        "matched_entity_ids"]))


def write_dataset(out_dir: Path, frames, split: str = "train") -> None:
    """Write frames in the dataset layout: <out_dir>/<split>/<split>_source{1,2,3}.tsv
    (+ ground truth). Tabs/newlines inside fields are replaced by spaces."""
    d = out_dir / split
    d.mkdir(parents=True, exist_ok=True)
    s1, s2, s3, gt = frames
    for n, f in ((1, s1), (2, s2), (3, s3)):
        f.replace({r"[\t\r\n]": " "}, regex=True).to_csv(
            d / f"{split}_source{n}.tsv", sep="\t", index=False, quoting=csv.QUOTE_NONE)
    gt.to_csv(d / f"{split}_ground_truth.tsv", sep="\t", index=False, quoting=csv.QUOTE_NONE)


def _background(cfg, france_s1: set[str], fc, seed: int) -> pd.DataFrame:
    """Real test France S2/S3 records whose best candidate in run
    ``fc.background_run`` has p < ``fc.background_max_p`` (confidently
    unmatched), sampled to ``fc.background_frac`` of the synthetic query count
    (estimated as n_entities x 4.2)."""
    from ..io import read_source

    scores = cfg.resolve(cfg.paths.artifacts_dir) / fc.background_run / "scorer" / "scores_test"
    keep = []
    for f in sorted(scores.glob("*.parquet")):
        s = pd.read_parquet(f).astype({"q_id": str, "s1_id": str})
        b = s.loc[s.groupby("q_id").p.idxmax()]
        keep.append(b.q_id[b.s1_id.isin(france_s1) & (b.p < fc.background_max_p)])
    ids = pd.concat(keep)
    n = min(len(ids), int(fc.background_frac * fc.n_entities * 4.2))
    ids = set(ids.sample(n=n, random_state=seed))
    recs = pd.concat([read_source(cfg, "test", k) for k in (2, 3)], ignore_index=True)
    return recs[recs.entity_id.isin(ids)]


def run_synth_france(cfg) -> None:
    """``ber synth-france``: sample test S1 France, generate labelled S2/S3, write a
    dataset folder usable as ``paths.data_dir`` of another run."""
    from ..artifacts import StageRun
    from ..io import read_ground_truth, read_source
    from ..log import timed

    fc = cfg.france
    with StageRun(cfg, "synth-france") as st:
        s1 = read_source(cfg, "test", 1)
        s1 = s1[s1.country == "France"]
        pool = s1.entity_id
        if fc.exclude:  # e.g. a training set must not reuse the proxy's scored S1
            pool = pool[~pool.isin(set(pd.read_csv(cfg.resolve(fc.exclude), sep="\t",
                                                   dtype=str).entity_id))]
        sampled = pd.Index(pool.sample(n=fc.n_entities, random_state=cfg.seed))
        # joint (S2, S3) cluster sizes per train S1: keeps the 5.6% singleton rate
        gt = read_ground_truth(cfg)
        lists = gt.matched_entity_ids.str.split(",")
        sizes = np.stack([lists.map(lambda x: sum(i.startswith(f"S{k}-") for i in x if i))
                          .to_numpy() for k in (2, 3)], axis=1)
        background = None
        if fc.background_run:
            with timed(st.log, "background records"):
                background = _background(cfg, set(s1.entity_id), fc, cfg.seed)
            st.metric("background_records", len(background))
        frames = generate(s1, sizes, fc.distractor_rate, cfg.seed, sampled, background)
        out = st.path("data")
        write_dataset(out, frames, "train")
        pd.DataFrame({"entity_id": sampled}).to_csv(out / "eval_s1.tsv", sep="\t", index=False)
        # normalize also reads test S1 (vocabulary); give it the same S1
        (out / "test").mkdir(parents=True, exist_ok=True)
        frames[0].to_csv(out / "test" / "test_source1.tsv", sep="\t", index=False,
                         quoting=csv.QUOTE_NONE)
        st.register("data", out)
        ev = frames[3][frames[3].source1_entity_id.isin(set(sampled))]
        st.metric("records", {"s1": len(frames[0]), "s1_scored": len(sampled),
                              "s2": len(frames[1]), "s3": len(frames[2]),
                              "true_pairs": int(ev.matched_entity_ids.str.count("S").sum()),
                              "singletons_scored": int((ev.matched_entity_ids == "").sum())})
        for src in (2, 3):
            st.log.info("sample S%d records:\n%s", src,
                        frames[src - 1].sample(12, random_state=1).to_string(index=False))
