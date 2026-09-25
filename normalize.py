"""Text normalisation for business names and addresses.

All heavy lifting is done with vectorised Polars string expressions so that
~10M records can be normalised in well under a minute on a laptop CPU.
Indic-script tokens are transliterated to Latin with a dictionary that is
*learned from the training pairs* (see resources.py); unseen native words are
left untouched.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable

import polars as pl

INDIC_RE = r"[ऀ-ൿ]"
_INDIC_PY = re.compile(INDIC_RE)

# --------------------------------------------------------------------------
# Generic linguistic knowledge (abbreviations / legal forms).  These are
# language conventions, not look-ups of business identities.
# --------------------------------------------------------------------------
LEGAL_CANON = {
    # English / India
    "incorporated": "inc", "inc": "inc", "corporation": "corp", "corp": "corp",
    "company": "co", "co": "co", "cos": "co", "limited": "ltd", "ltd": "ltd", "lt": "ltd",
    "private": "pvt", "pvt": "pvt", "pte": "pvt", "llc": "llc", "llp": "llp", "lp": "lp",
    "pllc": "pllc", "plc": "plc", "pc": "pc", "pa": "pa", "public": "public",
    "lc": "llc", "ltda": "ltd",
    # French
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "sci": "sci", "eurl": "eurl",
    "ei": "ei", "snc": "snc", "scop": "scop", "scm": "scm", "selarl": "selarl",
    "gie": "gie", "sca": "sca", "earl": "earl", "gaec": "gaec",
    "societe": "societe", "ste": "ste",
    "etablissements": "ets", "etablissement": "ets", "ets": "ets",
}
LEGAL_TOKENS = set(LEGAL_CANON.values())

OCR_MAP = {"0": "o", "1": "l", "5": "s", "8": "b", "6": "g", "3": "e", "4": "a", "7": "t", "9": "g", "2": "z"}

HONORIFICS = {"mr", "mrs", "ms", "dr", "smt", "shri", "sri", "shree", "m s", "messrs"}
NAME_NOISE = {"center", "centre", "ctr", "services", "service"}

STOPWORDS = {
    "the", "and", "of", "a", "an", "for", "at", "in", "on", "by",
    "de", "du", "des", "la", "le", "les", "et", "d", "l", "au", "aux", "en",
}

ADDR_CANON = {
    # street types (US / India)
    "street": "st", "str": "st", "st": "st", "road": "rd", "rd": "rd", "avenue": "ave",
    "ave": "ave", "av": "ave", "avn": "ave", "drive": "dr", "dr": "dr", "drv": "dr",
    "lane": "ln", "ln": "ln", "court": "ct", "ct": "ct", "crt": "ct", "circle": "cir",
    "cir": "cir", "circ": "cir", "boulevard": "blvd", "blvd": "blvd", "bd": "blvd",
    "boul": "blvd", "bld": "blvd", "highway": "hwy", "hwy": "hwy", "hiway": "hwy", "place": "pl",
    "pl": "pl", "plc": "pl", "cove": "cv", "cv": "cv", "trail": "trl", "trl": "trl",
    "parkway": "pkwy", "pkwy": "pkwy", "pky": "pkwy", "terrace": "ter", "ter": "ter",
    "terr": "ter", "square": "sq", "sq": "sq", "point": "pt", "pt": "pt", "loop": "loop",
    "way": "way", "wy": "way", "expressway": "expy", "expy": "expy", "freeway": "fwy",
    "fwy": "fwy", "turnpike": "tpke", "tpke": "tpke", "center": "ctr", "centre": "ctr",
    "ctr": "ctr", "cntr": "ctr", "mount": "mt", "mt": "mt", "mountain": "mtn", "mtn": "mtn",
    "heights": "hts", "hts": "hts", "junction": "jct", "jct": "jct", "crossing": "xing",
    "xing": "xing", "extension": "ext", "ext": "ext", "ridge": "rdg", "rdg": "rdg",
    "valley": "vly", "vly": "vly", "view": "vw", "vw": "vw", "village": "vlg", "vlg": "vlg",
    "run": "run", "path": "path", "pass": "pass", "row": "row", "walk": "walk",
    "creek": "crk", "crk": "crk", "lake": "lk", "lk": "lk", "park": "park", "pk": "park",
    "plaza": "plz", "plz": "plz", "station": "sta", "sta": "sta", "estate": "est",
    "estates": "est", "est": "est", "gardens": "gdns", "garden": "gdns", "gdns": "gdns",
    "grove": "grv", "grv": "grv", "harbor": "hbr", "hbr": "hbr", "island": "is",
    "meadows": "mdws", "mdws": "mdws", "orchard": "orch", "orch": "orch",
    "spring": "spg", "springs": "spgs", "spg": "spg", "spgs": "spgs",
    # directions
    "north": "n", "n": "n", "south": "s", "s": "s", "east": "e", "e": "e", "west": "w",
    "w": "w", "northeast": "ne", "ne": "ne", "northwest": "nw", "nw": "nw",
    "southeast": "se", "se": "se", "southwest": "sw", "sw": "sw",
    # unit designators
    "suite": "ste", "ste": "ste", "apartment": "apt", "apt": "apt", "appt": "apt",
    "unit": "unit", "floor": "fl", "fl": "fl", "flr": "fl", "building": "bldg",
    "bldg": "bldg", "bldng": "bldg", "room": "rm", "rm": "rm", "department": "dept",
    "dept": "dept", "box": "box", "po": "po",
    # saint (FR/US)
    "saint": "st", "sainte": "ste",
    # French street types
    "rue": "rue", "r": "rue", "allee": "allee", "alle": "allee", "all": "allee",
    "alee": "allee", "impasse": "imp", "imp": "imp", "chemin": "chem", "chem": "chem",
    "ch": "chem", "route": "rte", "rte": "rte", "faubourg": "fbg", "fbg": "fbg",
    "quai": "quai", "qu": "quai", "cours": "crs", "crs": "crs", "residence": "res",
    "res": "res", "lotissement": "lot", "lot": "lot", "hameau": "ham", "ham": "ham",
    "passage": "pass", "pas": "pass", "voie": "voie", "cite": "cite", "carrefour": "carr",
    "carr": "carr", "rond": "rond", "zone": "zone", "za": "za", "zi": "zi",
    "bis": "bis", "ter_": "ter",
    # India
    "nagar": "nagar", "ngr": "nagar", "marg": "marg", "mg": "marg", "sector": "sec",
    "sec": "sec", "sect": "sec", "opp": "opp", "opposite": "opp", "near": "nr", "nr": "nr",
    "behind": "bh", "bh": "bh", "colony": "colony", "col": "colony", "district": "dist",
    "dist": "dist", "distt": "dist", "tehsil": "teh", "teh": "teh", "taluka": "tq",
    "tq": "tq", "taluk": "tq", "village": "vill", "vill": "vill", "vpo": "vpo",
    "post": "po", "block": "blk", "blk": "blk", "phase": "ph", "ph": "ph", "industrial": "ind",
    "ind": "ind", "indl": "ind", "area": "area", "estate_": "est", "chowk": "chowk",
    "apartments": "apts", "apts": "apts", "appartment": "apts", "complex": "cmplx",
    "cmplx": "cmplx", "tower": "twr", "towers": "twr", "twr": "twr", "society": "soc",
    "soc": "soc", "housing": "hsg", "hsg": "hsg", "cooperative": "coop", "coop": "coop",
    "ground": "gr", "gr": "gr", "first": "1st", "second": "2nd", "third": "3rd",
    "fourth": "4th", "fifth": "5th", "sixth": "6th", "seventh": "7th", "eighth": "8th",
    "ninth": "9th", "tenth": "10th", "eleventh": "11th", "twelfth": "12th",
    "thirteenth": "13th", "fourteenth": "14th", "fifteenth": "15th", "sixteenth": "16th",
    "seventeenth": "17th", "eighteenth": "18th", "nineteenth": "19th", "twentieth": "20th", "main": "main", "cross": "crs_",
    "city": "city", "cty": "city", "county": "cnty", "cnty": "cnty", "township": "twp",
    "twp": "twp",
}

# Words that only mark a number / unit and carry no identity themselves.
ADDR_MARKERS = {
    "no", "num", "number", "h", "hno", "house", "door", "dno", "plot", "flat", "shop",
    "office", "off", "khasra", "kh", "survey", "sy", "sno", "ward", "gat", "unit", "ste",
    "apt", "fl", "bldg", "rm", "dept", "box", "po", "null", "na", "nan", "none",
    "c", "o", "co", "care", "of", "the", "and", "de", "du", "des", "la", "le", "les", "d", "l",
}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "district of columbia": "dc", "puerto rico": "pr",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn",
    "telangana": "tg", "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk",
    "uttaranchal": "uk", "west bengal": "wb", "delhi": "dl", "nct of delhi": "dl",
    "jammu and kashmir": "jk", "jammu kashmir": "jk", "ladakh": "la", "chandigarh": "ch",
    "puducherry": "py", "pondicherry": "py", "dadra and nagar haveli": "dn",
    "daman and diu": "dd", "lakshadweep": "ld", "andaman and nicobar islands": "an",
}


# --------------------------------------------------------------------------
# Indic transliteration
# --------------------------------------------------------------------------
def make_transliterator(tok_map: Dict[str, str], phrase_map: Dict[str, str]):
    """Return a python function that transliterates Indic tokens in a string."""
    phrases = sorted(phrase_map.items(), key=lambda kv: -len(kv[0]))

    def tr(s: str) -> str:
        if s is None or not _INDIC_PY.search(s):
            return s
        for k, v in phrases:
            if k in s:
                s = s.replace(k, " " + v + " ")
        if not _INDIC_PY.search(s):
            return s
        out = []
        for t in s.split():
            if _INDIC_PY.search(t):
                core = t.strip(".,()[]")
                m = tok_map.get(t) or tok_map.get(core)
                out.append(m if m is not None else t)   # unseen word: keep (cannot match Latin S1 anyway)
            else:
                out.append(t)
        return " ".join(out)

    return tr


def translit_series(s: pl.Series, tr) -> pl.Series:
    """Apply `tr` only to rows that contain Indic characters (fast path)."""
    mask = s.str.contains(INDIC_RE).fill_null(False)
    if not mask.any():
        return s
    idx = mask.arg_true()
    vals = s.gather(idx).to_list()
    new = pl.Series(s.name, [tr(v) for v in vals], dtype=pl.String)
    return s.scatter(idx, new)


# --------------------------------------------------------------------------
# Polars expressions
# --------------------------------------------------------------------------
def _basic(e: pl.Expr) -> pl.Expr:
    return (
        e.fill_null("")
        .str.normalize("NFKD")
        .str.replace_all(r"\p{M}", "")
        .str.to_lowercase()
    )


def _replace_tokens(e: pl.Expr, mapping: Dict[str, str]) -> pl.Expr:
    """Whole-token replacement via Aho-Corasick on a double-space padded string."""
    pats = [" " + k.replace(" ", "  ") + " " for k in mapping]
    reps = [f" {v} " if v else " " for v in mapping.values()]
    padded = pl.lit("  ") + e.str.replace_all(" ", "  ") + pl.lit("  ")
    return padded.str.replace_many(pats, reps).str.replace_all(r"\s+", " ").str.strip_chars()


DBA_RE = (
    r"^(.*?)\s*\b(?:doing business as|d\s*[/.]?\s*b\s*[/.]?\s*a\.?|formerly known as|formerly|nee|"
    r"f\s*[/.]?\s*k\s*[/.]?\s*a\.?|a\s*/\s*k\s*/\s*a|aka|t\s*/\s*a|trading as|operating as)(?:\s+|$)(.*)$"
)
DOMAIN_RE = r"(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:co\.in|com|net|org|in|co|fr|us|biz|info|io|shop|store)\b"


def name_exprs(col: str, canon_map: Dict[str, str]) -> list[pl.Expr]:
    """Build normalised-name columns.

    Returns expressions producing:
      n_full   – cleaned full name (legal forms canonicalised, noise removed)
      n_core   – n_full without legal forms / stopwords
      n_alt    – the part before a DBA/formerly marker ('' if none)
      n_isdom  – name was a web domain / handle
      n_hasdba – DBA marker present
    """
    base = _basic(pl.col(col))
    base = base.str.replace_all(r"\(\s*id\s*:?\s*\d+\s*\)", " ")
    base = base.str.replace_all(r"#\s*\d+", " ")
    base = base.str.replace_all(r"\b\d{7,}\b", " ")
    is_dom = base.str.contains(DOMAIN_RE) | base.str.contains(r"(^|\s)[@#][a-z][a-z0-9_]+")
    base = base.str.replace_all(DOMAIN_RE, " $1 ")
    base = base.str.replace_all(r"@([a-z0-9_]+)", " $1 ")
    has_dba = base.str.contains(DBA_RE)
    main = pl.when(has_dba).then(base.str.replace(DBA_RE, "$2")).otherwise(base)
    alt = pl.when(has_dba).then(base.str.replace(DBA_RE, "$1")).otherwise(pl.lit(""))

    def clean(e: pl.Expr) -> pl.Expr:
        e = e.str.replace_all("&", " and ")
        e = e.str.replace_all(r"['’`]", "")
        e = e.str.replace_all(r"\.", "")
        e = e.str.replace_all(r"[^a-z0-9 ]", " ")
        e = e.str.replace_all(r"\s+", " ").str.strip_chars()
        # OCR-style confusions inside alphanumeric tokens (hea1th, c0rp, 5ervices, 8rothers)
        e = e.str.split(" ").list.eval(
            pl.when(pl.element().str.contains(r"[a-z]") & pl.element().str.contains(r"[0-9]"))
            .then(pl.element().str.replace_many(list(OCR_MAP), list(OCR_MAP.values())))
            .otherwise(pl.element())
        ).list.join(" ")
        e = e.str.replace_all(r"\bln(\w)", "in$1")   # lndia -> india, lnc -> inc
        m = dict(LEGAL_CANON)
        m.update(canon_map)
        return _replace_tokens(e, m)

    full = clean(main)
    core = _replace_tokens(full, {t: "" for t in (LEGAL_TOKENS | STOPWORDS | HONORIFICS | NAME_NOISE)})
    return [
        full.alias("n_full"),
        core.alias("n_core"),
        clean(alt).alias("n_alt"),
        is_dom.alias("n_isdom"),
        has_dba.alias("n_hasdba"),
    ]


def addr_exprs(col: str) -> list[pl.Expr]:
    """Build normalised-address columns.

      a_full  – canonicalised address tokens (state names -> codes, abbreviations expanded)
      a_nums  – list of numeric tokens (leading zeros stripped) in order of appearance
      a_words – list of alphabetic tokens, markers removed
    """
    e = _basic(pl.col(col))
    e = e.str.replace_all(r"<\s*null\s*>|\bnull\b|\bn/a\b", " ")
    # a whole comma-separated component of exactly two letters is a state/region code
    e = pl.lit(",") + e.str.replace_all(",", ",,") + pl.lit(",")
    e = e.str.replace_all(r",\s*([a-z]{2})\s*,", ", state_$1 ,")
    e = e.str.replace_all(",+", ",").str.strip_chars(", ")
    e = e.str.replace_all(r"\bstate_ts\b", "state_tg")
    e = e.str.replace_all(r"n\s*[°º]", " no ")
    e = e.str.replace_all(r"[°º#]", " ")
    e = e.str.replace_all(r"['’`]", "")
    e = e.str.replace_all(r"[^a-z0-9_ ]", " ")
    e = e.str.replace_all(r"(\d)([a-z])", "$1 $2")   # 12b -> 12 b ; 5th -> 5 th
    e = e.str.replace_all(r"([a-z])(\d)", "$1 $2")   # d21 -> d 21
    e = e.str.replace_all(r"\s+", " ").str.strip_chars()
    # re-join ordinals split above (5 th -> 5th)
    e = e.str.replace_all(r"\b(\d+) (st|nd|rd|th)\b", "${1}${2}")
    states = {**{k: v for k, v in US_STATES.items()}, **{k: v for k, v in IN_STATES.items()}}
    e = _replace_tokens(e, {k: f"state_{v}" for k, v in states.items()})
    e = _replace_tokens(e, ADDR_CANON)
    nums = e.str.extract_all(r"\b\d+").list.eval(pl.element().str.strip_chars_start("0"))
    nums = nums.list.eval(pl.element().filter(pl.element() != ""))
    words = (
        e.str.replace_all(r"\b\d+\w*\b", " ")
        .str.split(" ")
        .list.eval(pl.element().filter((pl.element().str.len_chars() > 0) & ~pl.element().is_in(list(ADDR_MARKERS))))
    )
    return [e.alias("a_full"), nums.alias("a_nums"), words.alias("a_words")]


def normalise_frame(df: pl.DataFrame, res: dict) -> pl.DataFrame:
    """Full normalisation of one source frame (entity_id, business_name, business_address, country)."""
    tr = make_transliterator(res["translit_tokens"], res["translit_phrases"])
    df = df.with_columns(
        translit_series(df["business_name"].fill_null(""), tr).alias("business_name"),
        translit_series(df["business_address"].fill_null(""), tr).alias("business_address"),
    )
    df = df.with_columns(
        name_exprs("business_name", res.get("name_canon", {}))
        + addr_exprs("business_address")
        + [pl.col("country").fill_null("").str.to_lowercase().str.strip_chars().alias("ctry")]
    )
    return df
