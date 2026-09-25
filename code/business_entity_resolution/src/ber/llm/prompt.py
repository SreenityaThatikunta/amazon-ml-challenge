"""Select-prompt builder for the LLM matcher (task T5.2). PROMPT_VERSION is
recorded with every export.

One S2/S3 query + its top-N S1 candidates -> the model answers one digit:
``1..N`` for the matching candidate or ``0`` for none. Answers are read as
next-token probabilities over the digits (a single forward pass, no text
generation).

Each candidate carries **hint tags** computed by the feature stage, because
subword tokenisers hide the exact differences that separate true matches from
lookalike distractors (eda/FINDINGS.md §3): the house-number relation
(truncated/padded = fine, shifted = suspicious), extra numbers (PO box/PMB),
the kind of name edit (substitutions vs dropped/doubled/swapped letters),
legal-form agreement, region agreement, and the scorer's probability.
"""

from __future__ import annotations

import pandas as pd

from ..features.numbers import RELATION_NAMES

PROMPT_VERSION = "select-v1"
ANSWERS = [str(i) for i in range(10)]  # "0" = none, "1".."9" = candidate index

SYSTEM = ("You match business records. Several source records describe the same real "
          "business with noise: typos, scanner errors (0/o, 1/l, 5/s), abbreviations, "
          "legal-form changes, reordered or missing address parts, truncated or zero-padded "
          "house numbers, names in Indian scripts, added brand/website names. Lookalike "
          "records of DIFFERENT businesses also exist: a name with one letter or title "
          "changed at the same address, or the same name with a house number shifted by a "
          "few, often with an extra PO box/PMB. Answer with the number of the matching "
          "candidate, or 0 if none matches.")


def _name_edit(f: pd.Series) -> str:
    if f.n_core_eq:
        return "same name"
    if f.q_is_domain:
        return "query name is a website"
    if f.n_q_is_initials or f.n_s_is_initials:
        return "initials"
    if f.n_lev <= 3:
        kind = []
        if f.n_sub:
            kind.append(f"{int(f.n_sub)} letter(s) replaced")
        if f.n_transp:
            kind.append("letters swapped")
        if f.n_lev - f.n_sub - f.n_transp > 0:
            kind.append("letters dropped/added")
        return "small edit: " + ", ".join(kind) if kind else "small edit"
    return f"name similarity {int(f.n_tsr)}/100"


def hints(f: pd.Series) -> str:
    """Compact hint string for one (query, candidate) feature row."""
    legal = ("legal form same" if f.l_eq else "legal form CONFLICT" if f.l_conflict else
             "legal form missing on one side")
    parts = [_name_edit(f), legal,
             f"house no. {RELATION_NAMES[int(f.hn_rel)]}"]
    if f.num_q_only > 0:
        parts.append(f"{int(f.num_q_only)} extra number(s) in query")
    if f.box_q and not f.box_s:
        parts.append("query has PO box/PMB")
    parts.append("same region" if f.r_eq else "region CONFLICT" if f.r_conflict else
                 "region unknown")
    parts.append(f"address similarity {int(f.a_tsr)}/100")
    parts.append(f"scorer {f.p:.2f}")
    return "; ".join(parts)


def build_prompt(query: pd.Series, cands: pd.DataFrame) -> str:
    """query: raw fields of the S2/S3 record (name_raw, addr_raw, country, source).
    cands: candidate rows (name_raw, addr_raw + feature columns + p), best first."""
    lines = [SYSTEM, "",
             f"QUERY ({query.country}, source {query.source}):",
             f"  name: {query.name_raw}",
             f"  address: {query.addr_raw or '(empty)'}", "", "CANDIDATES:"]
    for i, (_, c) in enumerate(cands.iterrows(), start=1):
        lines += [f"{i}. name: {c.name_raw}",
                  f"   address: {c.addr_raw or '(empty)'}",
                  f"   hints: {hints(c)}"]
    lines += ["", f"Answer (0-{len(cands)}):"]
    return "\n".join(lines)
