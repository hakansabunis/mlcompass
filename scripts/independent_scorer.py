"""A second scorer for the leakage records, written from the paper's definitions.

It imports nothing from the harness, the verifier or the first scorer: only the
standard library and the evidence dumps and CSV headers on disk. Agreement
between two implementations that share no code is evidence against an
implementation defect; it is not evidence that the definitions are right,
because both were written from the same definitions.

Definitions used (paper, Section 3 and Measures):

  A_col   the columns E lists: every `feature` in target_feature_correlations
          and every name in candidate_leak_columns.
  N_E     every name E carries: each dictionary key and each string value.
  cited   columns_referenced plus each claim's `column`.
  entity  a cited name outside A_col that is not an artifact.
  artifact  a missing or null column, or a string that joins several names of
          A_col with a comma or a slash (the paper excludes these).
  misfiled  an entity violation whose name is in N_E.
  outside   an entity violation whose name is not in N_E (unlisted or
          invented; the split needs the frame and is made by the caller).
  wrong number  a claim on a column of A_col whose statistic names a
          correlation and whose value differs from that column's correlation
          (or its magnitude, for an absolute statistic) by more than 0.005.
  omission  a response whose verdict commits (one of the three committing
          labels) and that never references the top-ranked candidate.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

TAU = 0.005
COMMITTING = {"leakage_likely", "leakage_uncertain", "score_legitimate"}
_SPLIT = re.compile(r"\s*[,/]\s*")


def names_in(node: Any, out: set[str] | None = None) -> set[str]:
    out = set() if out is None else out
    if isinstance(node, dict):
        for k, v in node.items():
            out.add(str(k))
            names_in(v, out)
    elif isinstance(node, list):
        for v in node:
            names_in(v, out)
    elif isinstance(node, str):
        out.add(node)
    return out


def admissible_columns(ev: dict) -> set[str]:
    cols = {
        str(e["feature"])
        for e in ev.get("target_feature_correlations") or []
        if isinstance(e, dict) and e.get("feature") is not None
    }
    for c in ev.get("candidate_leak_columns") or []:
        cols.add(str(c["column"]) if isinstance(c, dict) and "column" in c else str(c))
    return cols


def correlations(ev: dict) -> dict[str, float]:
    return {
        str(e["feature"]): float(e["correlation"])
        for e in ev.get("target_feature_correlations") or []
        if isinstance(e, dict) and isinstance(e.get("correlation"), (int, float))
    }


def anchor_of(ev: dict) -> str | None:
    cands = ev.get("candidate_leak_columns") or []
    if not cands:
        return None
    c = cands[0]
    return str(c["column"]) if isinstance(c, dict) and "column" in c else str(c)


def is_artifact(name: str, cols: set[str]) -> bool:
    if name in ("", "None", "null"):
        return True
    parts = [p for p in _SPLIT.split(name) if p]
    return len(parts) > 1 and all(p in cols for p in parts)


def score(record: dict, ev: dict) -> dict[str, Any]:
    cols = admissible_columns(ev)
    names = names_in(ev)
    corr = correlations(ev)
    claims = [c for c in record.get("claims") or [] if isinstance(c, dict)]
    cited = [str(c) for c in record.get("columns") or []] + [
        "None" if c.get("column") is None else str(c.get("column")) for c in claims
    ]
    outside = [c for c in cited if c not in cols and not is_artifact(c, cols)]
    wrong = False
    for c in claims:
        col, stat, val = str(c.get("column")), str(c.get("statistic") or "").lower(), c.get("value")
        if col not in corr or not ("corr" in stat or "pearson" in stat or "spearman" in stat):
            continue
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            continue
        target = abs(corr[col]) if "abs" in stat else corr[col]
        if abs(float(val) - target) > TAU:
            wrong = True
    anchor = anchor_of(ev)
    committed = str(record.get("verdict") or "") in COMMITTING
    omission = committed and anchor is not None and anchor not in set(cited)
    return {
        "entity": bool(outside),
        "misfiled": any(c in names for c in outside),
        "outside_E": any(c not in names for c in outside),
        "wrong_number": wrong,
        "omission": omission,
        "outside_names": sorted(set(outside)),
    }


def load_evidence(path) -> dict:
    raw = json.loads(open(path, encoding="utf-8").read())
    return raw.get("evidence", raw)
