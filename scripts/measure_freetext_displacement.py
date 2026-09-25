"""Does enforcement push unfaithful content into the free-text field?

The guarantee covers the structured channels. The narration string is handed to
the user marked unverified and nothing checks it, and three separate reviewers
have now named the same worry: if Tier B strips a claim, does the model just say
it in prose instead?

The paper has always answered "we did not measure that". Writing this script
found out why, and the reason is worse than the omission: **the narration was
never recorded.** The harness asks for it, the schema marks it required, and
`_normalize` dropped it before the record was written. So every response logged
before 2026-09-18 carries the structured payload and no trace of the one
channel the guarantee does not cover.

That is a hole in the replication package, not just an unrun experiment, and it
is reported as one. `_normalize` now preserves the field, so a fresh battery
answers the question; this script is the analysis waiting for it. Run against
records that predate the fix it refuses rather than substituting `verdict`,
which in the unconstrained arms happens to hold prose and in the contract arms
holds a schema-pinned enum -- comparing those two would manufacture a 42 % vs
76 % "displacement effect" out of a field-naming difference. It did, on the
first draft of this file.

The measure. For each response, extract every number in the narration string
and every name that looks like an identifier, then classify against E:

  grounded      the value matches a measured quantity in E within tolerance,
                or the name is in A_E
  ungrounded    it is neither, and is not one of the shapes that are not
                claims at all (percentages of the model's own confidence,
                thresholds quoted from the prompt, row counts, years, small
                integers used as counts)

The comparison that matters is between arms, not the absolute rate. If
enforcement displaces content, the enforced arms should carry MORE ungrounded
prose than the bare arm, because the structured route is closed to them. If
they carry the same or less, displacement is not happening at the scale this
design can see.

What this cannot do. It cannot judge whether a sentence asserts a causal story
the evidence does not support -- "log_target_v2 causes the target" contains no
fabricated number and no fabricated name. That is the part of the gap that
needs human raters, and it stays open. This measures the numeric and entity
half, which is the half a deterministic check can reach.

    python scripts/measure_freetext_displacement.py
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys
from collections import defaultdict
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

sys.path.insert(0, str(ROOT / "src"))

from mlcompass.agents.evidence_contract import (  # noqa: E402
    CONFIDENCE_VALUES,
    LEAKAGE_VERDICTS,
)

RUNS = ROOT / "scripts" / "runs"
TOLERANCE = 0.005

# The response schema's own vocabulary. A narration that writes
# "leakage_likely" is restating its verdict label, not naming something in E;
# counting it made the contract arms look dirtier only because their schema
# shows the model the label (20 and 23 responses, against 0 on the bare arm).
_SCHEMA_VOCABULARY = set(LEAKAGE_VERDICTS) | set(CONFIDENCE_VALUES) | {
    "columns_referenced",
    "primary_hypothesis",
}

# Arm ids, in the order the manuscript's tables use.
ARM_ORDER = [
    "layer1",
    "guardrails_stock",
    "static_schema_noenum",
    "guardrails_choices",
    "guardrails_tierb",
    "layer3_bare",
    "layer3_stress",
    "static_schema",
    "tier_a",
    "stress_mech",
]
ARM_ID = {
    "layer1": "A-L1",
    "guardrails_stock": "A-GR-STOCK",
    "static_schema_noenum": "A-STRICT-STATIC",
    "guardrails_choices": "A-GR-CHOICES",
    "guardrails_tierb": "A-GR-OURS",
    "layer3_bare": "A-CONTRACT",
    "layer3_stress": "A-STRESS",
    "static_schema": "A-STATIC-ENUM-STALE",
    "tier_a": "A-TIER-A-ONLY",
    "stress_mech": "A-CONTRACT-WORST",
}
# Arms where Tier B actually deletes something. The displacement hypothesis is
# about *stripping*: if an unsound claim is removed from the structured fields,
# does the model put it in the prose instead? An arm that only constrains the
# schema never strips, so it belongs with the unenforced side of that question
# however much it lowers the headline rate -- `tier_a`, `static_schema` and
# `static_schema_noenum` are schema-only and are grouped accordingly.
ENFORCED = {
    "guardrails_choices",
    "guardrails_tierb",
    "layer3_bare",
    "layer3_stress",
    "stress_mech",
}

_NUMBER = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)(?![\w.])")
_IDENT = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")

# Numbers that are not claims about the evidence. Kept deliberately tight: a
# generous exclusion list would manufacture the null this section is testing
# for, so anything excluded here is either a literal from the prompt or a
# shape that cannot be a measured quantity.
_PROMPT_LITERALS = {0.999, 0.995, 0.9, 0.95, 0.99, 2.0, 1.0, 0.0, 100.0, 0.005}


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def evidence_quantities(evidence: dict[str, Any]) -> set[float]:
    """Every number E carries, at any depth. The reference set for prose."""
    out: set[float] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            # Stored as magnitudes: prose writes "|r| <= 0.0405" for a
            # correlation of -0.0405, and the sign is not what the sentence
            # claims. Comparing signed values flagged every such restatement.
            q = abs(float(node))
            out.add(q)
            # A correlation of 0.9989 is routinely written as 99.89% or 0.999.
            out.add(round(q, 3))
            out.add(round(q * 100, 2))

    walk(evidence)
    return out


def evidence_names(evidence: dict[str, Any]) -> set[str]:
    out: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                out.add(str(k))
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, str):
            out.add(node)

    walk(evidence)
    return out


def score_narration(
    text: str, quantities: set[float], names: set[str]
) -> tuple[int, int, list[str]]:
    """Return (ungrounded numbers, ungrounded identifiers, examples)."""
    bad_nums = 0
    bad_ids = 0
    examples: list[str] = []

    # U+2212 is what models write for a minus sign; the regex reads only "-".
    text = (text or "").replace("\u2212", "-")
    # "1,200 rows" is one number. Without this the regex read "1" and "200",
    # and "200" scored as ungrounded in most crowded-instance narrations.
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)
    for raw in _NUMBER.findall(text):
        try:
            v = abs(float(raw))
        except ValueError:
            continue
        if v in _PROMPT_LITERALS:
            continue
        # Small integers are counts and ordinals, not measurements.
        if v == int(v) and abs(v) <= 20:
            continue
        if any(abs(v - q) <= TOLERANCE for q in quantities):
            continue
        bad_nums += 1
        if len(examples) < 4:
            examples.append(raw)

    for ident in set(_IDENT.findall(text or "")):
        if ident in names or ident in _SCHEMA_VOCABULARY:
            continue
        bad_ids += 1
        if len(examples) < 6:
            examples.append(ident)

    return bad_nums, bad_ids, examples


def _prompt_vocabulary() -> set[str]:
    """Identifiers the system prompts themselves give the model.

    A narration that repeats ``y_true`` or ``candidate_leak_columns`` from its
    own instructions is not naming anything in E. The first pass on the
    2026-09-24 records scored those as ungrounded and put the shipped-prompt arms
    at 91 and 97.5 % (the fourteenth instrument fault).
    """
    from reproduce_hallucination_ablation import (  # noqa: PLC0415
        LIVE_SYSTEM_PROMPT_BARE,
        LIVE_SYSTEM_PROMPT_RULES,
        SWEEP_BARE_VARIANTS,
    )

    from mlcompass.agents.leakage_investigator import LEAKAGE_BOUND_PROMPT  # noqa: PLC0415

    texts = [LIVE_SYSTEM_PROMPT_BARE, LIVE_SYSTEM_PROMPT_RULES, LEAKAGE_BOUND_PROMPT]
    texts += [t for _, t in SWEEP_BARE_VARIANTS]
    return {w for t in texts for w in _IDENT.findall(t)}


# Arms whose verifier can send a claim back. Everything else is compared as the
# non-rejecting side, schema-only arms included: the hypothesis concerns what
# happens to content a verifier rejects.
REJECTING = {
    "guardrails_choices", "guardrails_tierb", "layer3", "layer3_bare",
    "layer3_stress", "layer3_stress_generic", "stress_mech", "stress_mech-strict",
}


def _cell(name: str) -> tuple[str, str] | None:
    """(task, arm) from a record file name; None for sweep and non-leakage files."""
    if "_synthetic_" not in name or "sweep" in name:
        return None
    rest = name.split("_synthetic_", 1)[1].rsplit("_n", 1)[0]
    if rest.startswith("crowded_"):
        return "crowded", rest[len("crowded_"):]
    return "reference", rest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=pathlib.Path, default=RUNS)
    ap.add_argument("--json", type=pathlib.Path, default=None)
    args = ap.parse_args()

    global _SCHEMA_VOCABULARY
    _SCHEMA_VOCABULARY = _SCHEMA_VOCABULARY | _prompt_vocabulary()

    # Evidence from every directory, keyed by its own hash. The first version
    # read only the top-level dumps and scored a record whose hash it did not
    # hold against whichever dump came first, so the crowded instance's prose
    # was checked against the reference task's names.
    ev_cache: dict[str, tuple[set[float], set[str]]] = {}
    for f in args.runs.rglob("evidence_*.json"):
        if "superseded" in str(f):
            continue
        blob = json.loads(f.read_text(encoding="utf-8"))
        ev = blob.get("evidence", blob)
        ev_cache[f.stem.split("_")[-1]] = (evidence_quantities(ev), evidence_names(ev))

    cells: dict[tuple[str, str, str, str], dict[str, Any]] = defaultdict(
        lambda: {"with_prose": 0, "dirty": 0, "retried": 0, "retried_dirty": 0,
                 "first": 0, "first_dirty": 0, "no_evidence": 0}
    )
    directories = [args.runs] + sorted(
        d for d in args.runs.iterdir() if d.is_dir() and d.name[:2] == "20"
    )
    for directory in directories:
        for f in sorted(directory.glob("*.jsonl")):
            cell = _cell(f.name)
            if cell is None:
                continue
            provider = f.name.split("_")[0]
            key = (provider, cell[0], cell[1], directory.name)
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("error") or "narration" not in rec:
                    continue
                prose = str(rec.get("narration") or "")
                if not prose.strip():
                    continue
                ev = ev_cache.get(str(rec.get("evidence_hash", "")))
                slot = cells[key]
                if ev is None:
                    slot["no_evidence"] += 1
                    continue
                bad_n, bad_i, _ = score_narration(prose, *ev)
                dirty = bool(bad_n or bad_i)
                slot["with_prose"] += 1
                slot["dirty"] += dirty
                if int(rec.get("rejections") or 0) > 0:
                    slot["retried"] += 1
                    slot["retried_dirty"] += dirty
                else:
                    slot["first"] += 1
                    slot["first_dirty"] += dirty

    print("Ungrounded content in the unverified narration, every prose-bearing cell\n")
    print(f"{'task':9s} {'arm':22s} {'run':22s} {'prose':>5s} {'dirty':>5s} {'rate':>7s} {'95% CI':>14s}")
    for (provider, task, arm, run), s in sorted(cells.items()):
        if provider != "deepseek" or not s["with_prose"]:
            continue
        lo, hi = wilson(s["dirty"], s["with_prose"])
        side = "R" if arm in REJECTING else " "
        print(f"{task:9s} {side} {ARM_ID.get(arm, arm):20s} {run:22s} {s['with_prose']:5d} "
              f"{s['dirty']:5d} {100 * s['dirty'] / s['with_prose']:6.1f}% [{lo:5.1f},{hi:5.1f}]")
    skipped = sum(s["no_evidence"] for s in cells.values())
    if skipped:
        print(f"\n  {skipped} narrations skipped: no evidence dump for their hash.")

    summary: dict[str, Any] = {}
    print("\nRejecting (R) against non-rejecting arms, deepseek, per task instance:")
    for task in ("reference", "crowded"):
        groups = {"rejecting": [0, 0], "non-rejecting": [0, 0], "retried": [0, 0], "first-pass": [0, 0]}
        for (provider, t, arm, _run), s in cells.items():
            if provider != "deepseek" or t != task:
                continue
            g = "rejecting" if arm in REJECTING else "non-rejecting"
            groups[g][0] += s["dirty"]
            groups[g][1] += s["with_prose"]
            if arm in REJECTING:
                groups["retried"][0] += s["retried_dirty"]
                groups["retried"][1] += s["retried"]
                groups["first-pass"][0] += s["first_dirty"]
                groups["first-pass"][1] += s["first"]
        summary[task] = {}
        for label, (d, n) in groups.items():
            if not n:
                continue
            lo, hi = wilson(d, n)
            summary[task][label] = {"dirty": d, "n": n, "rate": round(100 * d / n, 1),
                                    "ci": [round(lo, 1), round(hi, 1)]}
            print(f"  {task:9s} {label:14s} {d:5d}/{n:<5d} = {100 * d / n:5.1f}% [{lo:.1f}, {hi:.1f}]")
    print(
        "\n  Displacement predicts the rejecting arms read HIGHER. The instrument\n"
        "  sees numbers and identifiers only: a misfiled name is in E and counts\n"
        "  as grounded, and unsupported causal prose is invisible to it."
    )

    if args.json:
        args.json.write_text(json.dumps({
            "cells": {f"{p}/{t}/{a}/{r}": s for (p, t, a, r), s in cells.items()},
            "summary": summary,
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
