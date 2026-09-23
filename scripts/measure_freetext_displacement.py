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

RUNS = ROOT / "scripts" / "runs"
TOLERANCE = 0.005

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
            out.add(float(node))
            # A correlation of 0.9989 is routinely written as 99.89% or 0.999.
            out.add(round(float(node), 3))
            out.add(round(float(node) * 100, 2))

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

    for raw in _NUMBER.findall(text or ""):
        try:
            v = float(raw)
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
        if ident in names:
            continue
        bad_ids += 1
        if len(examples) < 6:
            examples.append(ident)

    return bad_nums, bad_ids, examples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=pathlib.Path, default=RUNS)
    ap.add_argument("--json", type=pathlib.Path, default=None)
    args = ap.parse_args()

    ev_cache: dict[str, tuple[set[float], set[str]]] = {}
    for f in args.runs.glob("evidence_*.json"):
        blob = json.loads(f.read_text(encoding="utf-8"))
        ev = blob.get("evidence", blob)
        ev_cache[f.stem.split("_")[-1]] = (
            evidence_quantities(ev),
            evidence_names(ev),
        )
    if not ev_cache:
        print(f"no evidence_*.json under {args.runs}", file=sys.stderr)
        return 1

    per_arm: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "with_prose": 0, "dirty": 0, "nums": 0, "ids": 0,
                 "chars": 0, "no_field": 0, "examples": []}
    )

    directories = [args.runs] + sorted(
        d for d in args.runs.iterdir() if d.is_dir() and d.name[:2] == "20"
    )
    for directory in directories:
        if not directory.exists():
            continue
        for f in sorted(directory.glob("*_synthetic_*.jsonl")):
            if "sweep" in f.name:
                continue
            provider = f.name.split("_")[0]
            arm = f.name.split("_synthetic_")[1].rsplit("_n", 1)[0]
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                ehash = str(rec.get("evidence_hash", ""))
                key = ehash if ehash in ev_cache else next(iter(ev_cache))
                quantities, names = ev_cache[key]

                slot = per_arm[(provider, arm)]
                slot["n"] += 1
                # Only the real field. `verdict` is not a fallback: it carries
                # prose in the unconstrained arms and an enum in the contract
                # arms, so reading it would compare two different things and
                # call the difference displacement.
                if "narration" not in rec:
                    slot["no_field"] += 1
                    continue
                prose = str(rec.get("narration") or "")
                if not prose.strip():
                    continue
                slot["with_prose"] += 1
                slot["chars"] += len(prose)
                bad_n, bad_i, ex = score_narration(prose, quantities, names)
                slot["nums"] += bad_n
                slot["ids"] += bad_i
                if bad_n or bad_i:
                    slot["dirty"] += 1
                    if len(slot["examples"]) < 3:
                        slot["examples"].append(ex[:3])

    print("Free-text displacement: ungrounded content in the unverified prose\n")
    print(f"{'provider':9s} {'arm':22s} {'N':>4s} {'prose':>6s} {'dirty':>6s} "
          f"{'rate':>7s} {'95% CI':>14s} {'nums':>5s} {'ids':>5s} {'chars':>6s}")
    missing = sum(s["no_field"] for s in per_arm.values())
    total = sum(s["n"] for s in per_arm.values())
    if missing:
        print(f"  {missing} of {total} records predate the narration fix and "
              f"carry no prose field.")
        print("  Those arms are omitted below rather than scored on `verdict`.\n")
    if missing == total:
        print("NOTHING TO MEASURE. Every record predates the fix.")
        print("Re-run the battery with the current harness, then run this again.")
        return 2

    rows = []
    for (provider, arm), s in per_arm.items():
        if not s["with_prose"]:
            continue
        rate = 100 * s["dirty"] / s["with_prose"]
        lo, hi = wilson(s["dirty"], s["with_prose"])
        rows.append((provider, arm, s, rate, lo, hi))
    rows.sort(key=lambda r: (r[0], ARM_ORDER.index(r[1]) if r[1] in ARM_ORDER else 99))
    for provider, arm, s, rate, lo, hi in rows:
        print(f"{provider:9s} {ARM_ID.get(arm, arm):22s} {s['n']:4d} "
              f"{s['with_prose']:6d} {s['dirty']:6d} {rate:6.1f}% "
              f"[{lo:5.1f},{hi:5.1f}] {s['nums']:5d} {s['ids']:5d} "
              f"{s['chars'] // max(1, s['with_prose']):6d}")

    print("\nThe comparison the reviewers asked for, on deepseek "
          "(the only provider that fails at all):")
    bare = [r for r in rows if r[0] == "deepseek" and r[1] not in ENFORCED]
    enf = [r for r in rows if r[0] == "deepseek" and r[1] in ENFORCED]
    for label, group in (("unenforced", bare), ("enforced", enf)):
        d = sum(r[2]["dirty"] for r in group)
        p = sum(r[2]["with_prose"] for r in group)
        if not p:
            continue
        lo, hi = wilson(d, p)
        print(f"  {label:12s} {d:5d}/{p:<5d} = {100*d/p:5.1f}% [{lo:.1f}, {hi:.1f}]")
    print(
        "\n  Displacement predicts the enforced arms are HIGHER. Read the two\n"
        "  intervals before concluding anything; this design sees the numeric\n"
        "  and entity half of the channel and not unsupported causal prose."
    )

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    f"{p}/{a}": {k: v for k, v in s.items() if k != "examples"}
                    for (p, a), s in per_arm.items()
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
