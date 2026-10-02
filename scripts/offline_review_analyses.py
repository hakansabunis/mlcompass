"""Three offline analyses the 2026-10-01 review asked for. No API calls.

  1. Second scorer. Every leakage record is scored by the paper's scorer
     (`score_one`) and by `definitions_scorer.py`, which shares no code with
     it, and the two are compared channel by channel.
  2. What deletion would cost. Tier B strips what is still unsound after its
     retries, and no live response was ever stripped. Here every response from
     an arm without a verifier is stripped as Tier B would strip it, and we
     count what survives: cited columns, claims, correct numbers, the anchor.
  3. A task-level slot. The profile contract admits only per-column
     statistics, so the class balance, which the profiler reports at task
     level, has no admissible slot. Here the profile responses are re-scored
     with the class shares admitted on the target column.

    python -X utf8 scripts/offline_review_analyses.py [--json benchmark/offline_review_analyses.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import definitions_scorer as ind  # noqa: E402
import reproduce_hallucination_ablation as h  # noqa: E402
from analyze_revision_runs import _evidence, _names, _numbers, _records  # noqa: E402
from unified_scoring import A9_DIRS  # noqa: E402

from mlcompass.agents.evidence_contract import (  # noqa: E402
    LEAKAGE,
    PROFILE,
    bind_leakage,
    bind_profile,
    strip_unsound,
    verify,
)

RUNS = ROOT / "scripts" / "runs"
UNVERIFIED_LEAK = {
    "layer1", "layer2", "baseline", "cautious", "expert", "helpful", "mechanical",
    "terse", "static_schema", "static_schema_noenum", "tier_a", "tier_a+strict",
    "strong_noverify",
}
UNVERIFIED_PROFILE = {"bare", "tier_a", "strong_noverify"}


def run_files(kind: str):
    for f in sorted(RUNS.rglob("*.jsonl")):
        rel = f.relative_to(RUNS).parts
        if any("superseded" in p or p == "transport_errors" for p in rel):
            continue
        if (rel[0] == "profile") != (kind == "profile"):
            continue
        if any(p in A9_DIRS for p in rel):  # the 2026-10-02 registered runs are reported apart
            continue
        ev_path = f.parent / f"evidence_{f.stem.rsplit('_e', 1)[1]}.json"
        if ev_path.exists():
            yield f, _evidence(ev_path)


# --------------------------------------------------------------------------- #
# 1. second scorer                                                            #
# --------------------------------------------------------------------------- #
def second_scorer() -> dict:
    n = 0
    agree: Counter = Counter()
    disagree: Counter = Counter()
    examples: dict[str, list] = {}
    for f, ev in run_files("leak"):
        allowed = set(h.evidence_allowed_columns(ev))
        corr, anchor = h.evidence_correlation_map(ev), h.top_candidate(ev)
        names, table = _names(ev), h.evidence_value_table(ev)
        for r in _records(f):
            n += 1
            resp = {k: r.get(k) for k in ("columns", "claims", "verdict", "omitted")}
            a = h.score_one(resp, allowed, corr, anchor, evidence_names=names, value_table=table)
            b = ind.score(r, ev)
            pairs = {
                "entity": (bool(a["entity"]), b["entity"]),
                "misfiled": (bool(a["entity_misfiled"]), b["misfiled"]),
                "outside_E": (bool(a["entity_invented"]), b["outside_E"]),
                "wrong_number": (bool(a["value"]), b["wrong_number"]),
            }
            # Omission only where the verdict is one of the four labels: the
            # first scorer also counts free-text verdicts as committing, the
            # paper's definition does not, and the second scorer follows it.
            if str(r.get("verdict") or "") in ind.COMMITTING | {"cannot_determine"}:
                pairs["omission"] = (bool(a["omission"]), b["omission"])
            for ch, (x, y) in pairs.items():
                if x == y:
                    agree[ch] += 1
                else:
                    disagree[ch] += 1
                    examples.setdefault(ch, [])
                    if len(examples[ch]) < 5:
                        examples[ch].append({"file": f"{f.parent.name}/{f.name}", "i": r.get("i"),
                                             "first": x, "second": y,
                                             "outside": b["outside_names"],
                                             "verdict": str(r.get("verdict"))[:40]})
    return {"records": n, "agree": dict(agree), "disagree": dict(disagree), "examples": examples}


# --------------------------------------------------------------------------- #
# 2. what deletion would cost                                                  #
# --------------------------------------------------------------------------- #
def _strip_stats(kind: str) -> dict:
    spec, binder, arms = (
        (LEAKAGE, bind_leakage, UNVERIFIED_LEAK) if kind == "leak"
        else (PROFILE, bind_profile, UNVERIFIED_PROFILE)
    )
    s: Counter = Counter()
    for f, ev in run_files(kind):
        bound = binder(ev)
        numbers = _numbers(ev) + [float(x) for x in bound.values.values()]  # fault 17
        for r in _records(f):
            if r.get("arm") not in arms:
                continue
            claims = [c for c in (r.get("raw_claims") or r.get("claims") or []) if isinstance(c, dict)]
            cited = [str(c) for c in (r.get("raw_columns") or r.get("columns") or [])]
            payload = {"verdict": r.get("verdict"), "columns_referenced": cited, "claims": claims}
            v = verify(payload, bound, spec)
            s["responses"] += 1
            if not (v.entity or v.value):
                continue
            s["violating"] += 1
            kept_cited, kept_claims = strip_unsound(payload, bound, spec)
            s["cited"] += len(cited)
            s["cited_kept"] += len(kept_cited)
            s["claims"] += len(claims)
            s["claims_kept"] += len(kept_claims)
            kept_ids = {id(c) for c in kept_claims}
            for c in claims:
                if id(c) in kept_ids:
                    continue
                val = c.get("value")
                if isinstance(val, (int, float)) and not isinstance(val, bool) and any(
                    abs(float(val) - x) <= bound.tolerance for x in numbers
                ):
                    s["dropped_correct_numbers"] += 1
            s["left_with_no_claim"] += bool(claims) and not kept_claims
            s["left_with_no_column"] += bool(cited) and not kept_cited
            if bound.anchor:
                before = bound.anchor in set(cited) | {str(c.get("column")) for c in claims}
                after = bound.anchor in set(kept_cited) | {str(c.get("column")) for c in kept_claims}
                s["anchor_before"] += before
                s["anchor_lost"] += before and not after
    return dict(s)


# --------------------------------------------------------------------------- #
# 3. a task-level slot for the class balance                                   #
# --------------------------------------------------------------------------- #
_CLASS_STAT = re.compile(r"class.*(balance|share|ratio|prop|pct|fraction)|(balance|share).*class")


def task_level_slot() -> dict:
    out = {}
    for f, ev in run_files("profile"):
        recs = [r for r in _records(f) if r.get("arm") in UNVERIFIED_PROFILE]
        if not recs:
            continue
        bound = bind_profile(ev)
        numbers = _numbers(ev) + [float(x) for x in bound.values.values()]  # fault 17
        shares = [float(x) for x in ((ev.get("task_hint") or {}).get("class_balance") or {}).values()]
        target = str((ev.get("target_hint") or {}).get("column") or "target")
        before = after = 0
        kinds_after: Counter = Counter()
        for r in recs:
            claims = [c for c in (r.get("raw_claims") or r.get("claims") or []) if isinstance(c, dict)]
            cited = [str(c) for c in (r.get("raw_columns") or r.get("columns") or [])]
            v = verify({"verdict": r.get("verdict"), "columns_referenced": cited, "claims": claims},
                       bound, PROFILE)
            before += bool(v.entity or v.value)
            kinds = set()
            if v.entity:
                kinds.add("entity")
            for c in claims:
                key = (str(c.get("column", "")), str(c.get("statistic", "")))
                val = c.get("value")
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    continue
                if key in bound.values:
                    if abs(float(val) - bound.values[key]) > bound.tolerance:
                        kinds.add("wrong number")
                    continue
                if key[0] == target and _CLASS_STAT.search(key[1].lower()):
                    if not any(abs(float(val) - x) <= bound.tolerance for x in shares):
                        kinds.add("wrong class share")
                    continue
                real = any(abs(float(val) - x) <= bound.tolerance for x in numbers)
                kinds.add("real, misplaced" if real else "not in E")
            after += bool(kinds)
            for k in kinds:
                kinds_after[k] += 1
        r0 = recs[0]
        out[f"{f.parent.name}/{f.name}"] = {
            "arm": r0.get("arm_id"), "names": r0.get("frame_names"), "model": r0.get("model"),
            "n": len(recs), "violating_before": before, "violating_with_slot": after,
            "kinds_with_slot": dict(kinds_after),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    res = {"second_scorer": second_scorer(),
           "strip": {"leakage": _strip_stats("leak"), "profile": _strip_stats("profile")},
           "task_level_slot": task_level_slot()}
    s = res["second_scorer"]
    print(f"1. second scorer on {s['records']} leakage records")
    for ch in s["agree"]:
        tot = s["agree"][ch] + s["disagree"].get(ch, 0)
        print(f"   {ch:13s} agree {s['agree'][ch]}/{tot}")
    for ch, ex in s["examples"].items():
        print(f"   disagreement on {ch}: {ex[:3]}")
    print("2. stripping every unverified response")
    for k, v in res["strip"].items():
        print(f"   {k}: {v}")
    print("3. task-level slot for the class balance (unverified profile arms)")
    for k, v in res["task_level_slot"].items():
        print(f"   {v['arm']:18s} {str(v['names']):9s} {v['model']:14s} N={v['n']:3d} "
              f"before {v['violating_before']:3d} with slot {v['violating_with_slot']:3d} "
              f"{v['kinds_with_slot']}")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
