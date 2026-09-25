"""Score the 2026-09-24 revision runs with the current scorer. Offline.

The review of the TSE manuscript asked for four controls, each a small N=200
battery on deepseek-chat (review items P0-3, P0-4, P0-7):

  2026-09-24_revision/   reference task: A-L1, A-L2 (the shipped faithfulness
                         prompt, open schema, no verification), A-L3-SHIPPED,
                         A-STRESS, A-STRESS-GENERIC; A-STRICT-ENUM with
                         --strict; and the synthetic-crowded instance, on
                         which the author-time enum is genuinely stale.
  2026-09-24_described/  A-L1 and A-L2 with field descriptions on the open
                         schema (columns_referenced: column names only).
  profile/2026-09-24_natural/  the RQ6 frame under distinct column names.

Every record is re-scored here from its raw fields against the evidence dump
carrying its own hash, so the numbers do not depend on the scorer version that
wrote the `scored` field at run time.

    python -X utf8 scripts/analyze_revision_runs.py [--json benchmark/revision_runs.json]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import reproduce_hallucination_ablation as h  # noqa: E402
from mlcompass.agents.evidence_contract import bind_profile  # noqa: E402

RUNS = ROOT / "scripts" / "runs"
LEAKAGE_DIRS = [RUNS / "2026-09-24_revision", RUNS / "2026-09-24_described",
                RUNS / "2026-09-25_described_neutral", RUNS / "2026-09-25_rules_prompt"]
PROFILE_DIRS = [RUNS / "profile", RUNS / "profile" / "2026-09-24_natural"]


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - r), 100 * min(1.0, c + r))


def _evidence(path: pathlib.Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw.get("evidence", raw)


def _records(path: pathlib.Path) -> list[dict]:
    """Valid responses only: a transport error is not data (plan A3.9b). The
    first pass of this script counted 402 'Insufficient Balance' records as
    clean responses and printed 0/200 for arms that never ran."""
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in recs if not r.get("error")]


def _names(node, out=None) -> set[str]:
    """Every key and string in E: what separates misfiled from invented."""
    out = set() if out is None else out
    if isinstance(node, dict):
        for k, v in node.items():
            out.add(str(k))
            _names(v, out)
    elif isinstance(node, list):
        for v in node:
            _names(v, out)
    elif isinstance(node, str):
        out.add(node)
    return out


def leakage_cell(f: pathlib.Path) -> dict:
    ehash = f.stem.rsplit("_e", 1)[1]
    ev = _evidence(f.parent / f"evidence_{ehash}.json")
    allowed = set(h.evidence_allowed_columns(ev))
    corr = h.evidence_correlation_map(ev)
    anchor = h.top_candidate(ev)
    names = _names(ev)
    table = h.evidence_value_table(ev)
    recs = _records(f)
    n = len(recs)
    ent = inv = mis = val = om = rej = strip = 0
    tokens: Counter = Counter()
    abst = 0
    for r in recs:
        resp = {k: r.get(k) for k in ("columns", "claims", "verdict", "omitted")}
        fl = h.score_one(resp, allowed, corr, anchor, evidence_names=names, value_table=table)
        ent += bool(fl.get("entity"))
        inv += bool(fl.get("entity_invented"))
        mis += bool(fl.get("entity_misfiled"))
        val += bool(fl.get("value"))
        om += bool(fl.get("omission"))
        rej += int(r.get("rejections") or 0) > 0
        strip += int(r.get("rejections") or 0) >= 3
        abst += str(r.get("verdict") or "") == "cannot_determine"
        cited = list(r.get("columns") or []) + [
            str(c.get("column", "")) for c in (r.get("claims") or []) if isinstance(c, dict)
        ]
        for c in set(cited):
            if c not in allowed and not h.is_scorer_artifact(c, allowed):
                tokens[c] += 1
    first = recs[0] if recs else {}
    return {
        "file": f"{f.parent.name}/{f.name}",
        "arm_id": first.get("arm_id"),
        "task": first.get("task"),
        "strict": first.get("strict"),
        "n": n,
        "entity": ent,
        "entity_ci": wilson(ent, n),
        "invented": inv,
        "misfiled": mis,
        "value": val,
        "omission": om,
        "abstained": abst,
        "rejected_once_or_more": rej,
        "stripped": strip,
        "top_tokens": tokens.most_common(6),
        "anchor": anchor,
        "anchor_in_allowed": anchor in allowed,
    }


def _numbers(node, out=None) -> list[float]:
    out = [] if out is None else out
    if isinstance(node, dict):
        for v in node.values():
            _numbers(v, out)
    elif isinstance(node, list):
        for v in node:
            _numbers(v, out)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        out.append(float(node))
    return out


def profile_cell(f: pathlib.Path) -> dict:
    """Classify every value-channel violation the verifier reports.

    wrong number     -- the pair is admissible and the number is off
    real, misplaced  -- the pair is not admissible, but the number is one E
                        carries somewhere (within tau): a correct quantity in a
                        slot the binder does not admit
    not in E         -- neither the pair nor the number exists in E
    The first pass of the manuscript called every class_balance claim an
    invented quantity; E carries it in task_hint and every one was correct.
    """
    ehash = f.stem.rsplit("_e", 1)[1]
    ev = _evidence(f.parent / f"evidence_{ehash}.json")
    bound = bind_profile(ev)
    numbers = _numbers(ev)
    recs = _records(f)
    n = len(recs)
    resp: Counter = Counter()
    items: Counter = Counter()
    from mlcompass.agents.evidence_contract import PROFILE, verify  # noqa: PLC0415

    for r in recs:
        claims = r.get("raw_claims") or r.get("claims") or []
        payload = {"verdict": r.get("verdict"),
                   "columns_referenced": r.get("raw_columns") or r.get("columns"),
                   "claims": claims}
        v = verify(payload, bound, PROFILE)
        kinds = set()
        for c in claims:
            key = (str(c.get("column", "")), str(c.get("statistic", "")))
            val = c.get("value")
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                continue
            if key in bound.values:
                if abs(float(val) - bound.values[key]) > bound.tolerance:
                    kinds.add("wrong number")
                    items["wrong number"] += 1
                continue
            real = any(abs(float(val) - x) <= bound.tolerance for x in numbers)
            k = "real, misplaced" if real else "not in E"
            kinds.add(k)
            items[k] += 1
        if v.entity:
            kinds.add("entity")
        for k in kinds:
            resp[k] += 1
        resp["any"] += bool(kinds)
    return {
        "file": f"{f.parent.name}/{f.name}",
        "arm": (recs[0].get("arm") if recs else None),
        "n": n,
        "responses": dict(resp),
        "items": dict(items),
        "any_ci": wilson(resp["any"], n),
        "wrong_number_ci": wilson(resp["wrong number"], n),
        "anchor": bound.anchor,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    out = {"leakage": [], "profile": []}
    for d in LEAKAGE_DIRS:
        for f in sorted(d.glob("deepseek_*.jsonl")):
            out["leakage"].append(leakage_cell(f))
    for d in PROFILE_DIRS:
        for f in sorted(d.glob("*_profile_*_n200_*.jsonl")):
            out["profile"].append(profile_cell(f))

    print(f"{'cell':62s} {'N':>4s} {'ent':>4s} {'95% CI':>13s} {'inv':>4s} {'mis':>4s} "
          f"{'val':>4s} {'om':>3s} {'abst':>4s} {'rej':>4s}")
    for c in out["leakage"]:
        lo, hi = c["entity_ci"]
        label = f"{c['arm_id']} ({c['task']}{', strict' if c['strict'] else ''})"
        print(f"{label[:62]:62s} {c['n']:4d} {c['entity']:4d} [{lo:4.1f},{hi:5.1f}] "
              f"{c['invented']:4d} {c['misfiled']:4d} {c['value']:4d} {c['omission']:3d} "
              f"{c['abstained']:4d} {c['rejected_once_or_more']:4d}")
        if c["top_tokens"]:
            print(f"{'':6s}tokens: {c['top_tokens']}")
    print()
    for c in out["profile"]:
        print(f"profile {c['file'][:70]:70s} N={c['n']} responses {c['responses']} items {c['items']}")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
