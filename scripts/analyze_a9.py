"""Analysis plan A9: the interleaved naming run and the leakage retry audit.

Exactly what A9 registered, and nothing else.

A9.1  Responses with at least one wrong number, per naming scheme, and one
      two-sided Fisher exact test (suffixed against natural, alpha 0.05).
      Descriptive: the share of suffixed wrong numbers equal to the value the
      same-suffix sibling column records for the same statistic.
A9.2  Per arm: delivered violations and retried responses; every rejected
      attempt re-judged by the second scorer (scripts/definitions_scorer.py,
      no code shared with the verifier); correct claims before and after each
      retry.

    python -X utf8 scripts/analyze_a9.py [--json benchmark/a9_results.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import rejudge_rejections as rj  # noqa: E402
import unified_scoring as u  # noqa: E402
from analyze_revision_runs import _evidence, _records  # noqa: E402

RUNS = ROOT / "scripts" / "runs"
NAMING_DIR = RUNS / "profile" / "2026-10-02_interleaved"
AUDIT_DIR = RUNS / "2026-10-02_retry_audit"
TAU = u.TAU


def _dump(f: pathlib.Path) -> pathlib.Path:
    return f.parent / f"evidence_{f.stem.rsplit('_e', 1)[1]}.json"


def _sibling(col: str) -> str | None:
    if col.startswith("num_feature_"):
        return col.replace("num_feature_", "cat_feature_", 1)
    if col.startswith("cat_feature_"):
        return col.replace("cat_feature_", "num_feature_", 1)
    return None


def naming() -> dict:
    from scipy.stats import fisher_exact  # noqa: PLC0415

    cells: dict[str, dict] = {}
    for f in sorted(NAMING_DIR.glob("*.jsonl")):
        ev = _evidence(_dump(f))
        ctx = u.context("profile", ev)
        recs = _records(f)
        if not recs:
            continue
        scheme = recs[0]["frame_names"]
        scored = [u.score(r, "profile", ctx) for r in recs]
        sibling_hits = wrong_claims = 0
        for r in recs:
            for c in r.get("raw_claims") or []:
                if not isinstance(c, dict):
                    continue
                col, stat, v = str(c.get("column")), str(c.get("statistic")), c.get("value")
                key = (col, stat)
                if key not in ctx["values"] or not isinstance(v, (int, float)) or isinstance(v, bool):
                    continue
                if abs(float(v) - ctx["values"][key]) <= TAU:
                    continue
                wrong_claims += 1
                sib = _sibling(col)
                if sib and (sib, stat) in ctx["values"] and abs(float(v) - ctx["values"][(sib, stat)]) <= TAU:
                    sibling_hits += 1
        cells[scheme] = {
            "file": f.name, "n": len(recs),
            "wrong_number_responses": sum(bool(s["c2_wrong"]) for s in scored),
            "any_violation": sum(bool(s["any"]) for s in scored),
            "c1": sum(bool(s["c1_cited"] or s["c1_claim"]) for s in scored),
            "not_carried": sum(bool(s["c2_not_carried"]) for s in scored),
            "kinds": dict(Counter(s["kind"] for s in scored if s["kind"])),
            "wrong_claims": wrong_claims, "wrong_claims_from_sibling": sibling_hits,
            "claims_per_response": sum(s["claims"] for s in scored) / len(scored),
        }
    out = {"cells": cells}
    if {"suffixed", "natural"} <= set(cells):
        a, b = cells["suffixed"], cells["natural"]
        table = [[a["wrong_number_responses"], a["n"] - a["wrong_number_responses"]],
                 [b["wrong_number_responses"], b["n"] - b["wrong_number_responses"]]]
        out["fisher_two_sided_p"] = float(fisher_exact(table)[1])
        out["rejects_at_0.05"] = out["fisher_two_sided_p"] < 0.05
    return out


def audit() -> dict:
    import definitions_scorer as ds  # noqa: PLC0415

    import reproduce_hallucination_ablation as h  # noqa: PLC0415

    arms: dict[str, dict] = {}
    for f in sorted(AUDIT_DIR.glob("*.jsonl")):
        ev = _evidence(_dump(f))
        ev_ds = ds.load_evidence(_dump(f))
        ctx = u.context("leakage", ev)
        recs = [r for r in _records(f)]
        if not recs:
            continue
        arm = recs[0].get("arm_id")
        scored = [u.score(r, "leakage", ctx) for r in recs]
        rejected, confirmed, final_clean = 0, 0, 0
        correct_before = correct_after = lost = gained = 0
        findings = Counter()
        for r in recs:
            at = r.get("attempts") or []
            if len(at) < 2:
                continue
            for a in at[:-1]:
                rejected += 1
                found = rj.judge_leakage_attempt(a, ev_ds)
                confirmed += bool(found)
                for x in found:
                    findings[x.split(":")[0]] += 1
            final_clean += not rj.judge_leakage_attempt(at[-1], ev_ds)

            def good(att):
                out = set()
                for c in att.get("claims") or []:
                    if not isinstance(c, dict):
                        continue
                    key = (str(c.get("column")), h._normalise_statistic(c.get("statistic")))
                    v = c.get("value")
                    if key in ctx["values"] and isinstance(v, (int, float)) and not isinstance(v, bool) \
                            and abs(float(v) - ctx["values"][key]) <= TAU:
                        out.add(key)
                return out

            g0, g1 = good(at[0]), good(at[-1])
            correct_before += len(g0)
            correct_after += len(g1)
            lost += len(g0 - g1)
            gained += len(g1 - g0)
        arms[arm] = {
            "file": f.name, "n": len(recs),
            "delivered_violations": sum(bool(s["any"]) for s in scored),
            "retried_responses": sum(int(r.get("rejections") or 0) > 0 for r in recs),
            "rejected_attempts": rejected, "confirmed_by_second_scorer": confirmed,
            "findings": dict(findings), "retried_final_clean": final_clean,
            "correct_claims_first_attempt": correct_before,
            "correct_claims_final_attempt": correct_after,
            "correct_lost": lost, "correct_gained": gained,
            "claims_per_response": sum(s["claims"] for s in scored) / len(scored),
        }
    total = sum(a["rejected_attempts"] for a in arms.values())
    conf = sum(a["confirmed_by_second_scorer"] for a in arms.values())
    return {"arms": arms, "rejected_attempts": total, "confirmed": conf,
            "confirmation_rate": (conf / total) if total else None,
            "meets_registered_95": (conf / total >= 0.95) if total else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    res = {"A9.1": naming(), "A9.2": audit()}
    print(json.dumps(res, indent=1))
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
