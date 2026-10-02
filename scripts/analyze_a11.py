"""Analysis plan A11: the verifier added to a well-specified configuration.

Exactly what A11 registered, and nothing else. Per arm, the responses with a
delivered violation, that is, any of (C1)-(C3) in the payload as the payload
reaches the user, scored by scripts/unified_scoring.py; one two-sided Fisher
exact test (no verifier against verifier, alpha 0.05), with the Newcombe 95 %
interval of the difference. Descriptive: the kinds, the retries, every rejected
attempt re-judged by scripts/rejudge_rejections.py, correct claims per
response, calls and latency.

    python -X utf8 scripts/analyze_a11.py [--json benchmark/a11_results.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import rejudge_rejections as rj  # noqa: E402
import unified_scoring as u  # noqa: E402
from analyze_revision_runs import _evidence, _records  # noqa: E402
from make_tables import newcombe_diff_ci95  # noqa: E402

A11_DIR = ROOT / "scripts" / "runs" / "profile" / "2026-10-02_a11"
ARMS = ("strong_noverify", "strong_verify")


def _dump(f: pathlib.Path) -> pathlib.Path:
    return f.parent / f"evidence_{f.stem.rsplit('_e', 1)[1]}.json"


def arm_cells() -> dict[str, dict]:
    cells: dict[str, dict] = {}
    for f in sorted(A11_DIR.glob("*.jsonl")):
        dump = _dump(f)
        ev = _evidence(dump)
        ctx = u.context("profile", ev)
        cols, values = rj.evidence_table(json.loads(dump.read_text(encoding="utf-8")))
        recs = _records(f)
        if not recs:
            continue
        arm = recs[0]["arm"]
        scored = [u.score(r, "profile", ctx, u.frame_columns("profile", ctx)) for r in recs]
        rejected = confirmed = 0
        findings: Counter = Counter()
        for r in recs:
            at = r.get("attempts") or []
            for a in at[:-1]:
                rejected += 1
                found = rj.problems(a, cols, values)
                confirmed += bool(found)
                for _, finding in found:
                    findings[finding.split(":")[0]] += 1
        cells[arm] = {
            "file": f.name, "n": len(recs), "indices": sorted(r["i"] for r in recs),
            "delivered_violations": sum(bool(s["any"]) for s in scored),
            "c1": sum(bool(s["c1_cited"] or s["c1_claim"]) for s in scored),
            "c2_wrong": sum(bool(s["c2_wrong"]) for s in scored),
            "c2_not_carried": sum(bool(s["c2_not_carried"]) for s in scored),
            "c3": sum(bool(s["c3"]) for s in scored),
            "kinds": dict(Counter(s["kind"] for s in scored if s["kind"])),
            "names_misfiled": sum(bool(s["c1_misfiled"]) for s in scored),
            "names_unlisted": sum(bool(s["names_unlisted"]) for s in scored),
            "names_invented": sum(bool(s["names_invented"]) for s in scored),
            "retried_responses": sum(int(r.get("rejections") or 0) > 0 for r in recs),
            "rejected_attempts": rejected, "confirmed_by_second_reading": confirmed,
            "findings": dict(findings),
            "omitted": sum(bool(r.get("omitted")) for r in recs),
            "verdicts": dict(Counter(str(r.get("verdict")) for r in recs)),
            "provider_calls": sum(int(r.get("provider_calls") or 0) for r in recs),
            "claims_per_response": statistics.fmean(s["claims"] for s in scored),
            "correct_claims_per_response": statistics.fmean(s["claims_correct"] for s in scored),
            "anchor_value_share": statistics.fmean(bool(s["anchor_value"]) for s in scored),
            "latency_ms_median": statistics.median(float(r.get("latency_ms") or 0) for r in recs),
        }
    return cells


def analyse() -> dict:
    from scipy.stats import fisher_exact  # noqa: PLC0415

    cells = arm_cells()
    out: dict = {"cells": cells}
    if set(ARMS) <= set(cells):
        a, b = cells["strong_noverify"], cells["strong_verify"]
        out["complete_pairs"] = len(set(a["indices"]) & set(b["indices"]))
        ka, na, kb, nb = a["delivered_violations"], a["n"], b["delivered_violations"], b["n"]
        out["fisher_two_sided_p"] = float(fisher_exact([[ka, na - ka], [kb, nb - kb]])[1])
        out["rejects_at_0.05"] = out["fisher_two_sided_p"] < 0.05
        out["difference"] = ka / na - kb / nb
        out["newcombe_95"] = list(newcombe_diff_ci95(ka, na, kb, nb))
    for c in cells.values():
        del c["indices"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    res = analyse()
    print(json.dumps(res, indent=1))
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
