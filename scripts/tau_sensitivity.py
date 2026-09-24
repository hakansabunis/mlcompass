"""How much does the value tolerance tau = 0.005 decide? Offline, no API calls.

Stanford review Q3 asked whether the value channel's absolute tolerance is a
free parameter the results lean on. This script answers from the recorded
responses: for every claim whose (entity, statistic) the evidence carries, it
takes |cited - measured|, then counts the responses that would carry a value
violation at each tau in TAUS.

Two things it cannot do, and says so in its output:

* Enforced arms were verified at 0.005 while they ran. A claim within 0.005
  that a tighter tau would reject was never sent back for correction, so the
  tighter-tau count on those arms is what the verifier *would have caught*,
  not what the retried response would have looked like. Where the harness kept
  the pre-verification claims (``raw_claims``, profile task), those are scored
  instead, and the column says which was used.
* An absolute tolerance means different things on a correlation in [-1, 1] and
  on a standard deviation of 846. The relative column shows the profile task's
  deviations as a fraction of the measured value, which is the honest scale
  there.

Usage:
    python -X utf8 scripts/tau_sensitivity.py [--json benchmark/tau_sensitivity.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from reproduce_hallucination_ablation import (  # noqa: E402
    _normalise_statistic,
    evidence_value_table,
)
from mlcompass.agents.evidence_contract import bind_profile  # noqa: E402

TAUS = (0.0005, 0.001, 0.005, 0.01, 0.05)
BUCKETS = (0.0, 0.0005, 0.001, 0.005, 0.01, 0.05, float("inf"))

RUNS = ROOT / "scripts" / "runs"
LEAKAGE_ARMS = [
    ("A-L1", RUNS, "layer1"),
    ("A-TIER-A", RUNS, "tier_a"),
    ("A-STRICT-STATIC", RUNS, "static_schema_noenum"),
    ("A-STATIC-ENUM-STALE", RUNS, "static_schema"),
    ("A-GR-STOCK", RUNS, "guardrails_stock"),
    ("A-CONTRACT", RUNS, "layer3_bare"),
    ("A-STRESS", RUNS, "layer3_stress"),
    ("A-GR-OURS", RUNS, "guardrails_tierb"),
    ("A-GR-CHOICES", RUNS / "2026-09-18_choices", "guardrails_choices"),
]
PROFILE_ARMS = [
    ("P-L1", "bare"),
    ("P-TIER-A", "tier_a"),
    ("P-CONTRACT", "contract"),
    ("P-STRESS", "stress"),
]


def _evidence(path: pathlib.Path) -> dict:
    """Evidence dumps are wrapped as {"task_label", "evidence"}."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw.get("evidence", raw)


def _records(path: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _deviations(claims, table, normalise):
    """|cited - measured| for every numeric claim the evidence can check."""
    out = []
    for c in claims or []:
        if not isinstance(c, dict):
            continue
        key = (str(c.get("column", "")), normalise(c.get("statistic")))
        val = c.get("value")
        if key not in table or isinstance(val, bool) or not isinstance(val, (int, float)):
            continue
        m = table[key]
        out.append((abs(float(val) - m), abs(m)))
    return out


def _arm(name, recs, table, normalise, source):
    per_resp = []
    all_dev = []
    for r in recs:
        claims = r.get("raw_claims") if source == "raw_claims" else r.get("claims")
        devs = _deviations(claims, table, normalise)
        per_resp.append(devs)
        all_dev.extend(devs)
    n = len(recs)
    viol = {t: sum(1 for d in per_resp if any(x > t for x, _ in d)) for t in TAUS}
    hist = []
    for lo, hi in zip(BUCKETS, BUCKETS[1:]):
        hist.append(sum(1 for x, _ in all_dev if (x > lo if lo else x >= 0) and x <= hi)
                    if lo == 0.0 else sum(1 for x, _ in all_dev if lo < x <= hi))
    exact = sum(1 for x, _ in all_dev if x == 0.0)
    rel = [x / m for x, m in all_dev if m > 0 and x > 0]
    return {
        "arm": name,
        "n": n,
        "claims_checked": len(all_dev),
        "source": source,
        "exact": exact,
        "hist": hist,
        "responses_violating": viol,
        "max_rel_dev_nonzero": max(rel) if rel else 0.0,
        "median_rel_dev_nonzero": sorted(rel)[len(rel) // 2] if rel else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write the table here")
    args = ap.parse_args()

    results = []
    # Every leakage record file, not a hand-picked list: the first version of
    # this script read only the root battery and missed a sign-flipped claim in
    # a re-run (review F-13). Each file is scored against the evidence dump that
    # carries its own hash; superseded runs are skipped.
    tables: dict[str, dict] = {}
    for f in sorted(RUNS.rglob("deepseek_*_synthetic*_seed0_e*.jsonl")):
        if "superseded" in f.parts[-2] or "superseded" in str(f.parent):
            continue
        ehash = f.stem.rsplit("_e", 1)[1]
        if ehash not in tables:
            dump = next((d / f"evidence_{ehash}.json" for d in (f.parent, RUNS)
                         if (d / f"evidence_{ehash}.json").exists()), None)
            if dump is None:
                print(f"{f.name}: no evidence dump for {ehash}", file=sys.stderr)
                continue
            tables[ehash] = evidence_value_table(_evidence(dump))
        recs = _records(f)
        arm_id = str(recs[0].get("arm_id") or f.stem) if recs else f.stem
        where = "root" if f.parent == RUNS else f.parent.name
        results.append(("leakage", _arm(f"{arm_id} [{where}]", recs, tables[ehash],
                                        _normalise_statistic, "claims")))
    if not results:
        raise SystemExit("no leakage records found")

    prof_table = dict(bind_profile(_evidence(RUNS / "profile" / "evidence_bcaea394.json")).values)
    if not prof_table:
        raise SystemExit("profile value table is empty: evidence did not load")
    for name, arm in PROFILE_ARMS:
        f = next((RUNS / "profile").glob(f"*_profile_{arm}_n200_*.jsonl"), None)
        if f is None:
            print(f"{name}: missing", file=sys.stderr)
            continue
        recs = _records(f)
        source = "raw_claims" if all("raw_claims" in r for r in recs) else "claims"
        results.append(("profile", _arm(name, recs, prof_table,
                                        lambda s: str(s or ""), source)))

    edges = ["=0", "<=5e-4", "<=1e-3", "<=5e-3", "<=1e-2", "<=5e-2", ">5e-2"]
    print("Claim deviations |cited - measured| (claims the evidence can check)")
    print(f"{'task':8s} {"arm":44s} {'src':10s} {'claims':>6s} " + " ".join(f"{e:>7s}" for e in edges))
    for task, r in results:
        h = [r["exact"], r["hist"][0] - r["exact"]] + r["hist"][1:]
        print(f"{task:8s} {r["arm"][:44]:44s} {r['source']:10s} {r['claims_checked']:6d} "
              + " ".join(f"{x:7d}" for x in h))
    print()
    print("Responses carrying >= 1 value violation, by tau")
    print(f"{'task':8s} {"arm":44s} {'N':>4s} " + " ".join(f"{t:>8g}" for t in TAUS))
    for task, r in results:
        print(f"{task:8s} {r["arm"][:44]:44s} {r['n']:4d} "
              + " ".join(f"{r['responses_violating'][t]:8d}" for t in TAUS))
    print()
    print("Enforced arms ran at tau=0.005: their counts at a tighter tau are what the")
    print("verifier would have rejected, not what a retried response would have been.")

    if args.json:
        out = pathlib.Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "taus": TAUS,
            "bucket_edges": edges,
            "results": [{"task": t, **r, "responses_violating":
                         {str(k): v for k, v in r["responses_violating"].items()}}
                        for t, r in results],
        }, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
