"""What a retry costs in content, and what each arm delivers that is correct.

Two questions from the 2026-10-02 review of V2.3. Offline.

1. Does a retry repair the violation, or does it also throw away acceptable
   content? Profile records written from 2026-09-24 keep every attempt, so for
   each rejected attempt we count the claims that matched E (within tau) and
   how many of those the final attempt no longer carries. For leakage, where
   only the final payload is kept, we compare the correct claims delivered by
   responses that went through a retry with those accepted at once, per arm.

2. How much correct content does each arm deliver? Per arm on the reference
   task: correct claims per response, and the share of responses that report
   the anchor's measured correlation (within tau).

    python -X utf8 scripts/retry_content.py [--json benchmark/retry_content.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import rejudge_rejections as rj  # noqa: E402
import unified_scoring as u  # noqa: E402
from analyze_revision_runs import _records  # noqa: E402

TAU = 0.005


def correct_claims(claims: list[dict], values: dict, norm=lambda s: str(s or "")) -> set:
    out = set()
    for c in claims or []:
        if not isinstance(c, dict):
            continue
        key = (str(c.get("column")), norm(c.get("statistic")))
        v = c.get("value")
        if key in values and isinstance(v, (int, float)) and not isinstance(v, bool) \
                and abs(float(v) - values[key]) <= TAU:
            out.add(key)
    return out


def profile_retries() -> list[dict]:
    rows = []
    for f in sorted((ROOT / "scripts" / "runs" / "profile").rglob("*.jsonl")):
        if any("superseded" in p or p == "transport_errors" for p in f.parts):
            continue
        dump = f.parent / f"evidence_{f.stem.rsplit('_e', 1)[1]}.json"
        cols, values = rj.evidence_table(json.loads(dump.read_text(encoding="utf-8")))
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            at = r.get("attempts") or []
            if len(at) < 2:
                continue
            first, final = at[0], at[-1]
            c_first = correct_claims(first.get("claims"), values)
            c_final = correct_claims(final.get("claims"), values)
            rows.append({"run": f.parent.name, "arm": r.get("arm_id"), "i": r.get("i"),
                         "claims_first": len(first.get("claims") or []),
                         "claims_final": len(final.get("claims") or []),
                         "correct_first": len(c_first), "correct_final": len(c_final),
                         "correct_lost": len(c_first - c_final),
                         "correct_gained": len(c_final - c_first)})
    return rows


def leakage_by_arm() -> dict:
    """Per (run, task, arm) on leakage: correct claims and anchor coverage,
    split by whether the response went through a retry."""
    acc = defaultdict(lambda: {"n": 0, "correct": 0, "anchor_value": 0,
                               "retried": 0, "correct_retried": 0, "correct_clean": 0})
    for f, rel, ev in u.run_files():
        if rel[0] == "profile":
            continue
        ctx = u.context("leakage", ev)
        anchor = ctx["anchor"]
        for r in _records(f):
            if r.get("provider") != "deepseek":
                continue
            good = correct_claims(r.get("claims"), ctx["values"], ctx["norm"])
            key = ("/".join(rel[:-1]) or ".", r.get("task"), r.get("arm_id"))
            a = acc[key]
            a["n"] += 1
            a["correct"] += len(good)
            a["anchor_value"] += any(col == anchor for col, _ in good) if anchor else 0
            retried = int(r.get("rejections") or 0) > 0
            a["retried"] += retried
            a["correct_retried" if retried else "correct_clean"] += len(good)
    return {"|".join(map(str, k)): v for k, v in acc.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    prof = profile_retries()
    content = [r for r in prof if r["claims_first"]]
    print("profile, every stored retry (first attempt -> final attempt):")
    for r in prof:
        print(f"  {r['arm']:18s} i={r['i']:<4} claims {r['claims_first']:>2} -> {r['claims_final']:>2}  "
              f"correct {r['correct_first']:>2} -> {r['correct_final']:>2}  "
              f"lost {r['correct_lost']:>2} gained {r['correct_gained']:>2}")
    tot = {k: sum(r[k] for r in content) for k in
           ("claims_first", "claims_final", "correct_first", "correct_final",
            "correct_lost", "correct_gained")}
    print("  totals over content retries:", tot)
    leak = leakage_by_arm()
    print("\nleakage, deepseek, per cell: mean correct claims, anchor value %, "
          "correct claims retried vs clean")
    for k, a in sorted(leak.items()):
        if not a["n"]:
            continue
        cr = a["correct_retried"] / a["retried"] if a["retried"] else float("nan")
        cc = a["correct_clean"] / (a["n"] - a["retried"]) if a["n"] > a["retried"] else float("nan")
        print(f"  {k[:70]:70s} n={a['n']:4d} correct {a['correct'] / a['n']:.2f} "
              f"anchor {100 * a['anchor_value'] / a['n']:5.1f}%  retried {a['retried']:3d} "
              f"({cr:.2f} vs {cc:.2f})")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps({"profile_retries": prof, "profile_totals": tot,
                                                        "leakage": leak}, indent=1),
                                           encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
