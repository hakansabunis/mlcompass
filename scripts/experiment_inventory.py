"""One row per run file: the inventory every count in the manuscript derives from.

Review 2026-09-30 asked that the reader be able to rebuild the total of live
responses from a single table. Each row is one record file (one arm, on one
task instance, from one battery): provider, model, arm, run directory, the
dates its records carry, planned and returned responses, transport errors
excluded from N, provider calls, responses that needed a retry, and the
harness commit pinned in the records. Superseded runs are listed separately
and never counted.

    python -X utf8 scripts/experiment_inventory.py [--csv benchmark/experiment_inventory.csv]
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import pathlib
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "scripts" / "runs"

FIELDS = [
    "run_dir", "file", "task", "provider", "model", "arm_id", "temperature",
    "dates", "harness_commits", "n_planned", "records", "errors_excluded",
    "valid", "provider_calls", "retried", "superseded",
]


def _provenance(rec: dict) -> dict:
    p = rec.get("provenance") or {}
    if isinstance(p, str):
        try:
            p = ast.literal_eval(p)
        except (ValueError, SyntaxError):
            p = {}
    return p if isinstance(p, dict) else {}


def row(path: pathlib.Path) -> dict:
    recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    valid = [r for r in recs if not r.get("error")]
    first = recs[0] if recs else {}
    dates, commits, temps, arms = Counter(), Counter(), Counter(), Counter()
    for r in recs:
        p = _provenance(r)
        dates[str(p.get("started_at", ""))[:10]] += 1
        commits[str(p.get("repo_commit", ""))] += 1
        temps[str((r.get("sampling") or {}).get("temperature"))] += 1
        arms[str(r.get("arm_id") or r.get("arm") or "")] += 1
    rel = path.relative_to(RUNS)
    return {
        "run_dir": str(rel.parent).replace("\\", "/"),
        "file": path.name,
        "task": first.get("task", "profile" if "profile" in path.name else ""),
        "provider": first.get("provider", path.name.split("_")[0]),
        "model": first.get("model", ""),
        "arm_id": "|".join(sorted(arms)),
        "temperature": "|".join(sorted(temps)),
        "dates": "|".join(sorted(d for d in dates if d)),
        "harness_commits": "|".join(sorted(c for c in commits if c)),
        "n_planned": first.get("n_planned", ""),
        "records": len(recs),
        "errors_excluded": len(recs) - len(valid),
        "valid": len(valid),
        "provider_calls": sum(int(r.get("provider_calls") or 1) for r in valid),
        "retried": sum(1 for r in valid if int(r.get("rejections") or 0) > 0),
        "superseded": "superseded" in str(rel),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=pathlib.Path, default=None)
    args = ap.parse_args()

    rows = [row(p) for p in sorted(RUNS.rglob("*.jsonl"))]
    counted = [r for r in rows if not r["superseded"]]
    by_provider: Counter = Counter()
    for r in counted:
        by_provider[r["provider"]] += r["valid"]
    print(f"{len(counted)} run files counted, {len(rows) - len(counted)} superseded files listed")
    print(f"valid live responses: {sum(r['valid'] for r in counted):,}  {dict(by_provider)}")
    print(f"transport errors excluded: {sum(r['errors_excluded'] for r in counted)}")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
