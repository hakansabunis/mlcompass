"""Time the local check: verify() plus strip_unsound(), per recorded response.

The paper quoted a median of 18 microseconds with no script in the tree that
produced it (EMSE review 2026-10-01). This is that script. It replays every
recorded leakage and profile response through the shipped contract layer on
this machine, timing each call with perf_counter_ns over REPEATS repetitions
and keeping the per-response median. Binding the evidence is done once per
evidence dump and timed separately, because the product binds once per call.

It measures the local check only. Request time, the larger schema and the
extra calls a retry makes are different costs, reported in the paper apart.

    python -X utf8 scripts/time_verifier.py [--json benchmark/verifier_timing.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import unified_scoring as u  # noqa: E402
from analyze_revision_runs import _records  # noqa: E402

from mlcompass.agents.evidence_contract import LEAKAGE, PROFILE, strip_unsound, verify  # noqa: E402

REPEATS = 5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    per_response: list[float] = []
    bind_times: list[float] = []
    by_size: dict[int, list[float]] = {}
    for f, rel, ev in u.run_files():
        spec = PROFILE if rel[0] == "profile" else LEAKAGE
        t0 = time.perf_counter_ns()
        bound = spec.bind(ev)
        bind_times.append((time.perf_counter_ns() - t0) / 1e3)
        for r in _records(f):
            payload = {"verdict": r.get("verdict"),
                       "columns_referenced": r.get("raw_columns") or r.get("columns") or [],
                       "claims": r.get("raw_claims") or r.get("claims") or []}
            samples = []
            for _ in range(REPEATS):
                t = time.perf_counter_ns()
                verify(payload, bound, spec)
                strip_unsound(payload, bound, spec)
                samples.append((time.perf_counter_ns() - t) / 1e3)
            per_response.append(statistics.median(samples))
            size = len(payload["columns_referenced"]) + len(payload["claims"])
            by_size.setdefault(min(size // 5, 4), []).append(per_response[-1])
    per_response.sort()
    # The five repetitions only steady each response's timing; they are not
    # five uses. The distribution is over the 19,132 responses.
    sizes = {f"{5 * k}-{5 * k + 4}" if k < 4 else "20+": round(statistics.median(v), 1)
             for k, v in sorted(by_size.items())}
    out = {
        "responses": len(per_response),
        "repeats": REPEATS,
        "median_us": round(statistics.median(per_response), 1),
        "p95_us": round(per_response[int(0.95 * len(per_response))], 1),
        "max_us": round(per_response[-1], 1),
        "bind_median_us": round(statistics.median(bind_times), 1),
        "median_us_by_cited_plus_claims": sizes,
        "python": platform.python_version(),
        "machine": f"{platform.system()} {platform.release()}, {platform.processor() or platform.machine()}",
    }
    print(json.dumps(out, indent=1))
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
