"""Analysis plan A14: author-time domains that are only partly stale.

Exactly what A14 registered. Per coverage level k (0, 3, 5, 8, 10 of the
crowded instance's ten evidence columns), the responses with an invented name,
a cited name in neither E nor the frame (primary), and, descriptively, any
contract violation, misfiled names, omissions of the anchor and abstentions,
scored by scripts/unified_scoring.py. P1: a two-sided Cochran-Armitage test for
a trend in the invented-name share over the five levels (scores k). P2: a
two-sided Fisher exact test of k = 8 against k = 10.

    python -X utf8 scripts/analyze_a14.py [--json benchmark/a14_results.json]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import unified_scoring as u  # noqa: E402
from analyze_revision_runs import _evidence, _records, wilson  # noqa: E402
from make_tables import fisher_exact_two_sided  # noqa: E402

A14_DIR = ROOT / "scripts" / "runs" / "2026-10-03_a14"


def cochran_armitage(counts: list[tuple[int, int, float]]) -> tuple[float, float]:
    """Two-sided Cochran-Armitage trend test over (events, n, score) groups."""
    big_n = sum(n for _, n, _ in counts)
    p = sum(x for x, _, _ in counts) / big_n
    t = sum(score * (x - n * p) for x, n, score in counts)
    s1 = sum(n * score for _, n, score in counts)
    s2 = sum(n * score * score for _, n, score in counts)
    var = p * (1 - p) * (s2 - s1 * s1 / big_n)
    if var <= 0:
        return 0.0, 1.0
    z = t / math.sqrt(var)
    return z, math.erfc(abs(z) / math.sqrt(2))


def level_cells() -> dict[int, dict]:
    cells: dict[int, dict] = {}
    for f in sorted(A14_DIR.glob("*.jsonl")):
        ev = _evidence(f.parent / f"evidence_{f.stem.rsplit('_e', 1)[1]}.json")
        ctx = u.context("leakage", ev)
        recs = [r for r in _records(f) if not r.get("error")]
        if not recs:
            continue
        k = int(str(recs[0]["arm"]).rsplit("_k", 1)[1])
        frame = u.frame_columns(str(recs[0].get("task")), ctx)
        scored = [u.score(r, "leakage", ctx, frame) for r in recs]
        n = len(recs)
        invented = sum(bool(s["names_invented"]) for s in scored)
        cells[k] = {
            "file": f.name, "n": n,
            "transport_errors": sum(1 for r in _records(f) if r.get("error")),
            "invented_name": invented, "invented_wilson95": list(wilson(invented, n)),
            "any_violation": sum(bool(s["any"]) for s in scored),
            "misfiled": sum(bool(s["c1_misfiled"]) for s in scored),
            "omission": sum(bool(s["c3"]) for s in scored),
            "abstain": sum(str(r.get("verdict") or "") == "cannot_determine" for r in recs),
            "kinds": {kind: sum(s["kind"] == kind for s in scored)
                      for kind in ("extrinsic", "contradicts", "misplaced", "omission")},
            "static_columns": (recs[0].get("baseline") or {}).get("static_columns"),
            "coverage": ((recs[0].get("baseline") or {}).get("coverage") or {}).get("overlap"),
        }
    return dict(sorted(cells.items()))


def analyse() -> dict:
    cells = level_cells()
    out: dict = {"cells": cells}
    if len(cells) == 5:
        z, p = cochran_armitage([(c["invented_name"], c["n"], float(k)) for k, c in cells.items()])
        out["P1_trend_z"], out["P1_trend_p"] = z, p
        out["P1_holds"] = p < 0.05 and z < 0
        a, b = cells[8], cells[10]
        out["P2_fisher_p"] = fisher_exact_two_sided(a["invented_name"], a["n"], b["invented_name"], b["n"])
        out["P2_holds"] = out["P2_fisher_p"] < 0.05 and a["invented_name"] / a["n"] > b["invented_name"] / b["n"]
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
