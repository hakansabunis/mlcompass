"""One representation policy for dataset- and task-level quantities, both tasks.

The leakage measurement accepts dataset-level quantities (the perfect-match
rate, the row count) under any listed column; the profile contract offers no
slot for the task-level class balance or the frame's shape. The 2026-10-02
review asked for one policy across the two tasks and a sensitivity analysis.
Three policies, applied to every response of the corpus:

  reported  as in the paper: leakage lenient, profile strict
  strict    neither task admits a dataset- or task-level quantity under a column
  lenient   both tasks admit them under any admissible column: on the profile
            task, a claim whose statistic names a class share (or the rows or
            columns of the frame) and whose value equals that quantity within
            tau

For each group of the kinds table it reports the responses that violate the
contract under each policy, and for the bare leakage arms the share of flagged
responses that misfile.

    python -X utf8 scripts/policy_sensitivity.py [--json benchmark/policy_sensitivity.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import unified_scoring as u  # noqa: E402
from analyze_revision_runs import _records  # noqa: E402

DATASET_LEVEL = ("perfect_match_rate", "row_count")
ROW_NAMES = {"rows", "row_count", "n_rows", "num_rows", "nrows", "row_total"}
COL_NAMES = {"cols", "columns", "n_cols", "num_columns", "column_count", "ncols"}


def task_level_numbers(ev: dict) -> dict[str, list[float]]:
    task = (ev.get("task_hint") or {}) if isinstance(ev, dict) else {}
    shape = (ev.get("shape") or {}) if isinstance(ev, dict) else {}
    return {"class": [float(v) for v in (task.get("class_balance") or {}).values()],
            "rows": [float(shape["rows"])] if "rows" in shape else [],
            "cols": [float(shape["cols"])] if "cols" in shape else []}


def admissible_task_level(claim: dict, numbers: dict[str, list[float]]) -> bool:
    stat = str(claim.get("statistic") or "").lower()
    v = claim.get("value")
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return False
    if any(w in stat for w in ("class", "balance", "share")):
        pool = numbers["class"]
    elif stat in ROW_NAMES:
        pool = numbers["rows"]
    elif stat in COL_NAMES:
        pool = numbers["cols"]
    else:
        return False
    return any(abs(float(v) - x) <= u.TAU for x in pool)


def violates(r: dict, kind: str, ctx: dict, frame: set[str], ev: dict, policy: str) -> bool:
    if kind == "leakage" and policy == "strict":
        ctx = {**ctx, "values": {k: v for k, v in ctx["values"].items() if k[1] not in DATASET_LEVEL}}
    base = u.score(r, kind, ctx, frame)
    if not (kind == "profile" and policy == "lenient") or not base["c2_not_carried"]:
        return bool(base["any"])
    if base["c1_cited"] or base["c1_claim"] or base["c2_wrong"] or base["c3"]:
        return True
    raw = kind == "profile" and r.get("arm") not in u.VERIFIED
    claims = [c for c in (r.get("raw_claims") if raw else r.get("claims")) or [] if isinstance(c, dict)]
    numbers = task_level_numbers(ev)
    for c in claims:
        col = "None" if c.get("column") is None else str(c.get("column"))
        if col in ctx["cols"] and (col, ctx["norm"](c.get("statistic"))) not in ctx["values"] \
                and not admissible_task_level(c, numbers):
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rows = []
    for f, rel, ev in u.run_files():
        kind = "profile" if rel[0] == "profile" else "leakage"
        ctx = u.context(kind, ev)
        for r in _records(f):
            frame = u.frame_columns(str(r.get("task")), ctx)
            base = u.score(r, kind, ctx, frame)
            rows.append({"run_dir": "/".join(rel[:-1]) or ".", "task": r.get("task"),
                         "names": r.get("frame_names") or "", "provider": r.get("provider"),
                         "arm_id": r.get("arm_id"), "c1_misfiled": base["c1_misfiled"],
                         **{p: violates(r, kind, ctx, frame, ev, p)
                            for p in ("reported", "strict", "lenient")}})
    out = {}
    print(f"{'group':52s} {'N':>6s} {'reported':>9s} {'strict':>7s} {'lenient':>8s}")
    for label, pred in u.LAYER_ROWS:
        cell = [r for r in rows if pred(r)]
        c = {p: sum(r[p] for r in cell) for p in ("reported", "strict", "lenient")}
        out[label] = {"n": len(cell), **c}
        print(f"{label[:52]:52s} {len(cell):6d} {c['reported']:9d} {c['strict']:7d} {c['lenient']:8d}")
    bare = [r for r in rows if u.LAYER_ROWS[0][1](r) or u.LAYER_ROWS[1][1](r)]
    for p in ("reported", "strict", "lenient"):
        flagged = [r for r in bare if r[p]]
        share = sum(r["c1_misfiled"] for r in flagged) / len(flagged) if flagged else 0
        out[f"bare_leakage_misfile_share_{p}"] = share
        print(f"bare leakage, {p}: {len(flagged)} flagged, {100 * share:.1f}% misfile")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
