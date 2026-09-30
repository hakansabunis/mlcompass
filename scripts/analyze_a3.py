"""Analyse the battery registered in analysis plan amendment A3. Offline.

    R1  profile, natural names: strong_noverify vs strong_verify, N=600 each,
        Fisher exact two-sided on delivered violations (new data only).
    R2a the 12 FabBench instances frozen 2026-07-07: A-L1 entity rate per
        instance, split misfiled / unlisted / invented.
    R2b the same instances: delivered violations pooled, A-STRONG-NOVERIFY vs
        A-STRONG-CONTRACT, Fisher exact two-sided.
    R3  the shipped default narrator (claude-opus-4-7): descriptive.

Holm over {R1, R2b}. A delivered violation is a response whose recorded
``scored`` block has entity, value or omission true, the definition A3 fixed.
The A-L1 split re-scores each record against its own evidence dump.

    python -X utf8 scripts/analyze_a3.py [--json benchmark/a3_results.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from scipy.stats import fisher_exact  # noqa: E402

import reproduce_hallucination_ablation as h  # noqa: E402
from analyze_revision_runs import _evidence, _names, _records, wilson  # noqa: E402

RUNS = ROOT / "scripts" / "runs"
R1_DIR = RUNS / "profile" / "2026-09-30_confirm_natural"
R2_DIR = RUNS / "2026-09-30_fabbench12"
R3_DIRS = {"leakage": RUNS / "2026-09-30_default_model",
           "profile": RUNS / "profile" / "2026-09-30_default_model"}
DATA = ROOT / "scripts" / "data"


def delivered(rec: dict) -> bool:
    s = rec.get("scored") or {}
    return bool(s.get("entity") or s.get("value") or s.get("omission"))


def fisher(k1: int, n1: int, k2: int, n2: int) -> float:
    return float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative="two-sided")[1])


def holm(ps: dict[str, float]) -> dict[str, float]:
    order = sorted(ps, key=ps.get)
    m, out, running = len(order), {}, 0.0
    for i, key in enumerate(order):
        running = max(running, min(1.0, (m - i) * ps[key]))
        out[key] = running
    return out


def _arm_file(d: pathlib.Path, pattern: str) -> pathlib.Path | None:
    hits = sorted(d.glob(pattern))
    return hits[0] if hits else None


def r1() -> dict:
    cells = {}
    for arm in ("strong_noverify", "strong_verify"):
        f = _arm_file(R1_DIR, f"*_profile_{arm}_n600_*.jsonl")
        recs = _records(f) if f else []
        k = sum(delivered(r) for r in recs)
        retried = sum(int(r.get("rejections") or 0) > 0 for r in recs)
        cells[arm] = {"n": len(recs), "delivered": k, "ci": wilson(k, len(recs)),
                      "retried": retried}
    a, b = cells["strong_noverify"], cells["strong_verify"]
    p = fisher(a["delivered"], a["n"], b["delivered"], b["n"]) if a["n"] and b["n"] else None
    return {"cells": cells, "p": p}


def _frame_columns(task: str) -> set[str]:
    # task label: csv:<file>/<target>/<injector>
    fname = task.split(":", 1)[1].split("/", 1)[0]
    return set(pd.read_csv(DATA / fname, nrows=0, low_memory=False).columns.astype(str))


def split_entity(rec: dict, allowed: set[str], names: set[str], frame: set[str]) -> set[str]:
    """Kinds of out-of-evidence names in one response."""
    cited = list(rec.get("columns") or []) + [
        str(c.get("column", "")) for c in (rec.get("claims") or []) if isinstance(c, dict)
    ]
    kinds = set()
    for c in cited:
        c = str(c)
        if c in allowed or h.is_scorer_artifact(c, allowed):
            continue
        if c in names:
            kinds.add("misfiled")
        elif c in frame:
            kinds.add("unlisted")
        else:
            kinds.add("invented")
    return kinds


def r2() -> dict:
    instances: dict[str, dict] = {}
    for f in sorted(R2_DIR.glob("deepseek_*.jsonl")):
        recs = _records(f)
        if not recs:
            continue
        task, arm = recs[0]["task"], recs[0]["arm"]
        inst = instances.setdefault(task, {})
        ehash = f.stem.rsplit("_e", 1)[1]
        ev = _evidence(f.parent / f"evidence_{ehash}.json")
        allowed = set(h.evidence_allowed_columns(ev))
        if arm == "layer1":
            names, frame = _names(ev), _frame_columns(task)
            corr, anchor, table = (h.evidence_correlation_map(ev), h.top_candidate(ev),
                                   h.evidence_value_table(ev))
            ent = val = om = 0
            kinds = {"misfiled": 0, "unlisted": 0, "invented": 0}
            tokens: dict[str, int] = {}
            for r in recs:
                resp = {k: r.get(k) for k in ("columns", "claims", "verdict", "omitted")}
                fl = h.score_one(resp, allowed, corr, anchor, evidence_names=names,
                                 value_table=table)
                ent += bool(fl.get("entity"))
                val += bool(fl.get("value"))
                om += bool(fl.get("omission"))
                ks = split_entity(r, allowed, names, frame) if fl.get("entity") else set()
                for k in ks:
                    kinds[k] += 1
                for c in set(list(r.get("columns") or [])):
                    if c not in allowed and not h.is_scorer_artifact(c, allowed):
                        tokens[c] = tokens.get(c, 0) + 1
            inst["A-L1"] = {"n": len(recs), "entity": ent, "ci": wilson(ent, len(recs)),
                            "value": val, "omission": om, **kinds,
                            "top_tokens": sorted(tokens.items(), key=lambda t: -t[1])[:5],
                            "anchor": anchor}
        else:
            k = sum(delivered(r) for r in recs)
            inst[recs[0]["arm_id"]] = {
                "n": len(recs), "delivered": k, "ci": wilson(k, len(recs)),
                "retried": sum(int(r.get("rejections") or 0) > 0 for r in recs)}
    pooled = {}
    for arm in ("A-STRONG-NOVERIFY", "A-STRONG-CONTRACT"):
        n = sum(v[arm]["n"] for v in instances.values() if arm in v)
        k = sum(v[arm]["delivered"] for v in instances.values() if arm in v)
        pooled[arm] = {"n": n, "delivered": k, "ci": wilson(k, n)}
    a, b = pooled["A-STRONG-NOVERIFY"], pooled["A-STRONG-CONTRACT"]
    p = fisher(a["delivered"], a["n"], b["delivered"], b["n"]) if a["n"] and b["n"] else None
    floor = [v["A-L1"] for v in instances.values() if "A-L1" in v]
    flagged = sum(c["entity"] for c in floor)
    invented = sum(c["invented"] for c in floor)
    return {"instances": instances, "pooled": pooled, "p": p,
            "floor_flagged": flagged, "floor_invented": invented,
            "invented_share": (invented / flagged) if flagged else None}


def r3() -> dict:
    out = {}
    for kind, d in R3_DIRS.items():
        for f in sorted(d.glob("anthropic_*.jsonl")):
            recs = _records(f)
            if not recs:
                continue
            k = sum(delivered(r) for r in recs)
            out[f.name] = {"kind": kind, "arm_id": recs[0].get("arm_id"), "n": len(recs),
                           "delivered": k, "ci": wilson(k, len(recs)),
                           "retried": sum(int(r.get("rejections") or 0) > 0 for r in recs),
                           "sampling": recs[0].get("sampling")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    res = {"R1": r1(), "R2": r2(), "R3": r3()}
    # Amendment A4: the budget ran out before R1 and R2b were complete, so
    # neither test is run and the Holm family is empty. The partial cells are
    # descriptive. The functions still compute p for a completed rerun.
    res["R1"]["p"] = res["R2"]["p"] = None
    res["holm"] = {}
    res["note"] = "A4: R1 and R2b not run (budget); partial cells descriptive only."

    c = res["R1"]["cells"]
    print("R1 profile natural, delivered violations")
    for arm, v in c.items():
        print(f"  {arm:16s} {v['delivered']:3d}/{v['n']:<4d} [{v['ci'][0]:.1f}, {v['ci'][1]:.1f}]"
              f"  retried {v['retried']}")
    print(f"  Fisher p = {res['R1']['p']}")

    print("\nR2 FabBench-12")
    for task, v in sorted(res["R2"]["instances"].items()):
        fl = v.get("A-L1", {})
        s = " ".join(f"{a}={v[a]['delivered']}/{v[a]['n']}" for a in
                     ("A-STRONG-NOVERIFY", "A-STRONG-CONTRACT") if a in v)
        print(f"  {task:48s} L1 {fl.get('entity', '-')}/{fl.get('n', '-')} "
              f"mis {fl.get('misfiled', '-')} unl {fl.get('unlisted', '-')} "
              f"inv {fl.get('invented', '-')} val {fl.get('value', '-')} | {s}")
        if fl.get("top_tokens"):
            print(f"  {'':48s} tokens {fl['top_tokens']}")
    for arm, v in res["R2"]["pooled"].items():
        print(f"  pooled {arm:18s} {v['delivered']}/{v['n']} [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]")
    print(f"  Fisher p = {res['R2']['p']}; floor flagged {res['R2']['floor_flagged']}, "
          f"invented {res['R2']['floor_invented']} (share {res['R2']['invented_share']})")
    print(f"\nHolm-adjusted: {res['holm']}")

    if res["R3"]:
        print("\nR3 claude-opus-4-7")
        for name, v in res["R3"].items():
            print(f"  {v['kind']:8s} {v['arm_id']:20s} {v['delivered']}/{v['n']} "
                  f"[{v['ci'][0]:.1f}, {v['ci'][1]:.1f}] retried {v['retried']} {v['sampling']}")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(res, indent=2, default=str),
                                           encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
