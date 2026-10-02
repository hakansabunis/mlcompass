"""Build the blind semantic audit of analysis plan amendment A7. Offline.

The automatic kinds (unified_scoring.py) say whether an item is in E and where;
they cannot say whether a misplaced item misleads a reader. This audit asks
people. One item is one flagged element of one response (a cited name or a
claim) shown with the evidence E and the rest of the response, and the
labeller answers one question about it:

  A  supported: E states this, in this place
  B  right information, wrong place: true of E (or of the data), but in a field
     or under a key that does not fit, and a reader would not be misled
  C  false or unsupported relation: built from things E contains, but the
     relation it states is not in E or contradicts it
  D  a name that neither E nor the data contains, or a number E contains nowhere
  E  cannot tell

Sampling (registered in A7 before any label): every response contributes at
most one item. Rare strata are taken whole, the others sampled with seed
20261002, then the order is shuffled. The sheet hides arm, model, provider,
date and stratum.

    python -X utf8 scripts/make_semantic_audit.py
    python -X utf8 scripts/make_semantic_audit.py --score a.csv [b.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import random
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import unified_scoring as u  # noqa: E402
from analyze_revision_runs import _records  # noqa: E402

OUT = ROOT / "benchmark" / "semantic_audit_A7"
SEED = 20261002
TAU = u.TAU
LABELS = ["A", "B", "C", "D", "E"]
# stratum -> quota (None: every eligible response)
QUOTA = {
    "S3 wrong number, another item's value": None,
    "S7 number E records nowhere": None,
    "S6 invented name": 30,
    "S5 invented name, list written in advance": 10,
    "S4 real column E does not list": 15,
    "S2 number E records under another pair": 30,
    "S1 misfiled name": 40,
    "S8 control: admissible name": 20,
    "S9 control: claim that matches E": 20,
}


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def items_of(r: dict, kind: str, ctx: dict, frame: set[str]) -> list[dict]:
    """Every element of one response with its automatic stratum."""
    profile_raw = kind == "profile" and r.get("arm") not in u.VERIFIED
    claims = [c for c in (r.get("raw_claims") if profile_raw else r.get("claims")) or []
              if isinstance(c, dict)]
    cited = [str(c) for c in (r.get("raw_columns") if profile_raw else r.get("columns")) or []]
    cols, names, values = ctx["cols"], ctx["names"], ctx["values"]
    out, seen = [], set()
    stale = r.get("arm_id") == "A-STATIC-ENUM-STALE"
    for where, name in [("cited", c) for c in cited] + [
            ("claim", "None" if c.get("column") is None else str(c.get("column"))) for c in claims]:
        if name in seen:
            continue
        seen.add(name)
        if name in cols:
            stratum = "S8 control: admissible name"
        elif u.h.is_scorer_artifact(name, cols):
            continue
        elif name in names:
            stratum = "S1 misfiled name"
        elif name in frame:
            stratum = "S4 real column E does not list"
        else:
            stratum = ("S5 invented name, list written in advance" if stale else "S6 invented name")
        out.append({"type": "name", "target": name, "stratum": stratum})
    for j, c in enumerate(claims):
        col = "None" if c.get("column") is None else str(c.get("column"))
        if col not in cols:
            continue  # judged through its name
        key, v = (col, ctx["norm"](c.get("statistic"))), c.get("value")
        if not _num(v):
            continue
        elsewhere = any(abs(float(v) - x) <= TAU for x in ctx["numbers"])
        if key not in values:
            stratum = "S2 number E records under another pair" if elsewhere else "S7 number E records nowhere"
        elif abs(float(v) - values[key]) <= TAU:
            stratum = "S9 control: claim that matches E"
        else:
            stratum = "S3 wrong number, another item's value" if elsewhere else "S7 number E records nowhere"
        out.append({"type": "claim", "target": j, "stratum": stratum})
    return out


def evidence_view(kind: str, ev: dict, ctx: dict, frame: set[str]) -> dict:
    grouped: dict[tuple[str, float], list[str]] = defaultdict(list)
    for (col, stat), v in ctx["values"].items():
        grouped[(col, v)].append(stat)
    rows = sorted([col, ", ".join(sorted(stats)), v] for (col, v), stats in grouped.items())
    other = sorted(n for n in ctx["names"] - ctx["cols"] if 0 < len(n) <= 60 and "\n" not in n)
    return {"values": rows, "other_names": other, "frame": sorted(frame),
            "task": "profile" if kind == "profile" else "leakage"}


def collect() -> list[dict]:
    pool = []
    for f, rel, ev in u.run_files():
        kind = "profile" if rel[0] == "profile" else "leakage"
        ctx = u.context(kind, ev)
        view = None
        for pos, r in enumerate(_records(f)):
            frame = u.frame_columns(str(r.get("task")), ctx)
            its = items_of(r, kind, ctx, frame)
            if not its:
                continue
            if view is None:
                view = evidence_view(kind, ev, ctx, frame)
            pool.append({"file": f"{'/'.join(rel[:-1])}/{f.name}", "i": r.get("i"), "pos": pos,
                         "rec": r, "kind": kind, "items": its, "view": view,
                         "arm_id": r.get("arm_id"), "provider": r.get("provider")})
    return pool


def draw(pool: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    used: set[int] = set()
    picks = []
    for stratum, quota in QUOTA.items():
        eligible = [k for k, p in enumerate(pool) if k not in used
                    and any(it["stratum"] == stratum for it in p["items"])]
        chosen = eligible if quota is None or len(eligible) <= quota else rng.sample(eligible, quota)
        for k in sorted(chosen):
            p = pool[k]
            it = rng.choice([it for it in p["items"] if it["stratum"] == stratum])
            picks.append({**p, "item": it})
            used.add(k)
    rng.shuffle(picks)
    return picks


def build() -> int:
    pool = collect()
    picks = draw(pool)
    rng = random.Random(SEED + 1)
    ids = rng.sample(range(1000, 10000), len(picks))
    items, key = [], {}
    profile_raw = {True: ("raw_claims", "raw_columns"), False: ("claims", "columns")}
    for item_id, p in zip(ids, picks):
        r = p["rec"]
        raw = p["kind"] == "profile" and r.get("arm") not in u.VERIFIED
        ck, nk = profile_raw[raw]
        items.append({
            "id": item_id, "evidence": p["view"],
            "columns_referenced": [str(c) for c in (r.get(nk) or [])],
            "claims": [c for c in (r.get(ck) or []) if isinstance(c, dict)],
            "verdict": r.get("verdict"),
            "target_type": p["item"]["type"], "target": p["item"]["target"],
        })
        key[str(item_id)] = {"stratum": p["item"]["stratum"], "file": p["file"], "i": p["i"],
                             "pos": p["pos"], "arm_id": p["arm_id"], "provider": p["provider"],
                             "type": p["item"]["type"], "target": p["item"]["target"]}
    OUT.mkdir(parents=True, exist_ok=True)
    template = (ROOT / "scripts" / "semantic_audit_template.html").read_text(encoding="utf-8")
    page = template.replace("__ITEMS__", json.dumps(items, default=str))
    (OUT / "audit_sheet.html").write_text(page, encoding="utf-8")
    (OUT / "KEY_sealed.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    counts = Counter(k["stratum"] for k in key.values())
    eligible = Counter(it["stratum"] for p in pool for it in {i["stratum"]: i for i in p["items"]}.values())
    print(f"pool: {len(pool)} responses with at least one element")
    for s in QUOTA:
        print(f"  {s:48s} eligible responses {eligible[s]:6d}  drawn {counts[s]:4d}")
    print(f"drew {len(picks)} items; wrote {OUT / 'audit_sheet.html'} and the sealed key")
    return 0


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z, p = 1.959964, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, c - h), min(1.0, c + h))


def kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def score(paths: list[str]) -> int:
    key = json.loads((OUT / "KEY_sealed.json").read_text(encoding="utf-8"))
    sheets = []
    for path in paths:
        rows = {row["id"]: row["label"] for row in csv.DictReader(open(path, encoding="utf-8"))}
        sheets.append(rows)
    res: dict = {"items": len(key), "labellers": len(sheets), "by_stratum": {}}
    for li, rows in enumerate(sheets):
        by = defaultdict(Counter)
        for i, k in key.items():
            by[k["stratum"]][rows.get(i, "missing")] += 1
        res["by_stratum"][f"labeller_{li + 1}"] = {
            s: {lab: {"n": c[lab], "share": c[lab] / sum(c.values()),
                      "wilson95": wilson(c[lab], sum(c.values()))} for lab in sorted(c)}
            for s, c in sorted(by.items())}
    if len(sheets) == 2:
        common = [i for i in key if i in sheets[0] and i in sheets[1]]
        res["kappa"] = kappa([sheets[0][i] for i in common], [sheets[1][i] for i in common])
        res["agreement"] = sum(sheets[0][i] == sheets[1][i] for i in common) / len(common)
        res["disagreements"] = [{"id": i, **key[i], "a": sheets[0][i], "b": sheets[1][i]}
                                for i in common if sheets[0][i] != sheets[1][i]]
    (OUT / "audit_result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "disagreements"}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", nargs="+", default=None, help="CSV(s) exported from the sheet.")
    args = ap.parse_args()
    return score(args.score) if args.score else build()


if __name__ == "__main__":
    raise SystemExit(main())
