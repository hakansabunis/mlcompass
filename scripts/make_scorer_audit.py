"""Build the blind scorer audit of analysis plan amendment A3 (R4). Offline.

Draws 200 responses from the R2 runs with seed 20260930: 100 the scorer flags
(stratified by kind in proportion) and 100 it does not. Writes

    benchmark/scorer_audit_A3/audit_sheet.html   what the labeller opens
    benchmark/scorer_audit_A3/KEY_sealed.json    scorer labels, kept from the
                                                 labeller until labels return

The sheet shows, per response, the names and values E carries, the frame's
column list and the structured payload. It hides arm, model, provider, the
scorer's label and the flagged/unflagged split, and the order is shuffled.
The labeller exports a CSV from the page; `--score labels.csv` then reports
agreement and Cohen's kappa against the key.

    python -X utf8 scripts/make_scorer_audit.py
    python -X utf8 scripts/make_scorer_audit.py --score labels.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import reproduce_hallucination_ablation as h  # noqa: E402
from analyze_a3 import R2_DIR, _frame_columns  # noqa: E402
from analyze_revision_runs import _evidence, _names, _records  # noqa: E402

OUT = ROOT / "benchmark" / "scorer_audit_A3"
SEED = 20260930
N_FLAGGED = N_CLEAN = 100
NAME_LABELS = ["column in E", "misfiled", "unlisted", "invented", "artifact"]
CLAIM_LABELS = ["matches E", "differs from E", "pair not in E", "covered by name label"]
# Added after the V1.8 review (R1.1): is a contract violation a false statement?
RESPONSE_LABELS = ["no false statement", "false statement", "correct but in the wrong field",
                   "unclear"]


def _is_identifier(s: str) -> bool:
    return 0 < len(s) <= 60 and not any(ch in s for ch in "\n{}[]") and not _is_number(s)


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def scorer_labels(rec: dict, ctx: dict) -> dict:
    """What the scorer says, per cited name and per claim."""
    allowed, names, frame = ctx["allowed"], ctx["names"], ctx["frame"]
    corr, table = ctx["corr"], ctx["table"]
    name_lab = {}
    cited = [str(c) for c in (rec.get("columns") or [])] + [
        str(c.get("column", "")) for c in (rec.get("claims") or []) if isinstance(c, dict)
    ]
    for c in cited:
        if c in allowed:
            name_lab[c] = "column in E"
        elif h.is_scorer_artifact(c, allowed):
            name_lab[c] = "artifact"
        elif c in names:
            name_lab[c] = "misfiled"
        elif c in frame:
            name_lab[c] = "unlisted"
        else:
            name_lab[c] = "invented"
    claim_lab = []
    for c in rec.get("claims") or []:
        if not isinstance(c, dict):
            continue
        col = str(c.get("column", ""))
        if col not in corr:
            claim_lab.append("covered by name label")
            continue
        key = (col, h._normalise_statistic(c.get("statistic")))
        if key not in table:
            claim_lab.append("pair not in E")
        elif h._within_tolerance(c.get("value"), table[key]):
            claim_lab.append("matches E")
        else:
            claim_lab.append("differs from E")
    resp = {k: rec.get(k) for k in ("columns", "claims", "verdict", "omitted")}
    fl = h.score_one(resp, allowed, corr, ctx["anchor"], evidence_names=names, value_table=table)
    flagged = bool(fl.get("entity") or fl.get("value"))
    kinds = sorted({v for v in name_lab.values() if v not in ("column in E", "artifact")}
                   | ({"value"} if fl.get("value") else set()))
    return {"names": name_lab, "claims": claim_lab, "flagged": flagged, "kinds": kinds}


def collect() -> list[dict]:
    pool = []
    for f in sorted(R2_DIR.glob("deepseek_*.jsonl")):
        recs = _records(f)
        if not recs:
            continue
        ehash = f.stem.rsplit("_e", 1)[1]
        ev = _evidence(f.parent / f"evidence_{ehash}.json")
        allowed = set(h.evidence_allowed_columns(ev))
        names = _names(ev)
        ctx = {"allowed": allowed, "names": names, "frame": _frame_columns(recs[0]["task"]),
               "corr": h.evidence_correlation_map(ev), "table": h.evidence_value_table(ev),
               "anchor": h.top_candidate(ev)}
        # The value table keys every accepted spelling of a statistic
        # (corr, pearson, spearman_correlation, ...). One row per (column,
        # value), with its spellings joined, keeps the sheet readable.
        grouped: dict[tuple[str, float], list[str]] = {}
        for (col, stat), v in ctx["table"].items():
            grouped.setdefault((col, v), []).append(stat)
        shown = {
            "values": sorted([col, ", ".join(sorted(stats)), v]
                             for (col, v), stats in grouped.items()),
            "other_names": sorted(n for n in names - allowed if _is_identifier(n)),
            "frame": sorted(ctx["frame"]),
        }
        # `i` can repeat within a cell refilled after transport errors, so the
        # key also carries the record's position among the valid responses.
        for pos, r in enumerate(recs):
            pool.append({"file": f"{f.parent.name}/{f.name}", "i": r["i"], "pos": pos, "rec": r,
                         "shown": shown, "scorer": scorer_labels(r, ctx)})
    return pool


def build() -> int:
    pool = collect()
    rng = random.Random(SEED)
    flagged = [p for p in pool if p["scorer"]["flagged"]]
    clean = [p for p in pool if not p["scorer"]["flagged"]]
    # Stratify the flagged draw by its first kind, in proportion.
    strata: dict[str, list] = {}
    for p in flagged:
        strata.setdefault(p["scorer"]["kinds"][0] if p["scorer"]["kinds"] else "value", []).append(p)
    # Largest-remainder allocation, so the strata sum to exactly N_FLAGGED.
    quota = {k: N_FLAGGED * len(v) / len(flagged) for k, v in strata.items()}
    share = {k: int(q) for k, q in quota.items()}
    for k in sorted(quota, key=lambda k: quota[k] - share[k], reverse=True)[
        : N_FLAGGED - sum(share.values())
    ]:
        share[k] += 1
    picks = []
    for kind, items in sorted(strata.items()):
        picks += rng.sample(items, min(share[kind], len(items)))
    picks += rng.sample(clean, N_CLEAN)
    rng.shuffle(picks)

    ids = rng.sample(range(1000, 10000), len(picks))
    items, key = [], {}
    for item_id, p in zip(ids, picks):
        r = p["rec"]
        items.append({
            "id": item_id,
            "evidence_values": p["shown"]["values"],
            "evidence_other_names": p["shown"]["other_names"],
            "frame_columns": p["shown"]["frame"],
            "columns_referenced": [str(c) for c in (r.get("columns") or [])],
            "claims": [c for c in (r.get("claims") or []) if isinstance(c, dict)],
            "verdict": r.get("verdict"),
        })
        key[str(item_id)] = {"file": p["file"], "i": p["i"], "pos": p["pos"], **p["scorer"]}
    OUT.mkdir(parents=True, exist_ok=True)
    template = (ROOT / "scripts" / "scorer_audit_template.html").read_text(encoding="utf-8")
    page = (template.replace("__ITEMS__", json.dumps(items, default=str))
                    .replace("__NAME_LABELS__", json.dumps(NAME_LABELS))
                    .replace("__CLAIM_LABELS__", json.dumps(CLAIM_LABELS))
                    .replace("__RESPONSE_LABELS__", json.dumps(RESPONSE_LABELS)))
    (OUT / "audit_sheet.html").write_text(page, encoding="utf-8")
    (OUT / "KEY_sealed.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    counts = Counter(k for p in picks for k in (p["scorer"]["kinds"] or ["clean"]))
    print(f"pool {len(pool)} (flagged {len(flagged)}), drew {len(picks)}; kinds {dict(counts)}")
    print(f"wrote {OUT / 'audit_sheet.html'} and the sealed key")
    return 0


def kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def score(labels_csv: str) -> int:
    key = json.loads((OUT / "KEY_sealed.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader(open(labels_csv, encoding="utf-8")))
    human_resp: dict[str, bool] = {}
    name_pairs, claim_pairs, disagreements = [], [], []
    substantive: dict[str, Counter] = {"flagged": Counter(), "clean": Counter()}
    for row in rows:
        k = key[row["id"]]
        if row["kind"] == "response":
            substantive["flagged" if k["flagged"] else "clean"][row["label"]] += 1
            continue
        if row["kind"] == "name":
            s = k["names"].get(row["target"])
            name_pairs.append((s, row["label"]))
            bad = row["label"] not in ("column in E", "artifact")
        else:
            s = k["claims"][int(row["target"])]
            claim_pairs.append((s, row["label"]))
            bad = row["label"] == "differs from E"
        human_resp[row["id"]] = human_resp.get(row["id"], False) or bad
        if s != row["label"]:
            disagreements.append({"id": row["id"], "kind": row["kind"], "target": row["target"],
                                  "scorer": s, "human": row["label"], "file": k["file"], "i": k["i"]})
    ids = sorted(key)
    sc = ["flagged" if key[i]["flagged"] else "clean" for i in ids]
    hu = ["flagged" if human_resp.get(i, False) else "clean" for i in ids]
    agree = sum(x == y for x, y in zip(sc, hu))
    res = {
        "responses": len(ids), "response_agreement": agree, "response_kappa": kappa(sc, hu),
        "name_agreement": sum(x == y for x, y in name_pairs), "names": len(name_pairs),
        "claim_agreement": sum(x == y for x, y in claim_pairs), "claims": len(claim_pairs),
        "false_statement_by_scorer_flag": {k: dict(v) for k, v in substantive.items()},
        "disagreements": disagreements,
    }
    (OUT / "audit_result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "disagreements"}, indent=1))
    print(f"{len(disagreements)} disagreements -> {OUT / 'audit_result.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", default=None, help="CSV exported from the audit sheet.")
    args = ap.parse_args()
    return score(args.score) if args.score else build()


if __name__ == "__main__":
    raise SystemExit(main())
