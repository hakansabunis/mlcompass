"""Cross-run heterogeneity for the arms the manuscript reports on several dates.

Review item P0-8: the manuscript read 35.0% -> 43.5% as drift without a test,
called the unconstrained rate "unstable" while A-L1 did not move, and dated a
run wrongly. This recomputes every count from the run records (transport errors
excluded) and tests each series with a chi-square on the k x 2 table, plus
pairwise Fisher exact tests. Offline.

    python -X utf8 scripts/drift_tests.py [--json benchmark/drift_tests.json]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

from scipy.stats import chi2_contingency, fisher_exact

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT / "scripts" / "runs"
PAT = "deepseek_deepseek-chat_synthetic_{arm}_n200_seed0_e618bd74a.jsonl"

# (label, directory, arm file stem, first record's calendar date is read from provenance)
SERIES = {
    "A-L1 entity": [
        (R, "layer1"),
        (R / "2026-09-18_choices", "layer1"),
        (R / "2026-09-19_freetext", "layer1"),
        (R / "2026-09-24_revision", "layer1"),
    ],
    "A-GR-STOCK entity": [
        (R, "guardrails_stock"),
        (R / "2026-09-18_choices", "guardrails_stock"),
        (R / "2026-09-19_freetext", "guardrails_stock"),
    ],
    "A-STRESS retried": [
        (R, "layer3_stress"),
        (R / "2026-09-18_choices", "layer3_stress"),
        (R / "2026-09-19_freetext", "layer3_stress"),
    ],
}


def wilson(k: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (c - r), 1), round(100 * (c + r), 1))


def count(path: pathlib.Path, label: str) -> tuple[int, int, str]:
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    recs = [r for r in recs if not r.get("error")]
    if label.endswith("retried"):
        k = sum(1 for r in recs if int(r.get("rejections") or 0) > 0)
    else:
        k = sum(1 for r in recs if (r.get("scored") or {}).get("entity"))
    started = str((recs[0].get("provenance") or {}).get("started_at", ""))[:10] if recs else ""
    return k, len(recs), started


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    out = {}
    for label, cells in SERIES.items():
        rows = []
        for d, arm in cells:
            f = d / PAT.format(arm=arm)
            if f.exists():
                k, n, started = count(f, label)
                rows.append({"dir": d.name if d != R else "root", "started": started, "k": k, "n": n})
        table = [[r["k"], r["n"] - r["k"]] for r in rows]
        chi, p, dof, _ = chi2_contingency(table)
        K, N = sum(r["k"] for r in rows), sum(r["n"] for r in rows)
        pairs = []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                pv = fisher_exact([table[i], table[j]])[1]
                pairs.append({"a": rows[i]["started"], "b": rows[j]["started"], "fisher_p": round(pv, 4)})
        out[label] = {"runs": rows, "chi2": round(chi, 2), "dof": dof, "p": round(p, 4),
                      "pooled": [K, N, round(100 * K / N, 1), wilson(K, N)], "pairs": pairs}
        print(f"{label}: " + ", ".join(f"{r['started']} {r['k']}/{r['n']}" for r in rows)
              + f" | chi2={chi:.2f} dof={dof} p={p:.4f} | pooled {K}/{N} = "
              f"{100 * K / N:.1f}% {wilson(K, N)}")
        for pr in pairs:
            print(f"    {pr['a']} vs {pr['b']}: Fisher p={pr['fisher_p']}")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
