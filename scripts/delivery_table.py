"""What reached the user, arm by arm, on the reference task. Offline.

The EMSE review of 2026-10-01 asked for content retention beside the
violation rates: a response with no claims satisfies C1-C2 vacuously, and an
abstention escapes C3. From benchmark/unified_scoring.csv this writes, per
arm, the share of responses with any violation, the mean number of claims
delivered, the share delivering at least one claim, abstentions, responses
with no admissible verdict (aborted or empty), and responses retried. The
2026-10-02 review asked how much correct content survives: the mean number of
claims that match E within tau, and the share of responses that report the
anchor's measured value, are added.

    python -X utf8 scripts/delivery_table.py [--tex paper/tse_latex/table_delivery_rows.tex]
"""

from __future__ import annotations

import argparse
import csv
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CELLS = [  # (run_dir, task, arm_id)
    (".", "synthetic", "A-L1"),
    ("2026-09-24_revision", "synthetic", "A-L2"),
    (".", "synthetic", "A-GR-STOCK"),
    (".", "synthetic", "A-STRICT-STATIC"),
    (".", "synthetic", "A-GR-OURS"),
    (".", "synthetic", "A-CONTRACT"),
    (".", "synthetic", "A-STRESS"),
    ("2026-09-24_revision", "synthetic", "A-L3-SHIPPED"),
    ("2026-09-30_strong", "synthetic", "A-STRONG-NOVERIFY"),
    ("2026-09-30_strong", "synthetic", "A-STRONG-CONTRACT"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=None)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(ROOT / "benchmark" / "unified_scoring.csv", encoding="utf-8")))
    out = []
    for run_dir, task, arm in CELLS:
        cell = [r for r in rows if r["run_dir"] == run_dir and r["task"] == task
                and r["arm_id"] == arm and r["provider"] == "deepseek"]
        n = len(cell)
        assert n == 200, (run_dir, arm, n)
        anyv = sum(r["any"] == "True" for r in cell)
        claims = sum(int(r["claims"]) for r in cell)
        with_claim = sum(int(r["claims"]) > 0 for r in cell)
        abstain = sum(r["verdict_admissible"] == "True" and r["committed"] == "False" for r in cell)
        no_verdict = sum(r["verdict_admissible"] == "False" and r["committed"] == "False" for r in cell)
        retried = sum(r["retried"] == "True" for r in cell)
        correct = sum(int(r["claims_correct"]) for r in cell)
        anchor = sum(r["anchor_value"] == "True" for r in cell)
        out.append((arm, n, anyv, claims / n, correct / n, 100 * anchor / n, with_claim,
                    abstain, no_verdict, retried))
        print(f"{arm:20s} N={n} any={anyv:3d} claims/resp={claims / n:5.2f} correct/resp={correct / n:5.2f} "
              f"anchor={100 * anchor / n:5.1f}% with>=1 claim={with_claim:3d} "
              f"abstain={abstain:2d} no-verdict={no_verdict:2d} retried={retried:3d}")
    if args.tex:
        lines = [f"\\texttt{{{a}}} & {100 * v / n:.1f} & {c:.1f} & {ok:.1f} & {an:.1f} & {w} & {ab} "
                 f"& {nv} & {rt} \\\\"
                 for a, n, v, c, ok, an, w, ab, nv, rt in out]
        pathlib.Path(args.tex).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {args.tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
