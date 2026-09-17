"""Run both defect scorers over every preserved script and report where they
disagree.

Agreement is reported per rule with Cohen's kappa, and every disagreeing run is
listed with its file and line so a reader can open it and decide for themselves.
The disagreement set is the output that matters: on the two inverted rules it
localises exactly where the first scorer's enumeration of correct forms runs
out.

Usage:
    python benchmark/audit_defect_scorers.py [--panel]

--panel restricts to the 143 runs the paper reports; the default covers every
preserved run directory, which is a wider net and a harder test.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import independent_defect_scorer as second  # noqa: E402
from rescore_ab import _target_from_split  # noqa: E402
from run_ab import check_defects  # noqa: E402

# The scoring version the manuscript reports. Bump when a rescore lands.
PANEL = {"v4", "v4cr"}
RULES = second.RULES


def kappa(a: list[bool], b: list[bool]) -> float | None:
    """Cohen's kappa for two binary raters. None when it is undefined -- both
    raters constant and agreeing, where every sensible kappa convention
    disagrees with every other."""
    n = len(a)
    if n == 0:
        return None
    po = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", action="store_true")
    ap.add_argument("--json", type=pathlib.Path, default=HERE / "scorer_audit.json")
    args = ap.parse_args()

    rows = list((HERE / "ab_results.csv").open(encoding="utf-8"))
    rows = list(csv.DictReader(rows))
    if args.panel:
        rows = [r for r in rows if r["experiment_id"] in PANEL]

    seen: set[str] = set()
    first_flags: dict[str, list[bool]] = {r: [] for r in RULES}
    second_flags: dict[str, list[bool]] = {r: [] for r in RULES}
    disagreements: list[dict] = []
    parse_errors: list[str] = []
    scored = 0

    for r in rows:
        d = HERE / r["artifacts_path"]
        script = d / "emitted.py"
        runjson = d / "run.json"
        if str(d) in seen or not script.exists() or not runjson.exists():
            continue
        seen.add(str(d))
        run = json.loads(runjson.read_text(encoding="utf-8"))
        blk = run.get("defects") or {}
        target = run.get("target") or (run.get("split") or {}).get("target")
        if not target:
            target = _target_from_split(run)
        need = ("input_duplicate_rows", "input_minority_fraction", "target_is_last_column")
        if not target or any(k not in blk for k in need):
            continue
        facts = dict(
            target=target,
            duplicate_rows=blk["input_duplicate_rows"],
            minority_fraction=blk["input_minority_fraction"],
            target_is_last_column=blk["target_is_last_column"],
        )
        src = script.read_text(encoding="utf-8", errors="replace")
        f1 = check_defects(src, **facts)["flags"]
        out2 = second.score(src, **facts)
        if out2["flags"] is None:
            parse_errors.append(f"{d.name}: {out2['parse_error']}")
            continue
        f2 = out2["flags"]
        scored += 1
        for rule in RULES:
            first_flags[rule].append(bool(f1[rule]))
            second_flags[rule].append(bool(f2[rule]))
            if bool(f1[rule]) != bool(f2[rule]):
                disagreements.append(
                    {
                        "run": d.name,
                        "arm": r["arm"],
                        "rule": rule,
                        "regex_scorer": bool(f1[rule]),
                        "ast_scorer": bool(f2[rule]),
                        "script": str(script),
                    }
                )

    print(f"scripts scored by both: {scored}")
    if parse_errors:
        print(f"scripts the ast scorer could not parse: {len(parse_errors)}")
        for p in parse_errors[:5]:
            print("   ", p)

    print(f"\n{'rule':<30}{'regex':>7}{'ast':>6}{'agree':>8}{'kappa':>8}")
    for rule in RULES:
        a, b = first_flags[rule], second_flags[rule]
        agree = sum(1 for x, y in zip(a, b, strict=True) if x == y)
        k = kappa(a, b)
        ks = "  n/a" if k is None else f"{k:>7.3f}"
        print(f"{rule:<30}{sum(a):>7}{sum(b):>6}{agree:>5}/{len(a):<3}{ks}")

    tot = sum(
        1
        for rule in RULES
        for x, y in zip(first_flags[rule], second_flags[rule], strict=True)
        if x != y
    )
    print(f"\ntotal rule-instance disagreements: {tot} of {scored * len(RULES)}")

    by = collections.Counter((d["rule"], d["regex_scorer"], d["ast_scorer"]) for d in disagreements)
    if by:
        print("\ndisagreement shape:")
        for (rule, r1, _r2), n in by.most_common():
            direction = "regex flags, ast clean" if r1 else "ast flags, regex clean"
            print(f"  {rule:<30}{direction:<26}{n}")

    args.json.write_text(
        json.dumps(
            {
                "scripts_scored": scored,
                "parse_errors": parse_errors,
                "per_rule": {
                    rule: {
                        "regex_flagged": sum(first_flags[rule]),
                        "ast_flagged": sum(second_flags[rule]),
                        "agreements": sum(
                            1
                            for x, y in zip(first_flags[rule], second_flags[rule], strict=True)
                            if x == y
                        ),
                        "n": len(first_flags[rule]),
                        "kappa": kappa(first_flags[rule], second_flags[rule]),
                    }
                    for rule in RULES
                },
                "disagreements": disagreements,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
