"""Score human ratings against the checklist, once two raters have filled in
their sheets.

Reports, per rule: the checklist's precision and recall against the human
reference, and Cohen's kappa between the two raters. Where the raters disagree
with each other, the rule is not scored against the checklist at all -- a
reference the raters cannot agree on is not a reference, and averaging them
into one would hide exactly the ambiguity worth reporting.

Usage:
    python benchmark/blind_review/score_blind_review.py rater_a.csv rater_b.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
RULES = (
    "no_seed",
    "leak_fit_before_split",
    "leak_duplicate_rows",
    "wrong_metric_for_imbalance",
    "no_validation",
    "target_in_features",
)
TRUE = {"1", "y", "yes", "true", "t"}
FALSE = {"0", "n", "no", "false", "f"}


def parse_cell(value: str, where: str) -> bool | None:
    v = (value or "").strip().lower()
    if v in TRUE:
        return True
    if v in FALSE:
        return False
    if v == "":
        return None
    raise SystemExit(f"{where}: cannot read {value!r}; use 1/0 or yes/no")


def load(path: pathlib.Path) -> dict[str, dict[str, bool | None]]:
    out: dict[str, dict[str, bool | None]] = {}
    for row in csv.DictReader(path.open(encoding="utf-8")):
        bid = row["blind_id"].strip()
        out[bid] = {r: parse_cell(row.get(r, ""), f"{path.name}:{bid}:{r}") for r in RULES}
    return out


def kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe >= 1.0 else (po - pe) / (1 - pe)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rater_a", type=pathlib.Path)
    ap.add_argument("rater_b", type=pathlib.Path)
    ap.add_argument(
        "--key",
        type=pathlib.Path,
        default=HERE / "UNBLINDING_KEY_do_not_open_before_rating.csv",
    )
    args = ap.parse_args()

    a, b = load(args.rater_a), load(args.rater_b)
    key = {
        r["blind_id"]: json.loads(r["checklist_flags"])
        for r in csv.DictReader(args.key.open(encoding="utf-8"))
    }

    shared = sorted(set(a) & set(b) & set(key))
    print(f"scripts rated by both and present in the key: {len(shared)}")
    if not shared:
        return 1

    print(f"\n{'rule':<30}{'kappa':>8}{'agreed':>8}{'prec':>7}{'rec':>7}{'TP FP FN':>12}")
    for rule in RULES:
        ra = [a[s][rule] for s in shared]
        rb = [b[s][rule] for s in shared]
        both = [
            (s, x, y)
            for s, x, y in zip(shared, ra, rb, strict=True)
            if x is not None and y is not None
        ]
        if not both:
            print(f"{rule:<30}{'no data':>8}")
            continue
        k = kappa([x for _, x, _ in both], [y for _, _, y in both])
        agreed = [(s, x) for s, x, y in both if x == y]
        tp = sum(1 for s, human in agreed if human and key[s].get(rule))
        fp = sum(1 for s, human in agreed if not human and key[s].get(rule))
        fn = sum(1 for s, human in agreed if human and not key[s].get(rule))
        prec = tp / (tp + fp) if (tp + fp) else None
        rec = tp / (tp + fn) if (tp + fn) else None
        ks = "  n/a" if k is None else f"{k:>7.3f}"
        ps = "  n/a" if prec is None else f"{prec:>6.2f}"
        rs = "  n/a" if rec is None else f"{rec:>6.2f}"
        print(f"{rule:<30}{ks}{len(agreed):>5}/{len(both):<3}{ps}{rs}{tp:>5}{fp:>3}{fn:>3}")

    print(
        "\nRules where the two raters disagree are excluded from precision and "
        "recall for that rule, and the agreed/rated column shows how many were "
        "dropped."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
