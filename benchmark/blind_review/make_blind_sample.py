"""Freeze a blind, stratified sample of emitted scripts for human rating.

Why this exists. Section 10 of the paper reports six defects in instruments we
wrote, and the differential audit in the same section can only show that two
implementations *by the same authors* agree. The check that would actually
settle whether the six-item checklist measures what it claims needs raters who
are not us and who cannot see which arm produced a script.

What it does. Draws a stratified sample across (arm, dataset, model), strips
every label, rewrites the script to a neutral filename, shuffles the order
under a fixed seed, and writes a rating sheet. The mapping from blind id back
to run id is written to a separate file that raters must not open -- keeping it
out of the sample directory is the only thing standing between this and an
unblinded rating, so it is written one level up and named accordingly.

The reference flags in that key are computed here, by the current checker, over
the redacted script the rater actually reads. An earlier version copied them
out of each run.json, which records the scoring as it stood when the run
executed; a key frozen before amendment A16 therefore carried four
`target_in_features` false positives that A16 had already removed, and the
scoring script reported a precision of 0.00 against raters who were right.

Stripping is not cosmetic. An emitted script can carry its own provenance: a
temp path containing the dataset id, a comment quoting the audit findings the
advise+audit arm was given. Both are removed and the removal is logged, so a
rater cannot infer the arm from the artefact.

Usage:
    python benchmark/blind_review/make_blind_sample.py --per-cell 2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import random
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH))

from rescore_ab import _target_from_split  # noqa: E402
from run_ab import check_defects  # noqa: E402

PANEL = {"v4", "v4cr"}
RULES = (
    "no_seed",
    "leak_fit_before_split",
    "leak_duplicate_rows",
    "wrong_metric_for_imbalance",
    "no_validation",
    "target_in_features",
)

# Anything that could tell a rater which arm produced the script. The last
# three matter most and were not obvious: a model that was told what to fix
# says so in its own comments -- "# Remove duplicate rows as advised",
# "# Hold-out validation split (for audit compliance)" -- which hands the arm
# to the rater as plainly as a filename would.
_TELLS = (
    (re.compile(r"mlcab-\d+-seed\d+-r\d+"), "<TMPDIR>"),
    (re.compile(r"ab-20260915-[0-9a-f]{6}[^\s\"']*"), "<RUNDIR>"),
    (re.compile(r"(?i)\bmlcompass\b"), "<TOOL>"),
    (re.compile(r"(?i)\b(?:advis|audit)\w*\b"), "<REDACTED>"),
    (re.compile(r"(?i)\bas (?:suggested|recommended|instructed|requested)\b"), "<REDACTED>"),
    (re.compile(r"(?i)\bper the (?:tool|report|findings|feedback|checklist)\b"), "<REDACTED>"),
)


def strip_tells(source: str) -> tuple[str, list[str]]:
    removed: list[str] = []
    out = source
    for pattern, placeholder in _TELLS:
        hits = pattern.findall(out)
        if hits:
            removed.extend(str(h) for h in hits[:4])
            out = pattern.sub(placeholder, out)
    return out, removed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=2, help="scripts per (arm, dataset) cell")
    ap.add_argument("--seed", type=int, default=20260917)
    args = ap.parse_args()

    rows = [
        r
        for r in csv.DictReader((BENCH / "ab_results.csv").open(encoding="utf-8"))
        if r["experiment_id"] in PANEL
    ]
    cells: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        script = BENCH / r["artifacts_path"] / "emitted.py"
        if script.exists():
            cells.setdefault((r["arm"], r["dataset_id"]), []).append(r)

    rng = random.Random(args.seed)
    picked: list[dict] = []
    for key in sorted(cells):
        pool = sorted(cells[key], key=lambda r: r["run_id"])
        picked.extend(rng.sample(pool, min(args.per_cell, len(pool))))
    rng.shuffle(picked)

    out = HERE / "sample"
    out.mkdir(exist_ok=True)
    for stale in out.glob("*.py"):
        stale.unlink()

    key_rows = []
    sheet_rows = []
    for i, r in enumerate(picked, start=1):
        blind_id = f"S{i:03d}"
        src = (BENCH / r["artifacts_path"] / "emitted.py").read_text(
            encoding="utf-8", errors="replace"
        )
        clean, removed = strip_tells(src)
        (out / f"{blind_id}.py").write_text(clean, encoding="utf-8")
        run = json.loads((BENCH / r["artifacts_path"] / "run.json").read_text(encoding="utf-8"))
        blk = run.get("defects") or {}
        target = (
            run.get("target") or (run.get("split") or {}).get("target") or _target_from_split(run)
        )

        # The key is SCORED HERE, by the current checker, over the REDACTED
        # source -- not read back from run.json.
        #
        # Reading it back was wrong twice over. run.json holds the scoring as
        # it stood when the run executed, so a key frozen after amendment A16
        # still carried the four `target_in_features` false positives A16
        # removed, and `score_blind_review.py` as shipped reported "precision
        # 0.00" against raters who were right. And the raters see the redacted
        # file, so the key has to score the redacted file: redaction touches
        # only comments and string literals, but a key computed from the
        # original could not detect it if that ever stopped being true.
        fresh = check_defects(
            clean,
            target=target,
            duplicate_rows=int(blk.get("input_duplicate_rows") or 0),
            minority_fraction=blk.get("input_minority_fraction"),
            target_is_last_column=bool(blk.get("target_is_last_column")),
        )
        flags = {k: bool(v) for k, v in (fresh.get("flags") or {}).items()}

        # Redaction must not move a rule. If it does, the sample is unusable
        # and refusing here is the only safe outcome -- a silently altered
        # reference is worse than no reference.
        on_original = check_defects(
            src,
            target=target,
            duplicate_rows=int(blk.get("input_duplicate_rows") or 0),
            minority_fraction=blk.get("input_minority_fraction"),
            target_is_last_column=bool(blk.get("target_is_last_column")),
        )
        moved = {
            k
            for k in flags
            if flags[k] != bool((on_original.get("flags") or {}).get(k))
        }
        if moved:
            print(
                f"REFUSING: redaction changed {sorted(moved)} on {blind_id} "
                f"({r['run_id']}). The sample would carry a wrong reference.",
                file=sys.stderr,
            )
            return 1
        key_rows.append(
            {
                "blind_id": blind_id,
                "run_id": r["run_id"],
                "arm": r["arm"],
                "dataset_id": r["dataset_id"],
                "model": r["panel_id"],
                "checklist_flags": json.dumps(flags),
                "sha256": hashlib.sha256(clean.encode("utf-8")).hexdigest(),
                "tells_removed": "; ".join(removed),
            }
        )
        sheet_rows.append(
            {
                "blind_id": blind_id,
                "target_column": target or "",
                "input_has_duplicate_rows": blk.get("input_duplicate_rows", ""),
                "minority_fraction": blk.get("input_minority_fraction", ""),
                **{rule: "" for rule in RULES},
                "notes": "",
            }
        )

    with (out / "rating_sheet.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sheet_rows[0].keys()))
        w.writeheader()
        w.writerows(sheet_rows)

    # One level up, deliberately outside the directory a rater is given.
    with (HERE / "UNBLINDING_KEY_do_not_open_before_rating.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(key_rows[0].keys()))
        w.writeheader()
        w.writerows(key_rows)

    # A blinding that is not checked is not a blinding. Re-read what was
    # written and fail loudly rather than shipping a sample that names its arms.
    survivors = []
    for script in sorted(out.glob("*.py")):
        text = script.read_text(encoding="utf-8")
        for pattern, _ in _TELLS:
            for hit in pattern.findall(text):
                survivors.append(f"{script.name}: {hit!r}")
    if survivors:
        print("BLINDING FAILED -- these would tell a rater the arm:")
        for s in survivors:
            print("   ", s)
        return 1

    by_arm: dict[str, int] = {}
    for k in key_rows:
        by_arm[k["arm"]] = by_arm.get(k["arm"], 0) + 1
    print(f"{len(picked)} scripts -> {out}")
    print("per arm:", by_arm)
    print(f"rating sheet: {out / 'rating_sheet.csv'}")
    print("key written one level up; do not open it before rating")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
