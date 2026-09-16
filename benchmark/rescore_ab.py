"""Re-score the A/B battery's preserved runs without re-executing anything.

Why this exists. `ab_protocol.md` §7 makes the emitted script the artefact
under study and keeps every one of them, precisely so a scoring defect found
later can be corrected against the same evidence rather than by spending
money again. This is the first time that provision has been used.

The defect, found by Yusuf Ünlü in review of the 108-run battery: the
`target_in_features` rule scores the *absence* of every enumerated way of
excluding the target, and the list of forms did not include a comprehension:

    feature_cols = [c for c in df.columns if c != target_col]

so scripts that correctly excluded the target were counted as defective.
Fifteen of the 108 preserved runs carried the false positive (amendment A7).

What this does, and what it deliberately does not do. It re-runs
`check_defects` over each preserved `emitted.py`, using the dataset facts
already recorded in that run's `run.json` — duplicate-row count, minority
fraction, whether the target is the last column. No dataset is re-read, no
model is called, no script is executed. That makes the rescore reproducible
from the repository alone and keeps it honest: nothing about the run can
change except the verdict the checker reaches about the same bytes.

Originals are never touched. New rows arrive under a new experiment id and
each run gains `scoring_rescored.md` alongside its original `scoring.md`, so
the correction reads as a correction and not as a number that quietly
improved. `holdout_score` is copied through unchanged — this pass revises
the defect channel and nothing else.

    python benchmark/rescore_ab.py --list
    python benchmark/rescore_ab.py ab-20260915-3b0221 --dry-run
    python benchmark/rescore_ab.py ab-20260915-3b0221
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_ab import DEFECT_IDS, check_defects  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
RESULTS = HERE / "ab_results.csv"


def _experiment_of(run_dir: Path) -> str:
    """`ab-20260915-3b0221-1464-control-deepseek-flash-r1` -> the first three parts."""
    return "-".join(run_dir.name.split("-")[:3])


def list_experiments() -> dict[str, int]:
    counts: dict[str, int] = {}
    for d in sorted(RUNS.glob("ab-*")):
        if d.is_dir() and (d / "run.json").exists():
            counts[_experiment_of(d)] = counts.get(_experiment_of(d), 0) + 1
    return counts


def existing_experiment_ids() -> set[str]:
    if not RESULTS.exists():
        return set()
    with RESULTS.open(encoding="utf-8", newline="") as f:
        return {row["experiment_id"] for row in csv.DictReader(f)}


def rescore_one(run_dir: Path) -> dict[str, Any] | None:
    """Re-run the checker over one preserved run. None if it cannot be rescored."""
    script = run_dir / "emitted.py"
    record = run_dir / "run.json"
    if not script.exists() or not record.exists():
        return None
    run = json.loads(record.read_text(encoding="utf-8"))
    old = run.get("defects")
    if not old:
        return None

    # The dataset facts the checker needs were recorded at scoring time. Read
    # them back rather than recomputing from the CSV: recomputing would let a
    # changed dataset silently alter a historical verdict, which is the exact
    # failure the pinned manifest exists to prevent.
    missing = [
        k
        for k in ("input_duplicate_rows", "input_minority_fraction", "target_is_last_column")
        if k not in old
    ]
    if missing:
        print(f"  skip     {run_dir.name}: run.json lacks {', '.join(missing)}")
        return None

    target = run.get("target") or _target_from_split(run)
    if not target:
        print(f"  skip     {run_dir.name}: no target column recorded")
        return None

    facts = {
        "target": target,
        "duplicate_rows": old["input_duplicate_rows"],
        "minority_fraction": old["input_minority_fraction"],
        "target_is_last_column": old["target_is_last_column"],
    }
    new = check_defects(script.read_text(encoding="utf-8", errors="replace"), **facts)

    # Both sides of a revision turn, or the pairing spans two scorer versions
    # and the reader gets a different answer depending on which columns they
    # use. Arms without a revision round simply have no such file.
    pre_script = run_dir / "emitted_pre_revision.py"
    new_pre = None
    if pre_script.exists():
        new_pre = check_defects(
            pre_script.read_text(encoding="utf-8", errors="replace"), **facts
        )

    return {"run": run, "old": old, "new": new, "new_pre": new_pre, "target": target}


def _target_from_split(run: dict[str, Any]) -> str | None:
    """The target is the last column listed in the split record."""
    columns = (run.get("split") or {}).get("columns")
    if not columns:
        return None
    names = [
        line.strip().lstrip("- ").split(" (")[0]
        for line in columns.splitlines()
        if line.strip().startswith("-")
    ]
    return names[-1] if names else None


def _scoring_markdown(run_dir: Path, result: dict[str, Any], new_experiment: str) -> str:
    old, new = result["old"], result["new"]
    changed = [d for d in DEFECT_IDS if bool(old["flags"][d]) != bool(new["flags"][d])]
    lines = [
        f"# Rescored: {run_dir.name}",
        "",
        f"Experiment id for these rows: `{new_experiment}`.",
        "",
        "Re-scored from the preserved `emitted.py` with the corrected",
        "`target_in_features` rule (amendment A7). Nothing was re-executed and",
        "no model was called; the original `scoring.md` is kept beside this file.",
        "",
        f"Defect count: **{old['defect_count']} -> {new['defect_count']}**",
        "",
        "| flag | before | after |",
        "| --- | :---: | :---: |",
    ]
    for d in DEFECT_IDS:
        a, b = bool(old["flags"][d]), bool(new["flags"][d])
        mark = " **<-- changed**" if a != b else ""
        lines.append(f"| `{d}` | {'yes' if a else 'no'} | {'yes' if b else 'no'}{mark} |")
    lines += [
        "",
        (
            "No flag changed."
            if not changed
            else f"Changed: {', '.join('`' + c + '`' for c in changed)}."
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("experiment_id", nargs="?", help="experiment to rescore")
    ap.add_argument("--rescore-id", default=None, help="id for the new rows; default <id>-rescored")
    ap.add_argument("--list", action="store_true", help="list preserved experiments and exit")
    ap.add_argument("--dry-run", action="store_true", help="report the changes, write nothing")
    args = ap.parse_args()

    if args.list:
        for exp, n in sorted(list_experiments().items()):
            print(f"  {exp}  {n} run(s)")
        return 0
    if not args.experiment_id:
        ap.error("give an experiment id, or --list")

    new_experiment = args.rescore_id or f"{args.experiment_id}-rescored"
    if not args.dry_run and new_experiment in existing_experiment_ids():
        print(
            f"{new_experiment} already has rows in {RESULTS.name}; rescoring into it again\n"
            "would duplicate every one of them. Pass --rescore-id with a different id.",
            file=sys.stderr,
        )
        return 1

    run_dirs = sorted(d for d in RUNS.glob(f"{args.experiment_id}-*") if d.is_dir())
    if not run_dirs:
        print(f"no preserved runs under {RUNS} for {args.experiment_id}", file=sys.stderr)
        return 1

    rows, changed_runs, scanned = [], [], 0
    for run_dir in run_dirs:
        result = rescore_one(run_dir)
        if result is None:
            continue
        scanned += 1
        run, old, new = result["run"], result["old"], result["new"]
        if old["defect_count"] != new["defect_count"]:
            changed_runs.append((run_dir.name, old["defect_count"], new["defect_count"]))
        if not args.dry_run:
            (run_dir / "scoring_rescored.md").write_text(
                _scoring_markdown(run_dir, result, new_experiment), encoding="utf-8"
            )
        rows.append((run_dir, run, new, result["new_pre"]))

    print(f"\nscanned {scanned} preserved run(s); {len(changed_runs)} changed")
    for name, a, b in changed_runs:
        print(f"  {name}: {a} -> {b}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    _append_rows(rows, new_experiment)
    print(f"\nappended {len(rows)} row(s) to {RESULTS.name} under experiment_id {new_experiment}")
    print("original rows and every original scoring.md are untouched.")
    return 0


def _append_rows(
    rows: list[tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any] | None]],
    new_experiment: str,
) -> None:
    with RESULTS.open(encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    existing = {}
    with RESULTS.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            existing[row["run_id"]] = row

    out = []
    for run_dir, _run, new, new_pre in rows:
        base = dict(existing.get(run_dir.name) or {})
        if not base:
            print(f"  skip     {run_dir.name}: no original CSV row to carry forward")
            continue
        base["experiment_id"] = new_experiment
        base["defect_count"] = str(new["defect_count"])
        if new_pre is not None:
            # Rescored under the same version as the post-revision side, so the
            # two are comparable.
            base["defect_count_pre_revision"] = str(new_pre["defect_count"])
            base["defect_count_post_revision"] = str(new["defect_count"])
        for d in DEFECT_IDS:
            base[d] = str(int(bool(new["flags"][d])))
        note = "rescored from the preserved emitted.py under amendment A7; not re-executed"
        base["notes"] = f"{base.get('notes', '')}; {note}".lstrip("; ")
        out.append([base.get(c, "") for c in header])

    with RESULTS.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(out)


if __name__ == "__main__":
    sys.exit(main())
