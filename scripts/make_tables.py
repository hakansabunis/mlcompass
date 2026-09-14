"""Build manuscript tables from committed run logs (Q1_ROADMAP.md R8).

Every number in every results table must derive mechanically from a committed
JSONL run log — no hand-typed rates. This script walks a runs directory,
re-scores every cell with the INDEPENDENT scorer (analysis_plan A3.9a; the
harness's inline flags are only cross-checked, never used for the tables),
and emits one Markdown table per (task, provider, model) group plus a
combined file. Each output carries the git commit of the working tree and a
dirty flag, so a table can always be traced back to the exact logs.

Since the Round-2 baseline block (analysis_plan_2026-09 §2) it also emits an
arm-by-channel enforcement-mechanism comparison, ordered by enforcement
strength, carrying the composite user-facing violation rate the plan's primary
contrast uses plus the withheld-answer and calls-per-response columns §2.3
requires beside the rates. That file appears only when baseline cells exist.

Usage:
    python scripts/make_tables.py --runs-dir scripts/runs
    python scripts/make_tables.py --runs-dir scripts/runs --out-dir paper/tables --latex
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import re
import subprocess
import sys
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))

_spec = importlib.util.spec_from_file_location(
    "independent_scorer", os.path.join(_HERE, "independent_scorer.py")
)
assert _spec is not None and _spec.loader is not None
scorer = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("independent_scorer", scorer)
_spec.loader.exec_module(scorer)


def _git_stamp() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_HERE,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=_HERE,
            check=True,
        ).stdout.strip()
        return f"{sha}{'+dirty' if dirty else ''}"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _fmt(ch: dict[str, Any] | None) -> str:
    if ch is None:
        return "n/a"
    lo, hi = ch["ci95"]
    return f"{ch['k']}/{ch['n']} ({ch['rate'] * 100:.1f}%, CI [{lo * 100:.2f}, {hi * 100:.2f}])"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "unknown"


# --------------------------------------------------------------------------- #
# Baseline comparison (analysis_plan_2026-09 §2) — arm by channel              #
# --------------------------------------------------------------------------- #

# The enforcement ladder, weakest first. Rows are ordered by this rather than
# alphabetically so a reader walks mechanisms, not arm-name spellings. Arms not
# listed sort after these, alphabetically.
ARM_ORDER: tuple[str, ...] = (
    "layer1",
    "guardrails_stock",
    "static_schema",
    "layer1+strict",
    "tier_a",
    "tier_a+strict",
    "guardrails_tierb",
    "layer3_stress",
    "layer3_stress_generic",
    "layer3_bare",
    "layer3",
)

# Arms that carry a baseline mechanism. A run directory holding none of these
# has nothing to compare, and the comparison table is skipped rather than
# emitted empty.
BASELINE_ARM_NAMES: frozenset[str] = frozenset(
    {"guardrails_stock", "guardrails_tierb", "static_schema"}
)

# Which cells are ANALYTIC rather than empirical on the entity/value channels:
# Tier B strips, so a zero there is a consistency check on an invariant that
# holds by construction, not an estimated rate (analysis_plan_2026-09 §1.2,
# §1.3). The plan requires this label in the table itself, not in a footnote.
TIER_B_ARMS: frozenset[str] = frozenset(
    {"layer3", "layer3_bare", "layer3_stress", "layer3_stress_generic", "stress_mech"}
)


def _arm_rank(arm: str) -> tuple[int, str]:
    base = arm.removesuffix("+strict")
    for index, known in enumerate(ARM_ORDER):
        if arm == known:
            return (index, arm)
    for index, known in enumerate(ARM_ORDER):
        if base == known:
            return (index, arm)
    return (len(ARM_ORDER), arm)


def _label(cell: dict[str, Any]) -> str:
    """Analytic-or-empirical label, required on every row by §1.3."""
    return (
        "analytic"
        if cell["identity"]["arm"].removesuffix("+strict") in TIER_B_ARMS
        else "empirical"
    )


def _calls(cell: dict[str, Any]) -> str:
    value = cell.get("calls_per_response")
    return f"{value:.2f}" if isinstance(value, (int, float)) else "n/a"


def build_baseline_comparison(groups: dict[tuple[str, str, str], list[dict[str, Any]]]) -> str:
    """Arm-by-channel comparison, one block per (task, provider, model) cell.

    Every number here is recomputed by the independent scorer from the cell's
    own records; nothing is transcribed. Groups without a baseline arm are
    skipped, so this file reports comparisons that actually ran.
    """
    blocks: list[str] = [
        "# Baseline comparison -- enforcement mechanisms",
        "",
        "Arms differ only in the enforcement mechanism: same task, same evidence",
        "hash, same model pin, same sampling pin, same scorer. Rows are ordered by",
        "enforcement strength, not alphabetically. `composite` is any user-facing",
        "violation (entity OR value OR flagged omission) -- the outcome measure the",
        "Round-2 plan fixes for its primary contrast (analysis_plan_2026-09 section 2.2).",
        "",
        "`label` marks whether a cell's entity/value zeros are ANALYTIC (Tier B",
        "strips, so the zero is a consistency check on an invariant that holds by",
        "construction) or EMPIRICAL (sections 1.2, 1.3). `no answer` is the share of",
        "responses where a validate-and-reask toolkit withheld the answer instead",
        "of repairing it: withholding is not fabricating, so it is reported beside",
        "the channels rather than folded into them. `calls/rsp` is provider calls",
        "spent per response reaching the user (section 2.3 requires cost beside rates).",
        "",
    ]

    emitted = 0
    for key, cells in sorted(groups.items()):
        if not any(
            c["identity"]["arm"].removesuffix("+strict") in BASELINE_ARM_NAMES for c in cells
        ):
            continue
        emitted += 1
        task, provider, model = key
        evidence_files = sorted({c["evidence_file"] for c in cells})
        blocks += [
            f"### {task} -- {provider} / {model}",
            "",
            f"Evidence: {', '.join(evidence_files)}",
            "",
            "| arm | mechanism | label | N | err | composite | entity | value "
            "| omission | abstained | no answer | calls/rsp | catches |",
            "| --- | --- | --- | --: | --: | --- | --- | --- | --- | --- | --- | --: | --: |",
        ]
        for cell in sorted(cells, key=lambda c: _arm_rank(c["identity"]["arm"])):
            blocks.append(
                f"| {cell['identity']['arm']} | {cell.get('mechanism') or '-'} "
                f"| {_label(cell)} | {cell['n']} | {cell['errors_excluded']} "
                f"| {_fmt(cell['composite'])} | {_fmt(cell['entity'])} "
                f"| {_fmt(cell['value'])} | {_fmt(cell['omission'])} "
                f"| {_fmt(cell['abstained'])} | {_fmt(cell.get('no_output'))} "
                f"| {_calls(cell)} | {cell['tier_b']['total_catches']} |"
            )
        blocks.append("")

    if not emitted:
        return ""
    return "\n".join(blocks)


def build_baseline_comparison_latex(
    groups: dict[tuple[str, str, str], list[dict[str, Any]]],
) -> str:
    def _tex(ch: dict[str, Any] | None) -> str:
        if ch is None:
            return "n/a"
        lo, hi = ch["ci95"]
        return f"{ch['k']}/{ch['n']} ({ch['rate'] * 100:.1f}\\%, [{lo * 100:.2f}, {hi * 100:.2f}])"

    out: list[str] = []
    for key, cells in sorted(groups.items()):
        if not any(
            c["identity"]["arm"].removesuffix("+strict") in BASELINE_ARM_NAMES for c in cells
        ):
            continue
        task, provider, model = key
        rows = [
            f"    {cell['identity']['arm'].replace('_', chr(92) + '_')} & {_label(cell)} "
            f"& {cell['n']} & {_tex(cell['composite'])} & {_tex(cell['entity'])} "
            f"& {_tex(cell['value'])} & {_tex(cell['omission'])} "
            f"& {_tex(cell.get('no_output'))} & {_calls(cell)} \\\\"
            for cell in sorted(cells, key=lambda c: _arm_rank(c["identity"]["arm"]))
        ]
        caption = f"Enforcement-mechanism comparison, {task}: {provider} / {model}".replace(
            "_", "\\_"
        )
        out += [
            "\\begin{table}[t]",
            "  \\centering",
            f"  \\caption{{{caption}}}",
            "  \\small",
            "  \\begin{tabular}{llrlllllr}",
            "    \\toprule",
            "    arm & label & $N$ & composite & entity & value & omission "
            "& no answer & calls \\\\",
            "    \\midrule",
            *rows,
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
            "",
        ]
    return "\n".join(out)


def build_group_markdown(key: tuple[str, str, str], cells: list[dict[str, Any]]) -> str:
    task, provider, model = key
    lines = [
        f"### {task} -- {provider} / {model}",
        "",
        "| arm | label | N | err | composite | entity | value | omission "
        "| abstained | Tier B catches |",
        "| --- | --- | --: | --: | --- | --- | --- | --- | --- | --: |",
    ]
    for cell in sorted(cells, key=lambda c: _arm_rank(c["identity"]["arm"])):
        ident = cell["identity"]
        tb = cell["tier_b"]
        lines.append(
            f"| {ident['arm']} | {_label(cell)} | {cell['n']} | {cell['errors_excluded']} "
            f"| {_fmt(cell['composite'])} | {_fmt(cell['entity'])} | {_fmt(cell['value'])} "
            f"| {_fmt(cell['omission'])} | {_fmt(cell['abstained'])} "
            f"| {tb['total_catches']} |"
        )
    disagreements = sum(
        sum(c["agreement"]["disagreements"].values()) for c in cells if c["agreement"]["compared"]
    )
    compared = sum(c["agreement"]["compared"] for c in cells)
    lines += [
        "",
        f"Scorer cross-check: {compared} records compared against the harness's "
        f"inline flags, {disagreements} disagreement(s).",
        "",
    ]
    return "\n".join(lines)


def build_group_latex(key: tuple[str, str, str], cells: list[dict[str, Any]]) -> str:
    task, provider, model = key

    def _tex(ch: dict[str, Any] | None) -> str:
        if ch is None:
            return "n/a"
        lo, hi = ch["ci95"]
        return f"{ch['k']}/{ch['n']} ({ch['rate'] * 100:.1f}\\%, [{lo * 100:.2f}, {hi * 100:.2f}])"

    rows = []
    for cell in sorted(cells, key=lambda c: c["identity"]["arm"]):
        ident = cell["identity"]
        arm = ident["arm"].replace("_", "\\_")
        rows.append(
            f"    {arm} & {cell['n']} & {cell['errors_excluded']} & {_tex(cell['entity'])} "
            f"& {_tex(cell['value'])} & {_tex(cell['omission'])} "
            f"& {_tex(cell['abstained'])} \\\\"
        )
    caption = f"{task}: {provider} / {model}".replace("_", "\\_")
    return "\n".join(
        [
            "\\begin{table}[t]",
            "  \\centering",
            f"  \\caption{{{caption}}}",
            "  \\small",
            "  \\begin{tabular}{lrrllll}",
            "    \\toprule",
            "    arm & $N$ & err & entity & value & omission & abstained \\\\",
            "    \\midrule",
            *rows,
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
            "",
        ]
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Build manuscript tables from run logs (R8).")
    ap.add_argument("--runs-dir", required=True, help="Directory with *.jsonl run logs.")
    ap.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(_HERE), "paper", "tables"),
        help="Output directory (default: paper/tables).",
    )
    ap.add_argument("--latex", action="store_true", help="Also emit LaTeX (booktabs) tables.")
    args = ap.parse_args()

    logs = sorted(
        p
        for p in glob.glob(os.path.join(args.runs_dir, "*.jsonl"))
        if not os.path.basename(p).startswith("evidence_")
    )
    if not logs:
        raise SystemExit(f"No run logs in {args.runs_dir}.")

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for path in logs:
        cell = scorer.score_log_file(path)
        ident = cell["identity"]
        groups.setdefault((ident["task"], ident["provider"], ident["model"]), []).append(cell)

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = _git_stamp()
    footer = (
        f"\n---\nGenerated by `scripts/make_tables.py` at commit `{stamp}` from "
        f"{len(logs)} committed JSONL log(s) via the independent scorer "
        "(analysis_plan A3.9a). Do not edit numbers by hand.\n"
    )

    combined: list[str] = [f"# Results tables (commit {stamp})", ""]
    for key, cells in sorted(groups.items()):
        md = build_group_markdown(key, cells)
        combined.append(md)
        name = _slug("_".join(key))
        with open(os.path.join(args.out_dir, f"{name}.md"), "w", encoding="utf-8") as f:
            f.write(md + footer)
        if args.latex:
            with open(os.path.join(args.out_dir, f"{name}.tex"), "w", encoding="utf-8") as f:
                f.write(build_group_latex(key, cells) + f"% {stamp}\n")
        print(f"  wrote {name}.md{' + .tex' if args.latex else ''}", file=sys.stderr)

    with open(os.path.join(args.out_dir, "ALL_TABLES.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(combined) + footer)
    print(f"  wrote ALL_TABLES.md ({len(groups)} group(s))", file=sys.stderr)

    # Baseline comparison — emitted only when baseline cells exist, so a runs
    # directory that predates them produces no empty or misleading table.
    comparison = build_baseline_comparison(groups)
    if comparison:
        with open(os.path.join(args.out_dir, "BASELINE_COMPARISON.md"), "w", encoding="utf-8") as f:
            f.write(comparison + footer)
        print("  wrote BASELINE_COMPARISON.md", file=sys.stderr)
        if args.latex:
            with open(
                os.path.join(args.out_dir, "BASELINE_COMPARISON.tex"), "w", encoding="utf-8"
            ) as f:
                f.write(build_baseline_comparison_latex(groups) + f"% {stamp}\n")
            print("  wrote BASELINE_COMPARISON.tex", file=sys.stderr)
    else:
        print(
            "  no baseline comparison table: none of the cells in "
            f"{args.runs_dir} is a baseline arm "
            f"({', '.join(sorted(BASELINE_ARM_NAMES))}). Run the harness with "
            "--only-baselines to produce them.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
