"""Build manuscript tables from committed run logs (Q1_ROADMAP.md R8).

Every number in every results table must derive mechanically from a committed
JSONL run log — no hand-typed rates. This script walks a runs directory,
re-scores every cell with the INDEPENDENT scorer (analysis_plan A3.9a; the
harness's inline flags are only cross-checked, never used for the tables),
and emits one Markdown table per (task, provider, model) group plus a
combined file. Each output carries the git commit of the working tree and a
dirty flag, so a table can always be traced back to the exact logs.

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


def build_group_markdown(key: tuple[str, str, str], cells: list[dict[str, Any]]) -> str:
    task, provider, model = key
    lines = [
        f"### {task} -- {provider} / {model}",
        "",
        "| arm | N | err | entity | value | omission | abstained | Tier B catches |",
        "| --- | --: | --: | --- | --- | --- | --- | --: |",
    ]
    for cell in sorted(cells, key=lambda c: c["identity"]["arm"]):
        ident = cell["identity"]
        tb = cell["tier_b"]
        lines.append(
            f"| {ident['arm']} | {cell['n']} | {cell['errors_excluded']} "
            f"| {_fmt(cell['entity'])} | {_fmt(cell['value'])} "
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
