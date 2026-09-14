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

Two further §2 requirements are met here rather than by hand:

- **BASELINE_CONTRASTS.md** — the §2.2 contrasts X1-X5 (plus the amendment-A1
  exploratory X3b), each as a difference against the contract arm with a
  **Newcombe hybrid-score 95% interval** and a **two-sided Fisher exact** test,
  Holm-adjusted within the frozen family {X1, X3, X4, X5} (§2.3). No difference,
  interval or p-value is ever typed by hand, and a contrast whose arm is absent
  from the run directory prints as NOT RUN rather than as a tie.
- **The two-column omission split** (§1.3): every table reports flagged and
  undetected omissions separately, with `n/a` on arms that have no completeness
  verifier to be undetected by.

Usage:
    python scripts/make_tables.py --runs-dir scripts/runs
    python scripts/make_tables.py --runs-dir scripts/runs --out-dir paper/tables --latex
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import math
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


def _fmt_undetected(ch: dict[str, Any] | None) -> str:
    """The §1.3 second omission column.

    `n/a` means the arm has no completeness verifier, so there is nothing for
    an omission to go undetected BY — reported as unavailable rather than as a
    zero the arm did not earn. A non-zero is a defect marker, not a rate, and
    is rendered so nobody reads it as one.
    """
    if ch is None:
        return "n/a (no verifier)"
    if ch["k"]:
        return f"**DEFECT {ch['k']}/{ch['n']}**"
    return f"{ch['k']}/{ch['n']}"


# --------------------------------------------------------------------------- #
# Interval and test methods (analysis_plan_2026-09 §2.3)                       #
# --------------------------------------------------------------------------- #


def newcombe_diff_ci95(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Newcombe hybrid-score 95% interval for p1 - p2 (Newcombe 1998, method 10).

    §2.3 prohibits Wald intervals for arm differences: zero cells are the
    normal case in this campaign and Wald is degenerate on them (zero width at
    k = 0). The hybrid-score method composes each arm's Wilson interval, so a
    0/200 vs 0/200 comparison still returns the ±1.88pp the plan's §2.4
    equivalence band is calibrated against.

    Reference
    ---------
    Newcombe, R. G. (1998). Interval estimation for the difference between
    independent proportions: comparison of eleven methods. Statistics in
    Medicine, 17(8), 873-890.
    """
    if n1 <= 0 or n2 <= 0:
        return (-1.0, 1.0)
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = scorer.wilson(k1, n1)
    l2, u2 = scorer.wilson(k2, n2)
    diff = p1 - p2
    lower = diff - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    upper = diff + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (max(-1.0, lower), min(1.0, upper))


def fisher_exact_two_sided(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided Fisher exact p for the 2x2 table, alpha = 0.05 (§2.3).

    Computed by exact integer arithmetic over the hypergeometric weights — the
    p is the total probability of every table at most as likely as the observed
    one. Integer weights are compared directly, so there is no floating-point
    tie tolerance to tune and no dependency on SciPy (this script deliberately
    runs on the standard library, so a reviewer can rebuild the tables without
    installing the project).
    """
    if n1 <= 0 or n2 <= 0:
        return 1.0
    successes = k1 + k2
    total = math.comb(n1 + n2, successes)
    if total == 0:
        return 1.0
    observed = math.comb(n1, k1) * math.comb(n2, successes - k1)
    numerator = 0
    for x in range(max(0, successes - n2), min(n1, successes) + 1):
        weight = math.comb(n1, x) * math.comb(n2, successes - x)
        if weight <= observed:
            numerator += weight
    return min(1.0, numerator / total)


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, returned in the input order.

    §2.2 fixes the family as {X1, X3, X4, X5} per provider; X2 is a single
    pre-specified test and is NOT passed through here.
    """
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * pvalues[index])
        running = max(running, candidate)  # enforce monotonicity
        adjusted[index] = running
    return adjusted


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
    "static_schema_noenum",
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
    {"guardrails_stock", "guardrails_tierb", "static_schema", "static_schema_noenum"}
)

# The registered arm id for every harness arm name (plan 2026-09 §2.1, §9 A1).
# Kept in lockstep with ARM_IDS in the harness; a table names the hypothesis a
# cell tests, never just the spelling of its arm.
ARM_IDS: dict[str, str] = {
    "layer1": "A-L1",
    "layer2": "A-L2",
    "layer3": "A-L3-SHIPPED",
    "layer3_bare": "A-CONTRACT",
    "layer3_stress": "A-STRESS",
    "layer3_stress_generic": "A-STRESS-GENERIC",
    "tier_a": "A-STRICT-ENUM",
    "tier_a+strict": "A-STRICT-ENUM",
    "stress_mech": "A-L3-WORST-PARAPHRASE",
    "guardrails_stock": "A-GR-STOCK",
    "guardrails_tierb": "A-GR-OURS",
    "static_schema_noenum": "A-STRICT-STATIC",
    "static_schema": "A-STATIC-ENUM-STALE",
}

# The contract comparator for every registered contrast (§2.1: A-CONTRACT is
# the contract arm of this comparison, NOT the shipped L3, whose strict prompt
# would confound prompt with mechanism).
CONTRACT_ARM = "layer3_bare"

# §2.2 contrast precedence, frozen. Each entry is
# (id, baseline arm, registered outcome channel, other channels, family, note).
#
# `family=True` enters the Holm correction over {X1, X3, X4, X5}. X2 is a
# single pre-specified test and is not corrected. X3b is the amendment-A1
# exploratory contrast: it is deliberately OUTSIDE the family, because adding
# a test to a frozen family after the fact would alter the pre-registered
# multiplicity correction for every other contrast in it.
CONTRASTS: tuple[tuple[str, str, str, tuple[str, ...], bool, str], ...] = (
    (
        "X1",
        "guardrails_stock",
        "composite",
        ("entity", "value", "omission"),
        True,
        "primary confirmatory -- deployed-default question (H7)",
    ),
    (
        "X2",
        "tier_a+strict",
        "value",
        ("omission", "entity", "composite"),
        False,
        "primary differentiating (H10) -- VOID unless plan section 4 activates a "
        "non-entity channel; uncorrected by design",
    ),
    (
        "X3",
        "static_schema_noenum",
        "entity",
        ("composite", "value", "omission"),
        True,
        "secondary -- static strict schema constrains shape, not content (H8)",
    ),
    (
        "X4",
        "guardrails_tierb",
        "composite",
        ("entity", "value", "omission"),
        True,
        "secondary -- loop parity, our checks in their loop (H11)",
    ),
    (
        "X5",
        "nemo",
        "composite",
        ("entity", "value", "omission"),
        True,
        "secondary -- NeMo Guardrails; no adapter exists, so this contrast did "
        "not run and is reported as not run, never as a tie",
    ),
    (
        "X3b",
        "static_schema",
        "entity",
        ("composite", "value", "omission"),
        False,
        "EXPLORATORY (amendment A1, not in the Holm family) -- a STALE "
        "author-time enum against the live call-time one",
    ),
)

# §2.4 decision thresholds, fixed before the run.
DELTA_PP = 0.02  # equivalence band
MATERIAL_PP = 0.05  # superiority also requires a point difference this large


def _verdict(diff: float, ci: tuple[float, float]) -> str:
    """Apply §2.4 mechanically. Rates are violations, so a NEGATIVE difference
    (contract minus baseline) favours the contract."""
    low, high = ci
    excludes_zero = low > 0.0 or high < 0.0
    if low >= -DELTA_PP and high <= DELTA_PP:
        return "equivalent within delta=2pp (LC-1)"
    if not excludes_zero:
        return "inconclusive"
    if diff <= -MATERIAL_PP:
        return "contract lower (material)"
    if diff >= MATERIAL_PP:
        return "BASELINE lower (material)"
    return "significant, below the 5pp materiality bar"


def build_contrast_tables(groups: dict[tuple[str, str, str], list[dict[str, Any]]]) -> str:
    """The §2.2/§2.3 contrast block: Newcombe intervals and Fisher tests.

    Every cell of this table is recomputed here from each arm's own k/N as the
    independent scorer re-derived them. Contrasts whose arms are missing from
    the run directory are printed as NOT RUN with the missing arm named — an
    absent comparison is never rendered as a tie.
    """
    blocks: list[str] = [
        "# Baseline contrasts -- differences between enforcement mechanisms",
        "",
        "Differences are `contract - baseline` on a VIOLATION rate, so a negative",
        "difference favours the contract. Intervals are Newcombe hybrid-score 95%",
        "(analysis_plan_2026-09 section 2.3; Wald is prohibited here because zero",
        "cells are the normal case and Wald is degenerate on them). Tests are",
        "two-sided Fisher exact, alpha = 0.05, computed by exact integer",
        "arithmetic over the hypergeometric weights.",
        "",
        "`p (Holm)` is the step-down adjusted p over the frozen family",
        "{X1, X3, X4, X5} within one provider (section 2.2). X2 is a single",
        "pre-specified test and is not corrected. X3b is the amendment-A1",
        "exploratory contrast and is deliberately outside the family.",
        "",
        "Each contrast has ONE registered outcome channel, marked `family` below;",
        "that row is what enters Holm. Its other channels are reported for",
        "completeness and marked `descriptive` -- they are not additional tests.",
        "",
        "Verdicts apply section 2.4 mechanically: superiority needs the interval",
        "to exclude 0 AND a point difference of at least 5pp; an interval lying",
        "entirely inside plus or minus 2pp is the LC-1 equivalence reading.",
        "",
    ]

    emitted = 0
    for key, cells in sorted(groups.items()):
        by_arm = {c["identity"]["arm"]: c for c in cells}
        if CONTRACT_ARM not in by_arm:
            continue
        emitted += 1
        task, provider, model = key
        contract = by_arm[CONTRACT_ARM]
        blocks += [
            f"### {task} -- {provider} / {model}",
            "",
            f"Contract arm: `{CONTRACT_ARM}` ({ARM_IDS[CONTRACT_ARM]}), "
            f"evidence {contract['evidence_file']}",
            "",
            "| contrast | baseline | channel | role | contract k/N | baseline k/N "
            "| diff (pp) | Newcombe 95% (pp) | Fisher p | p (Holm) | verdict |",
            "| --- | --- | --- | --- | --- | --- | --: | --- | --: | --: | --- |",
        ]

        rows: list[dict[str, Any]] = []
        for cid, baseline_arm, channel, others, in_family, note in CONTRASTS:
            baseline = by_arm.get(baseline_arm)
            if baseline is None:
                # Kept in the row list rather than printed here, so a contrast
                # that did not run still appears in its registered position
                # instead of floating to the top of the table.
                rows.append(
                    {
                        "cid": cid,
                        "baseline_arm": baseline_arm,
                        "channel": channel,
                        "not_run": True,
                        "registered": True,
                        "in_family": False,
                        "note": note,
                    }
                )
                continue
            for index, ch_name in enumerate((channel, *others)):
                c_ch, b_ch = contract.get(ch_name), baseline.get(ch_name)
                if not c_ch or not b_ch or not c_ch["n"] or not b_ch["n"]:
                    continue
                rows.append(
                    {
                        "cid": cid,
                        "baseline_arm": baseline_arm,
                        "channel": ch_name,
                        "registered": index == 0,
                        "in_family": in_family and index == 0,
                        "c": c_ch,
                        "b": b_ch,
                        "note": note,
                    }
                )

        for row in rows:
            if row.get("not_run"):
                continue
            c_ch, b_ch = row["c"], row["b"]
            row["diff"] = c_ch["rate"] - b_ch["rate"]
            row["ci"] = newcombe_diff_ci95(c_ch["k"], c_ch["n"], b_ch["k"], b_ch["n"])
            row["p"] = fisher_exact_two_sided(c_ch["k"], c_ch["n"], b_ch["k"], b_ch["n"])

        family = [r for r in rows if r["in_family"]]
        if family:
            for r, adj in zip(family, holm_adjust([r["p"] for r in family]), strict=True):
                r["p_holm"] = adj

        # A stale-enum cell says nothing at either extreme: a domain covering
        # 100% of the evidence makes the arm a copy of the live-enum arm, and
        # one covering 0% manufactures a fabrication rate out of the mismatch.
        # Both are rigged in opposite directions, so the verdict is withheld
        # rather than printed and then caveated in prose nobody reads.
        stale = by_arm.get("static_schema")
        stale_coverage = stale.get("static_domain_coverage") if stale else None
        degenerate_stale = stale_coverage is not None and (
            stale_coverage <= 0.0 or stale_coverage >= 1.0
        )

        for row in rows:
            if row.get("not_run"):
                blocks.append(
                    f"| {row['cid']} | `{row['baseline_arm']}` | {row['channel']} | -- "
                    f"| -- | NOT RUN (no cell for this arm) | -- | -- | -- | -- | not run |"
                )
                continue
            c_ch, b_ch = row["c"], row["b"]
            low, high = row["ci"]
            holm_text = f"{row['p_holm']:.4f}" if "p_holm" in row else "--"
            role = (
                "family" if row["in_family"] else ("registered" if row["registered"] else "desc.")
            )
            verdict = (
                f"DEGENERATE CELL (domain covers {stale_coverage * 100:.0f}%) -- not interpretable"
                if row["cid"] == "X3b" and degenerate_stale
                else _verdict(row["diff"], row["ci"])
            )
            blocks.append(
                f"| {row['cid']} | `{row['baseline_arm']}` | {row['channel']} | {role} "
                f"| {c_ch['k']}/{c_ch['n']} | {b_ch['k']}/{b_ch['n']} "
                f"| {row['diff'] * 100:+.2f} | [{low * 100:+.2f}, {high * 100:+.2f}] "
                f"| {row['p']:.4f} | {holm_text} | {verdict} |"
            )

        blocks.append("")
        if degenerate_stale:
            blocks += [
                f"**X3b is not interpretable in this cell.** The frozen author-time "
                f"domain covers {stale_coverage * 100:.0f}% of this task's evidence "
                "columns. At 100% the stale-enum arm is a copy of the live-enum arm "
                "by construction; at 0% its rate is a property of the mismatch, not "
                "of the mechanism. X3b needs a task with PARTIAL overlap -- the "
                "condition a schema actually goes stale under -- pinned in section 9 "
                "before the run.",
                "",
            ]
        for cid, _arm, _ch, _others, _fam, note in CONTRASTS:
            if any(r["cid"] == cid for r in rows):
                blocks.append(f"- **{cid}**: {note}")
        blocks.append("")

        # LC-2, computed rather than asserted: if nothing in this cell exceeds
        # 2% on any channel, the block cannot separate mechanisms at this N.
        peak = 0.0
        for cell in cells:
            if cell["identity"]["arm"].removesuffix("+strict") not in BASELINE_ARM_NAMES:
                continue
            for ch_name in ("entity", "value", "omission", "composite"):
                ch = cell.get(ch_name)
                if ch and ch["n"]:
                    peak = max(peak, ch["rate"])
        if peak <= 0.02:
            blocks += [
                f"**LC-2 (whole-comparison underpowering) HOLDS here.** No baseline "
                f"arm in this cell exceeds 2% on any channel (peak {peak * 100:.2f}%). "
                "Per section 2.6 this block is reported as an inconclusive "
                "comparison, not as superiority, and N is not raised afterwards "
                "to rescue it without a section 9 amendment logged first.",
                "",
            ]

    if not emitted:
        return ""
    return "\n".join(blocks)


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
        "construction) or EMPIRICAL (sections 1.2, 1.3). Omission is reported in",
        "TWO columns as section 1.3 requires: `flagged` is what the arm's own",
        "completeness verifier caught, `undetected` is a committed answer that",
        "skipped the anchor while the verifier reported nothing. `undetected` is",
        "`n/a` on arms with no completeness verifier -- there is nothing for an",
        "omission to go undetected by -- and any non-zero there is a verifier",
        "defect under section 3.5, not a rate. `no answer` is the share of",
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
            "| arm | arm id | mechanism | label | N | err | composite | entity | value "
            "| omission (flagged) | omission (undetected) | abstained | no answer "
            "| calls/rsp | catches |",
            "| --- | --- | --- | --- | --: | --: | --- | --- | --- | --- | --- | --- "
            "| --- | --: | --: |",
        ]
        for cell in sorted(cells, key=lambda c: _arm_rank(c["identity"]["arm"])):
            arm = cell["identity"]["arm"]
            blocks.append(
                f"| {arm} | {ARM_IDS.get(arm.removesuffix('+strict'), '-')} "
                f"| {cell.get('mechanism') or '-'} "
                f"| {_label(cell)} | {cell['n']} | {cell['errors_excluded']} "
                f"| {_fmt(cell['composite'])} | {_fmt(cell['entity'])} "
                f"| {_fmt(cell['value'])} | {_fmt(cell['omission'])} "
                f"| {_fmt_undetected(cell.get('omission_undetected'))} "
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
        "| arm | arm id | label | N | err | composite | entity | value "
        "| omission (flagged) | omission (undetected) | abstained | Tier B catches |",
        "| --- | --- | --- | --: | --: | --- | --- | --- | --- | --- | --- | --: |",
    ]
    for cell in sorted(cells, key=lambda c: _arm_rank(c["identity"]["arm"])):
        ident = cell["identity"]
        tb = cell["tier_b"]
        lines.append(
            f"| {ident['arm']} | {ARM_IDS.get(ident['arm'].removesuffix('+strict'), '-')} "
            f"| {_label(cell)} | {cell['n']} | {cell['errors_excluded']} "
            f"| {_fmt(cell['composite'])} | {_fmt(cell['entity'])} | {_fmt(cell['value'])} "
            f"| {_fmt(cell['omission'])} | {_fmt_undetected(cell.get('omission_undetected'))} "
            f"| {_fmt(cell['abstained'])} | {tb['total_catches']} |"
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
        contrasts = build_contrast_tables(groups)
        if contrasts:
            path = os.path.join(args.out_dir, "BASELINE_CONTRASTS.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(contrasts + footer)
            print("  wrote BASELINE_CONTRASTS.md", file=sys.stderr)
        else:
            print(
                f"  no contrast table: no `{CONTRACT_ARM}` cell in {args.runs_dir}. "
                "Every registered contrast X1-X5 is against the contract arm "
                "(A-CONTRACT), so without it there is nothing to contrast.",
                file=sys.stderr,
            )
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
