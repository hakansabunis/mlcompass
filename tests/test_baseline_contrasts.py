"""Tests for the Round-2 contrast machinery (analysis_plan_2026-09 §2.3).

Two things are checked here and they are different in kind:

1. **The methods reproduce the plan's own pre-computed constants.** §2.4 was
   written before this code existed and states three numbers outright: a 0/200
   vs 0/200 Newcombe interval of plus or minus 1.88pp, and Fisher p = 0.061 at
   k = 5 and p = 0.030 at k = 6 against a 0/200 arm. Those are independent
   fixed points. Code that misses them is wrong no matter how plausible its
   output looks, and the plan's stated resolution would silently stop being
   the design's actual resolution.

2. **The decision rules apply §2.4 mechanically**, including the losing
   conditions. A verdict a human picks after seeing the interval is not a
   pre-registered verdict.

No API calls, no run logs, no product imports.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


scorer = _load("independent_scorer", "independent_scorer.py")
tables = _load("make_tables", "make_tables.py")


# --------------------------------------------------------------------------- #
# The plan's pre-computed constants                                            #
# --------------------------------------------------------------------------- #


def test_newcombe_reproduces_the_plans_zero_vs_zero_interval() -> None:
    """§2.4: "0/200 vs 0/200 gives a Newcombe interval of plus or minus 1.88pp"
    — the number the equivalence band delta = 2pp was chosen against."""
    low, high = tables.newcombe_diff_ci95(0, 200, 0, 200)
    assert round(low * 100, 2) == -1.88
    assert round(high * 100, 2) == 1.88


def test_fisher_reproduces_the_plans_significance_threshold() -> None:
    """§2.4: "k = 5 gives p = 0.061; k = 6 gives p = 0.030" against a 0/200 arm.

    This is what fixes SIX events as the design's detection threshold, which
    H7 and H10 both state as their rejecting outcome.
    """
    assert round(tables.fisher_exact_two_sided(0, 200, 5, 200), 3) == 0.061
    assert round(tables.fisher_exact_two_sided(0, 200, 6, 200), 3) == 0.030
    # ... and the threshold really is at six, not five.
    assert tables.fisher_exact_two_sided(0, 200, 5, 200) > 0.05
    assert tables.fisher_exact_two_sided(0, 200, 6, 200) < 0.05


def test_fisher_is_symmetric_and_bounded() -> None:
    assert tables.fisher_exact_two_sided(3, 200, 21, 200) == tables.fisher_exact_two_sided(
        21, 200, 3, 200
    )
    assert tables.fisher_exact_two_sided(0, 200, 0, 200) == 1.0
    for k1, k2 in ((0, 0), (1, 0), (7, 3), (200, 0), (13, 29)):
        p = tables.fisher_exact_two_sided(k1, 200, k2, 200)
        assert 0.0 <= p <= 1.0


def test_newcombe_is_not_degenerate_where_wald_would_be() -> None:
    """The reason §2.3 prohibits Wald: at k = 0 in both arms the Wald interval
    has zero width, which would read as perfect equivalence from no evidence."""
    low, high = tables.newcombe_diff_ci95(0, 200, 0, 200)
    assert high - low > 0.03
    # Wider at smaller N, as any honest interval must be.
    small = tables.newcombe_diff_ci95(0, 20, 0, 20)
    assert (small[1] - small[0]) > (high - low)


def test_newcombe_sign_follows_the_contract_minus_baseline_convention() -> None:
    low, high = tables.newcombe_diff_ci95(0, 200, 20, 200)
    assert high < 0, "a contract with fewer violations must give a negative interval"


# --------------------------------------------------------------------------- #
# Holm, over the frozen family only                                            #
# --------------------------------------------------------------------------- #


def test_holm_is_monotone_and_never_shrinks_a_p_value() -> None:
    raw = [0.001, 0.04, 0.02, 0.6]
    adjusted = tables.holm_adjust(raw)
    assert all(a >= r for a, r in zip(adjusted, raw, strict=True))
    ordered = [a for _, a in sorted(zip(raw, adjusted, strict=True))]
    assert ordered == sorted(ordered), "adjusted p must not decrease with raw p"
    assert all(a <= 1.0 for a in adjusted)


def test_holm_family_is_exactly_x1_x3_x4_x5() -> None:
    """§2.2 freezes the family. X2 is uncorrected by design and the
    amendment-A1 exploratory contrast must not be added to it — enlarging a
    frozen family after the fact changes every other member's correction."""
    family = {cid for cid, _b, _c, _o, in_family, _n in tables.CONTRASTS if in_family}
    assert family == {"X1", "X3", "X4", "X5"}
    outside = {cid for cid, _b, _c, _o, in_family, _n in tables.CONTRASTS if not in_family}
    assert outside == {"X2", "X3b"}


def test_every_contrast_is_against_the_contract_arm() -> None:
    assert tables.CONTRACT_ARM == "layer3_bare"
    assert tables.ARM_IDS[tables.CONTRACT_ARM] == "A-CONTRACT"
    baselines = {b for _c, b, _ch, _o, _f, _n in tables.CONTRASTS}
    assert tables.CONTRACT_ARM not in baselines, "the contract cannot be its own baseline"


# --------------------------------------------------------------------------- #
# §2.4 decision rules                                                          #
# --------------------------------------------------------------------------- #


def test_verdicts_apply_the_materiality_and_equivalence_bars() -> None:
    # Significant and material: interval excludes 0 and the point diff >= 5pp.
    assert tables._verdict(-0.09, (-0.14, -0.045)) == "contract lower (material)"
    # Significant but under the 5pp bar: reported as such, never as superiority.
    verdict = tables._verdict(-0.04, (-0.077, -0.013))
    assert "(material)" not in verdict and verdict.startswith("significant")
    # LC-1 equivalence: the whole interval inside plus or minus 2pp.
    assert "LC-1" in tables._verdict(0.0, (-0.0188, 0.0188))
    # Interval spans 0 and is wider than the band: nothing is claimed.
    assert tables._verdict(0.015, (-0.03, 0.06)) == "inconclusive"
    # A baseline that beats the contract says so in the same words.
    assert "BASELINE" in tables._verdict(0.08, (0.03, 0.13))


# --------------------------------------------------------------------------- #
# The §1.3 two-column omission split                                           #
# --------------------------------------------------------------------------- #

EVIDENCE = {
    "candidate_leak_columns": ["leak_col"],
    "target_feature_correlations": [
        {"feature": "leak_col", "correlation": 0.99},
        {"feature": "other_col", "correlation": 0.42},
    ],
}


def _record(**kw: Any) -> dict[str, Any]:
    base = {
        "columns": ["leak_col"],
        "claims": [{"column": "leak_col", "statistic": "correlation", "value": 0.99}],
        "verdict": "leakage_likely",
    }
    base.update(kw)
    return base


def test_flagged_and_undetected_omissions_are_separate_columns() -> None:
    records = [
        # Flagged: skipped the anchor and the verifier said so.
        _record(columns=["other_col"], claims=[], omitted=True),
        # Clean: addressed the anchor, verifier agrees.
        _record(omitted=False),
        # UNDETECTED: committed, skipped the anchor, verifier reported nothing.
        _record(columns=["other_col"], claims=[], omitted=False),
    ]
    cell = scorer.score_cell(records, EVIDENCE)
    assert cell["omission"]["k"] == 1, "the flagged column reads the verifier's flag"
    assert cell["omission_undetected"]["k"] == 1, "the verifier missed one"
    assert cell["omission_undetected"]["n"] == 3


def test_undetected_is_unavailable_on_arms_without_a_verifier() -> None:
    """An arm with no completeness verifier reports n/a, never 0: there is
    nothing for an omission to go undetected BY, and a zero would read as a
    passing grade the arm never sat."""
    records = [_record(columns=["other_col"], claims=[]) for _ in range(3)]
    cell = scorer.score_cell(records, EVIDENCE)
    assert cell["omission"]["k"] == 3, "recomputed from the fields"
    assert cell["omission_undetected"] is None
    assert tables._fmt_undetected(None) == "n/a (no verifier)"


def test_a_nonzero_undetected_cell_is_rendered_as_a_defect_not_a_rate() -> None:
    assert "DEFECT" in tables._fmt_undetected({"k": 2, "n": 200, "rate": 0.01, "ci95": [0, 0.03]})
    assert "DEFECT" not in tables._fmt_undetected(
        {"k": 0, "n": 200, "rate": 0.0, "ci95": [0, 0.01]}
    )


def test_abstention_is_not_an_undetected_omission() -> None:
    """An explicit cannot_determine is not a committed answer (Prop. 2), so it
    can never be an omission the verifier missed."""
    records = [_record(columns=[], claims=[], verdict="cannot_determine", omitted=False)]
    cell = scorer.score_cell(records, EVIDENCE)
    assert cell["omission_undetected"]["k"] == 0
    assert cell["abstained"]["k"] == 1


# --------------------------------------------------------------------------- #
# End to end: a contrast table built from logs, with nothing typed by hand     #
# --------------------------------------------------------------------------- #


def _write_cell(runs: Path, arm: str, n: int, entity_k: int, ev_hash: str = "abcd1234") -> None:
    path = runs / f"prov_model_task_{arm}_n{n}_seed0_e{ev_hash}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            columns = ["leak_col"] + (["phantom"] if i < entity_k else [])
            f.write(
                json.dumps(
                    {
                        "provider": "prov",
                        "task": "task",
                        "model": "model",
                        "arm": arm,
                        "arm_id": tables.ARM_IDS.get(arm, "?"),
                        "evidence_hash": ev_hash,
                        "columns": columns,
                        "claims": [],
                        "verdict": "leakage_likely",
                        "provider_calls": 1,
                    }
                )
                + "\n"
            )


def test_contrast_table_is_built_from_logs_and_names_missing_arms(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "evidence_abcd1234.json").write_text(
        json.dumps({"task_label": "task", "evidence": EVIDENCE}), encoding="utf-8"
    )
    _write_cell(runs, "layer3_bare", 200, 0)
    _write_cell(runs, "guardrails_stock", 200, 24)

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for log in sorted(runs.glob("*.jsonl")):
        cell = tables.scorer.score_log_file(str(log))
        ident = cell["identity"]
        groups.setdefault((ident["task"], ident["provider"], ident["model"]), []).append(cell)

    out = tables.build_contrast_tables(groups)
    assert "X1" in out and "guardrails_stock" in out
    # X1 ran: 0/200 vs 24/200 is well past the six-event threshold.
    assert "| X1 | `guardrails_stock` | composite | family | 0/200 | 24/200 |" in out
    # X5 has no adapter, so it is NOT RUN rather than a tie at zero.
    assert "NOT RUN" in out
    assert "| X5 |" in out
    # Nothing was typed: the contract arm's own id labels the block.
    assert "A-CONTRACT" in out


def test_no_contrast_table_without_the_contract_arm(tmp_path: Path) -> None:
    """Every registered contrast is against A-CONTRACT. Without that cell the
    file is not emitted at all, rather than emitted with an implied baseline."""
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "evidence_abcd1234.json").write_text(
        json.dumps({"task_label": "task", "evidence": EVIDENCE}), encoding="utf-8"
    )
    _write_cell(runs, "guardrails_stock", 20, 4)
    cell = tables.scorer.score_log_file(str(next(runs.glob("*.jsonl"))))
    groups = {("task", "prov", "model"): [cell]}
    assert tables.build_contrast_tables(groups) == ""


# --------------------------------------------------------------------------- #
# The stale-enum arm is degenerate at both extremes                            #
# --------------------------------------------------------------------------- #


def _write_static_cell(runs: Path, coverage: float | None, entity_k: int, n: int = 200) -> None:
    baseline = {
        "mechanism": "provider-strict static schema, author-time enum",
        "arm_id": "A-STATIC-ENUM-STALE",
        "enum_declared": True,
        "strict_applied": True,
        "coverage": (None if coverage is None else {"coverage": coverage, "overlap": 1}),
    }
    path = runs / "prov_model_task_static_schema_n200_seed0_eabcd1234.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "provider": "prov",
                        "task": "task",
                        "model": "model",
                        "arm": "static_schema",
                        "evidence_hash": "abcd1234",
                        "columns": ["leak_col"] + (["phantom"] if i < entity_k else []),
                        "claims": [],
                        "verdict": "leakage_likely",
                        "provider_calls": 1,
                        "baseline": baseline,
                    }
                )
                + "\n"
            )


def _groups_from(runs: Path) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for log in sorted(runs.glob("*.jsonl")):
        cell = tables.scorer.score_log_file(str(log))
        ident = cell["identity"]
        groups.setdefault((ident["task"], ident["provider"], ident["model"]), []).append(cell)
    return groups


def _runs_with_contract(tmp_path: Path) -> Path:
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "evidence_abcd1234.json").write_text(
        json.dumps({"task_label": "task", "evidence": EVIDENCE}), encoding="utf-8"
    )
    _write_cell(runs, "layer3_bare", 200, 0)
    return runs


def test_scorer_surfaces_the_static_domain_coverage_as_a_number(tmp_path: Path) -> None:
    runs = _runs_with_contract(tmp_path)
    _write_static_cell(runs, coverage=0.4, entity_k=8)
    cells = _groups_from(runs)[("task", "prov", "model")]
    stale = next(c for c in cells if c["identity"]["arm"] == "static_schema")
    assert stale["static_domain_coverage"] == 0.4
    contract = next(c for c in cells if c["identity"]["arm"] == "layer3_bare")
    assert contract["static_domain_coverage"] is None, "arms with no domain report none"


def test_a_fully_covering_stale_domain_gets_no_verdict(tmp_path: Path) -> None:
    """100% coverage makes the stale-enum arm a copy of the live-enum arm, so
    the contrast compares an arm against itself. No verdict is printed."""
    runs = _runs_with_contract(tmp_path)
    _write_static_cell(runs, coverage=1.0, entity_k=0)
    out = tables.build_contrast_tables(_groups_from(runs))
    x3b = [line for line in out.splitlines() if line.startswith("| X3b |")]
    assert x3b, "X3b should still appear"
    assert all("DEGENERATE CELL" in line for line in x3b)
    assert "not interpretable in this cell" in out


def test_a_disjoint_stale_domain_gets_no_verdict_either(tmp_path: Path) -> None:
    """0% coverage manufactures a fabrication rate out of the mismatch. That is
    rigged in our favour, which is the direction that needs the harder guard."""
    runs = _runs_with_contract(tmp_path)
    _write_static_cell(runs, coverage=0.0, entity_k=180)
    out = tables.build_contrast_tables(_groups_from(runs))
    x3b = [line for line in out.splitlines() if line.startswith("| X3b |")]
    assert all("DEGENERATE CELL" in line for line in x3b)
    assert "contract lower (material)" not in "\n".join(x3b)


def test_a_partially_stale_domain_is_interpretable(tmp_path: Path) -> None:
    """The condition a schema actually goes stale under: some overlap, not all."""
    runs = _runs_with_contract(tmp_path)
    _write_static_cell(runs, coverage=0.4, entity_k=24)
    out = tables.build_contrast_tables(_groups_from(runs))
    x3b = [line for line in out.splitlines() if line.startswith("| X3b |")]
    assert x3b and not any("DEGENERATE" in line for line in x3b)
    assert "not interpretable in this cell" not in out


def test_the_degeneracy_guard_does_not_touch_the_registered_contrasts(tmp_path: Path) -> None:
    """X1/X3/X4 do not depend on the frozen domain and must stay unaffected."""
    runs = _runs_with_contract(tmp_path)
    _write_static_cell(runs, coverage=1.0, entity_k=0)
    _write_cell(runs, "guardrails_stock", 200, 24)
    out = tables.build_contrast_tables(_groups_from(runs))
    x1 = [line for line in out.splitlines() if line.startswith("| X1 |")]
    assert x1 and not any("DEGENERATE" in line for line in x1)
