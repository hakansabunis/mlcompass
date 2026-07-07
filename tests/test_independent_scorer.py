"""Tests for the independent re-scorer + table builder (analysis_plan A3.9a, R8).

The scorer is a SECOND implementation of the preregistered channel rules —
these tests check (a) it agrees with the harness's inline scorer on every
channel for hand-built records, (b) transport errors are excluded, (c) the
evidence-pairing plumbing works on the exact files the harness writes, and
(d) the table builder derives its numbers from the logs alone.
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
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


scorer = _load("independent_scorer_ut", "independent_scorer.py")
harness = _load("ablation_harness_for_scorer_ut", "reproduce_hallucination_ablation.py")

EVIDENCE = {
    "row_count": 100,
    "trustworthy_sample_size": True,
    "suspicious_metric": {"name": "r2", "value": 1.0},
    "target_feature_correlations": [
        {"feature": "leak_col", "correlation": 0.99},
        {"feature": "other_col", "correlation": 0.42},
    ],
    "perfect_match_rate": 0.99,
    "candidate_leak_columns": ["leak_col"],
    "notes": [],
}

ALLOWED = {"leak_col", "other_col"}
CORR = {"leak_col": 0.99, "other_col": 0.42}
ANCHOR = "leak_col"


def _rec(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "columns": ["leak_col"],
        "claims": [],
        "verdict": "leakage_likely",
        "omitted": None,
        "rejections": 0,
    }
    base.update(kw)
    return base


# --------------------------------------------------------------------------- #
# Rule-by-rule agreement with the harness's inline scorer                      #
# --------------------------------------------------------------------------- #


def test_independent_scorer_agrees_with_harness_on_every_channel() -> None:
    cases = [
        _rec(),  # clean
        _rec(columns=["leak_col", "ghost"]),  # entity fabrication (cited)
        _rec(
            claims=[{"column": "phantom", "statistic": "correlation", "value": 0.9}]
        ),  # entity via claim
        _rec(claims=[{"column": "leak_col", "statistic": "correlation", "value": 0.5}]),  # misquote
        _rec(claims=[{"column": "leak_col", "statistic": "correlation", "value": 0.99}]),  # sound
        _rec(columns=["other_col"]),  # omission (committed, anchor unaddressed)
        _rec(columns=[], verdict="cannot_determine"),  # abstention, not an omission
        _rec(columns=[], claims=[], verdict="leakage_likely"),  # empty answer: no commitment
        _rec(columns=["other_col"], omitted=True),  # contract flag read as data
        _rec(columns=["leak_col"], omitted=False),  # contract flag, negative
    ]
    for record in cases:
        ours = scorer.score_record(record, ALLOWED, CORR, ANCHOR)
        theirs = harness.score_one(record, ALLOWED, CORR, ANCHOR)
        assert ours["entity"] == theirs["entity"], record
        assert ours["value"] == theirs["value"], record
        assert ours["omission"] == theirs["omission"], record


def test_error_records_are_excluded_not_scored_clean() -> None:
    records = [_rec(), {**_rec(), "error": "TimeoutError: boom"}]
    cell = scorer.score_cell(records, EVIDENCE)
    assert cell["n_total"] == 2
    assert cell["errors_excluded"] == 1
    assert cell["n"] == 1
    assert cell["entity"]["n"] == 1  # the error never enters a denominator


def test_omission_undefined_without_anchor() -> None:
    no_anchor = {**EVIDENCE, "candidate_leak_columns": []}
    cell = scorer.score_cell([_rec(columns=["other_col"])], no_anchor)
    assert cell["omission"] is None  # excluded from omission denominators (A3.3)
    assert cell["anchor"] is None


def test_wilson_matches_harness_formula() -> None:
    for k, n in [(0, 200), (3, 100), (0, 0), (7, 7)]:
        assert scorer.wilson(k, n) == harness.wilson_ci_95(k, n)


def test_cross_check_counts_disagreements() -> None:
    # A record whose STORED flags disagree with a recomputation must surface
    # in the agreement stats — that disagreement is the statistic the paper
    # reports, not something to silently overwrite.
    record = {**_rec(columns=["leak_col", "ghost"]), "scored": {"entity": False, "value": False}}
    cell = scorer.score_cell([record], EVIDENCE)
    assert cell["agreement"]["compared"] == 1
    assert cell["agreement"]["disagreements"]["entity"] == 1
    assert cell["agreement"]["disagreements"]["value"] == 0


# --------------------------------------------------------------------------- #
# File plumbing on the exact artifacts the harness writes                      #
# --------------------------------------------------------------------------- #


def _write_cell(tmp_path: Path) -> tuple[Path, str]:
    ev_hash = harness._evidence_hash(EVIDENCE)
    with open(tmp_path / f"evidence_{ev_hash}.json", "w", encoding="utf-8") as f:
        json.dump({"task_label": "synthetic", "evidence": EVIDENCE}, f)
    log = harness.RunLog(
        str(tmp_path),
        provider="qwen",
        model="qwen-flash",
        task_label="synthetic",
        arm="layer1",
        n=3,
        seed=0,
        resume=False,
        evidence_hash=ev_hash,
    )
    meta = {"provider": "qwen", "task": "synthetic", "seed": 0, "evidence_hash": ev_hash}
    for i, r in enumerate(
        [_rec(), _rec(columns=["leak_col", "ghost"]), {**_rec(), "error": "TimeoutError: x"}]
    ):
        flags = harness.score_one(r, ALLOWED, CORR, ANCHOR)
        log.append({**meta, "i": i, "arm": "layer1", "model": "qwen-flash", **r, "scored": flags})
    return Path(log.path), ev_hash


def test_score_log_file_pairs_evidence_by_hash(tmp_path: Path) -> None:
    log_path, ev_hash = _write_cell(tmp_path)
    assert scorer.find_evidence_for(str(log_path)) == str(tmp_path / f"evidence_{ev_hash}.json")
    cell = scorer.score_log_file(str(log_path))
    assert cell["identity"]["provider"] == "qwen"
    assert cell["identity"]["arm"] == "layer1"
    assert cell["n"] == 2 and cell["errors_excluded"] == 1
    assert cell["entity"]["k"] == 1  # the ghost record, recomputed independently
    assert cell["agreement"]["compared"] == 2
    assert sum(cell["agreement"]["disagreements"].values()) == 0


def test_make_tables_builds_from_logs_alone(tmp_path: Path) -> None:
    _write_cell(tmp_path)
    tables = _load("make_tables_ut", "make_tables.py")
    out_dir = tmp_path / "tables"
    argv = sys.argv
    sys.argv = ["make_tables.py", "--runs-dir", str(tmp_path), "--out-dir", str(out_dir)]
    try:
        assert tables.main() == 0
    finally:
        sys.argv = argv
    combined = (out_dir / "ALL_TABLES.md").read_text(encoding="utf-8")
    assert "synthetic -- qwen / qwen-flash" in combined
    assert "1/2" in combined  # entity k/N from the log, via the independent scorer
    assert "Do not edit numbers by hand" in combined
