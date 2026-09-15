"""The entity channel, split into what was invented and what was misfiled.

Why this exists. The manuscript's headline failure mode is *phantom-entity
fabrication*, defined as "a column the data does not contain". The baseline
battery of 2026-09-15 scored 1,400 live responses on that channel and found
that **every** flagged entity was ``r2`` — the name of the suspicious metric,
which the evidence dictionary carries as ``suspicious_metric.name`` and which
the model wrote into ``columns_referenced``. Nothing was invented, on any arm.

Both are contract violations: a consumer that drops or correlates ``r2`` is
acting on something that is not a column. But only one of them is a
hallucination, and a blended rate lets a reviewer say the number does not
measure what the paper calls it. So the scorer now reports both, and these
tests pin the distinction.

`entity` keeps its old meaning — the disjunction — so no existing comparison
moves.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "ablation_harness_split", ROOT / "scripts" / "reproduce_hallucination_ablation.py"
)
assert _spec is not None and _spec.loader is not None
harness = importlib.util.module_from_spec(_spec)
sys.modules["ablation_harness_split"] = harness
_spec.loader.exec_module(harness)

# A trimmed copy of the shape `detect_leakage` really emits, kept faithful in
# the one respect that matters: the metric's name lives in the dict, and it is
# not a column.
EVIDENCE: dict[str, Any] = {
    "suspicious_metric": {"name": "r2", "value": 1.0},
    "target_feature_correlations": [
        {"feature": "log_target_v2", "correlation": 0.9990, "method": "spearman"},
        {"feature": "feature_5", "correlation": 0.31, "method": "pearson"},
    ],
    "candidate_leak_columns": ["log_target_v2"],
    "perfect_match_rate": 0.0,
}

ALLOWED = {"log_target_v2", "feature_5"}
CORR = {"log_target_v2": 0.9990, "feature_5": 0.31}
ANCHOR = "log_target_v2"
NAMES = harness.evidence_name_set(EVIDENCE)


def _score(**kw: Any) -> dict[str, bool]:
    response = {"columns": [], "claims": [], "verdict": "leakage_likely", "omitted": None}
    response.update(kw)
    return harness.score_one(response, ALLOWED, CORR, ANCHOR, NAMES)


def test_evidence_name_set_collects_keys_and_string_values() -> None:
    # The metric name is a value; the field it sits under is a key. Both count,
    # because either is somewhere the narrator could have read "r2".
    assert "r2" in NAMES
    assert "suspicious_metric" in NAMES
    assert "perfect_match_rate" in NAMES
    assert "log_target_v2" in NAMES
    assert "ghost_column" not in NAMES


def test_the_metric_name_in_a_column_field_is_misfiled_not_invented() -> None:
    """The exact case that produced 84/200 on the bare arm."""
    flags = _score(columns=["log_target_v2", "r2"])
    assert flags["entity"] is True, "still a contract violation: r2 is not a column"
    assert flags["entity_misfiled"] is True
    assert flags["entity_invented"] is False


def test_a_name_absent_from_the_evidence_is_invented() -> None:
    flags = _score(columns=["log_target_v2", "shadow_price"])
    assert flags["entity"] is True
    assert flags["entity_invented"] is True
    assert flags["entity_misfiled"] is False


def test_both_kinds_in_one_response_are_reported_separately() -> None:
    flags = _score(columns=["log_target_v2", "r2", "shadow_price"])
    assert flags["entity_invented"] is True
    assert flags["entity_misfiled"] is True


def test_a_clean_response_flags_neither() -> None:
    flags = _score(
        columns=["log_target_v2"],
        claims=[{"column": "feature_5", "statistic": "correlation", "value": 0.31}],
    )
    assert flags["entity"] is False
    assert flags["entity_invented"] is False
    assert flags["entity_misfiled"] is False


def test_a_claim_column_is_scored_like_a_cited_column() -> None:
    """The split must read `claims[].column`, not only `columns_referenced`.

    Scoring only one of the two would report a clean entity channel for a
    response whose claims cite something out of domain, which is the failure
    the value channel does not cover.
    """
    invented = _score(
        columns=["log_target_v2"],
        claims=[{"column": "shadow_price", "statistic": "correlation", "value": 0.5}],
    )
    assert invented["entity_invented"] is True
    misfiled = _score(
        columns=["log_target_v2"],
        claims=[{"column": "r2", "statistic": "correlation", "value": 1.0}],
    )
    assert misfiled["entity_misfiled"] is True
    assert misfiled["entity_invented"] is False


def test_entity_is_still_the_disjunction() -> None:
    """`entity` must not move, or every published comparison moves with it."""
    for kw in (
        {"columns": ["log_target_v2", "r2"]},
        {"columns": ["log_target_v2", "shadow_price"]},
        {"columns": ["log_target_v2", "r2", "shadow_price"]},
    ):
        flags = _score(**kw)
        assert flags["entity"] == (flags["entity_invented"] or flags["entity_misfiled"])


def test_the_split_is_opt_in_so_existing_callers_are_unchanged() -> None:
    """Without an evidence-name set the scorer returns exactly the old keys.

    The harness has callers and preserved-log readers that predate the split;
    a scorer that suddenly grew keys for them would change what those tools
    write without anyone asking for it.
    """
    response = {"columns": ["r2"], "claims": [], "verdict": "leakage_likely", "omitted": None}
    flags = harness.score_one(response, ALLOWED, CORR, ANCHOR)
    assert set(flags) == {"entity", "value", "omission"}
    assert flags["entity"] is True
