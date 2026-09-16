"""The value check must read the `statistic` field, not only the column.

Two reviewers disagreed about this and the disagreement is the reason the file
exists. One recomputed the paraphrase sweep, found the value channel firing at
up to 27/100, and said the paper was wrong to report it as silent. The other
opened the flagged records and found that 69 of ~85 of them look like

    {column: log_target_v2, statistic: "perfect_match_rate", value: 0.0}

scored against that column's *correlation* of 0.9987 -- while 0.0 is the
correct perfect-match rate. The scorer was calling the narrator wrong for
being right.

The second reviewer was correct, and the distinction is not cosmetic. It
separates three different things the old scorer blended into one:

  * the narrator misquoted a quantity the evidence carries -- a real value
    fabrication, and the thing the paper's value channel is supposed to count;
  * the narrator named a quantity the evidence does not carry for that column
    -- not a misquote, because there is nothing to compare against, and
    reported on its own channel;
  * the scorer compared against the wrong number.

Note what is NOT affected: the production contract. Tier A pins the schema's
`statistic` field to the enum ["correlation"], so a claim reaching Tier B
cannot be about anything else and comparing against the correlation is right
there. The defect was confined to scoring the *unconstrained* arms, where the
narrator chooses the statistic itself -- which is also why it went unnoticed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "ablation_value", ROOT / "scripts" / "reproduce_hallucination_ablation.py"
)
assert _spec is not None and _spec.loader is not None
harness = importlib.util.module_from_spec(_spec)
sys.modules["ablation_value"] = harness
_spec.loader.exec_module(harness)

EVIDENCE: dict[str, Any] = {
    "row_count": 1000,
    "suspicious_metric": {"name": "r2", "value": 1.0},
    "target_feature_correlations": [
        {"feature": "log_target_v2", "correlation": 0.9990, "method": "spearman"},
        {"feature": "feature_12", "correlation": -0.0405, "method": "pearson"},
    ],
    "perfect_match_rate": 0.0,
    "candidate_leak_columns": ["log_target_v2"],
}

ALLOWED = {"log_target_v2", "feature_12"}
CORR = {"log_target_v2": 0.9990, "feature_12": -0.0405}
ANCHOR = "log_target_v2"
TABLE = harness.evidence_value_table(EVIDENCE)


def _score(claims: list[dict[str, Any]], **kw: Any) -> dict[str, bool]:
    response = {
        "columns": ["log_target_v2"],
        "claims": claims,
        "verdict": "leakage_likely",
        "omitted": None,
    }
    return harness.score_one(response, ALLOWED, CORR, ANCHOR, None, TABLE, **kw)


def test_the_case_that_produced_69_false_positives() -> None:
    """Verbatim from the sweep records. 0.0 is the correct perfect-match rate."""
    flags = _score([{"column": "log_target_v2", "statistic": "perfect_match_rate", "value": 0.0}])
    assert flags["value"] is False, "a correct global quantity is not a value fabrication"
    assert flags["value_unverifiable"] is False, "the evidence does carry this quantity"


def test_a_genuinely_misquoted_correlation_still_fires() -> None:
    """Widening the check must not make it unable to fire."""
    flags = _score([{"column": "log_target_v2", "statistic": "correlation", "value": 0.42}])
    assert flags["value"] is True


def test_a_misquoted_global_quantity_fires() -> None:
    flags = _score([{"column": "log_target_v2", "statistic": "perfect_match_rate", "value": 0.87}])
    assert flags["value"] is True


def test_a_statistic_the_evidence_does_not_carry_is_unverifiable_not_wrong() -> None:
    """The channel the fix uncovered.

    The narrator names a quantity that is simply not in the evidence for this
    column. There is nothing to compare against, so calling it a misquote would
    be a category error -- the same error the entity channel made before it was
    split into invented and misfiled.
    """
    flags = _score(
        [
            {
                "column": "log_target_v2",
                "statistic": "residual_r2_after_excluding_leaks",
                "value": 0.31,
            }
        ]
    )
    assert flags["value"] is False
    assert flags["value_unverifiable"] is True


@pytest.mark.parametrize(
    "alias",
    ["correlation", "abs_corr", "Pearson Correlation With Target", "spearman-correlation"],
)
def test_correlation_aliases_resolve_to_the_measured_correlation(alias: str) -> None:
    """The narrator picks the wording; the evidence stores one number.

    Case and separators are normalised, so `Abs Corr` meets `abs_corr`. An
    alias the scorer failed to recognise would be reported as unverifiable and
    quietly deflate the value channel.
    """
    assert (
        _score([{"column": "log_target_v2", "statistic": alias, "value": 0.999}])["value"] is False
    )
    assert _score([{"column": "log_target_v2", "statistic": alias, "value": 0.10}])["value"] is True


def test_the_sign_of_a_correlation_is_part_of_the_value() -> None:
    """Six of the ten evidence features are negatively correlated.

    The evidence stores the correlation with the larger absolute value with its
    sign preserved. A narrator quoting the magnitude alone is quoting a
    different number, and the contract's 0.005 tolerance is far tighter than
    the gap.
    """
    assert (
        _score([{"column": "feature_12", "statistic": "correlation", "value": -0.0405}])["value"]
        is False
    )
    assert (
        _score([{"column": "feature_12", "statistic": "correlation", "value": 0.0405}])["value"]
        is True
    )


def test_an_out_of_evidence_column_belongs_to_the_entity_channel() -> None:
    """One failure, one channel. A phantom column is not also a value fault."""
    flags = _score([{"column": "ghost", "statistic": "correlation", "value": 0.5}])
    assert flags["entity"] is True
    assert flags["value"] is False
    assert flags["value_unverifiable"] is False


def test_the_legacy_path_is_byte_identical_for_contract_arms() -> None:
    """Callers that pass no value table must score exactly as before.

    Contract arms are scored on the legacy path and their published numbers
    must not move: Tier A pins `statistic` to "correlation" there, so keying on
    the column alone was always correct for them.
    """
    response = {
        "columns": ["log_target_v2"],
        "claims": [{"column": "log_target_v2", "statistic": "correlation", "value": 0.42}],
        "verdict": "leakage_likely",
        "omitted": None,
    }
    legacy = harness.score_one(response, ALLOWED, CORR, ANCHOR)
    assert set(legacy) == {"entity", "value", "omission"}
    assert legacy["value"] is True


def test_the_production_schema_pins_the_statistic() -> None:
    """The reason the defect never reached the shipped contract.

    If this enum ever widens, the legacy scoring path above stops being correct
    for contract arms and this test is where that shows up.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from mlcompass.agents.leakage_investigator import build_submit_investigation_tool

    tool = build_submit_investigation_tool(["log_target_v2", "feature_12"])
    statistic = repr(tool)
    assert "'statistic'" in statistic or '"statistic"' in statistic
    # The enum is what confines a claim to one quantity, and therefore what
    # makes column-keyed comparison correct on the contract path.
    assert "correlation" in statistic
    assert statistic.count("correlation") >= 1
