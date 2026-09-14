"""The measured width of the Tier B value channel.

The contract advertises value soundness over a ``{column, statistic, value}``
triple. These tests establish what that triple actually is and what fraction
of the evidence's quantities it can carry. They are deliberately
characterisation tests: they change no threshold, no schema and no verdict,
because the investigator is the artifact under measurement and published cells
were produced by it.

What the schema permits (``_submit_input_schema``):

* ``column``    — enum bound to the evidence columns at call time (Tier A).
* ``statistic`` — ``{"type": "string", "enum": ["correlation"]}``: a
  one-element enum. No other statistic is expressible.
* ``value``     — an unconstrained number.

What the verifier checks (``investigate_leakage_bound``):

* ``column`` must be a key of ``evidence_correlation_map``.
* ``value`` must be within ``VALUE_TOLERANCE`` of that column's correlation.
* ``statistic`` is **never read**.

So the advertised triple is verified as a pair, and the channel covers exactly
one of the four quantity families the evidence carries. The consequence is
measured below: a fabricated ``perfect_match_rate`` or ``suspicious_metric``
value cannot be expressed as a claim, therefore cannot be verified, therefore
can only ever appear in the unverified ``narration`` string.
"""

from __future__ import annotations

from typing import Any

import pytest
from agentlite.testing import MockClient, tool_use_response

from mlcompass.agents.leakage_investigator import (
    SUBMIT_TOOL_NAME,
    _submit_input_schema,
    evidence_allowed_columns,
    evidence_correlation_map,
    investigate_leakage_bound,
)

# Evidence carrying all four quantity families §III-C Layer 1 names:
# the suspicious metric, the per-feature correlations, the ranked candidate
# set, and P(y_pred == y_true).
EVIDENCE: dict[str, Any] = {
    "row_count": 1000,
    "trustworthy_sample_size": True,
    "suspicious_metric": {"name": "r2", "value": 1.0},
    "target_feature_correlations": [
        {"feature": "log_target_v2", "correlation": 1.0, "method": "spearman"},
        {"feature": "near_target_proxy", "correlation": 0.95, "method": "pearson"},
    ],
    "perfect_match_rate": 0.987,
    "candidate_leak_columns": ["log_target_v2"],
    "notes": [],
}


def _submit(**payload: Any) -> MockClient:
    return MockClient(responses=[tool_use_response(SUBMIT_TOOL_NAME, payload)])


# --------------------------------------------------------------------------- #
# What the schema permits                                                     #
# --------------------------------------------------------------------------- #


def test_claim_schema_admits_exactly_one_statistic() -> None:
    """``statistic`` is a one-element enum, so only correlations are sayable.

    This is the structural reason the value channel was silent. Three of the
    four quantity families in the evidence have no expressible claim shape.
    """
    schema = _submit_input_schema(evidence_allowed_columns(EVIDENCE))
    statistic = schema["properties"]["claims"]["items"]["properties"]["statistic"]
    assert statistic["enum"] == ["correlation"]
    assert len(statistic["enum"]) == 1


def test_verifiable_quantities_are_a_strict_subset_of_the_evidence() -> None:
    """Exactly one of the four quantity families is verifiable.

    ``evidence_correlation_map`` is the entire reference set Tier B compares
    against. The metric value, the match rate and the row count are in the
    evidence, are quoted by narrators, and appear in no reference set.
    """
    verifiable = set(evidence_correlation_map(EVIDENCE))
    assert verifiable == {"log_target_v2", "near_target_proxy"}

    # Present in E, quoted by narrators, unverifiable by construction.
    assert EVIDENCE["suspicious_metric"]["value"] == 1.0
    assert EVIDENCE["perfect_match_rate"] == 0.987
    assert EVIDENCE["row_count"] == 1000
    for quantity in ("suspicious_metric", "perfect_match_rate", "row_count"):
        assert quantity not in verifiable


# --------------------------------------------------------------------------- #
# What the verifier actually checks                                           #
# --------------------------------------------------------------------------- #


def test_statistic_field_is_never_read_by_the_verifier() -> None:
    """The advertised triple is verified as a pair.

    The same claim, submitted with a ``statistic`` the schema does not admit,
    produces an identical verdict — because Tier B looks the column up in the
    correlation map and compares the value, and never inspects ``statistic``.
    """
    as_correlation = investigate_leakage_bound(
        EVIDENCE,
        client=_submit(
            verdict="leakage_likely",
            confidence="high",
            columns_referenced=["log_target_v2"],
            claims=[{"column": "log_target_v2", "statistic": "correlation", "value": 1.0}],
            narration="ok",
        ),
    )
    as_nonsense = investigate_leakage_bound(
        EVIDENCE,
        client=_submit(
            verdict="leakage_likely",
            confidence="high",
            columns_referenced=["log_target_v2"],
            claims=[
                # Not in the enum, and semantically a different statistic.
                {"column": "log_target_v2", "statistic": "perfect_match_rate", "value": 1.0}
            ],
            narration="ok",
        ),
    )
    assert as_nonsense["schema_rejections"] == as_correlation["schema_rejections"] == 0
    assert as_nonsense["had_unrecoverable_violation"] is False
    # The mislabelled claim is returned to the user intact, statistic and all.
    assert as_nonsense["claims"] == [
        {"column": "log_target_v2", "statistic": "perfect_match_rate", "value": 1.0}
    ]


def test_correlation_misquote_is_caught_the_channel_is_narrow_not_empty() -> None:
    """The positive control: within its one family, the channel does fire.

    This is what keeps the finding honest. The value channel is not broken and
    not unreachable — it is *narrow*. A misquoted correlation on an
    enum-admitted column is caught, retried and stripped exactly as documented.
    """
    stubborn = {
        "verdict": "leakage_likely",
        "confidence": "high",
        "columns_referenced": ["log_target_v2"],
        "claims": [{"column": "log_target_v2", "statistic": "correlation", "value": 0.42}],
        "narration": "misquote",
    }
    client = MockClient(
        responses=[tool_use_response(SUBMIT_TOOL_NAME, dict(stubborn)) for _ in range(3)]
    )
    out = investigate_leakage_bound(EVIDENCE, client=client, max_retries=2)
    assert out["schema_rejections"] == 3
    assert out["rejection_kinds"] == ["value", "value", "value"]
    assert out["claims"] == []
    assert out["had_unrecoverable_violation"] is True


# --------------------------------------------------------------------------- #
# The reachability limit                                                      #
# --------------------------------------------------------------------------- #
#
# The two tests below assert the property the manuscript's guarantee sentence
# implies — that a number the user sees equals the measured value — for the
# two quantity families a narrator most often quotes. Both currently fail.
#
# They are marked ``xfail(strict=True)`` rather than deleted or inverted,
# because the gap is the finding and a strict xfail is the only form that
# stays honest in both directions: it records the limit today, and the moment
# somebody widens the ``statistic`` enum and closes it, the suite turns red
# and forces the manuscript's §V-C explanation to be rewritten with it.
#
# Fixing them here is out of scope by construction: widening the enum changes
# what a cell can produce and would break comparability with the published
# runs. See the report accompanying this file.


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Reachability limit: perfect_match_rate has no expressible claim shape "
        "(statistic enum is ['correlation']), so a fabricated match rate is "
        "structurally invisible to Tier B. Widening the enum would change what a "
        "cell can produce and break comparability with the published runs."
    ),
)
def test_fabricated_match_rate_is_caught() -> None:
    """Evidence says 0.987; the narrator tells the user 0.42. Nothing fires."""
    out = investigate_leakage_bound(
        EVIDENCE,
        client=_submit(
            verdict="leakage_likely",
            confidence="high",
            columns_referenced=["log_target_v2"],
            claims=[{"column": "log_target_v2", "statistic": "correlation", "value": 1.0}],
            narration="Only 42% of predictions match the labels exactly.",
        ),
    )
    assert out["schema_rejections"] > 0 or out["had_unrecoverable_violation"] is True


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Reachability limit: suspicious_metric.value has no expressible claim "
        "shape, so a fabricated headline metric is structurally invisible to "
        "Tier B. Same comparability constraint as above."
    ),
)
def test_fabricated_metric_value_is_caught() -> None:
    """Evidence says R² = 1.0; the narrator tells the user 0.13. Nothing fires."""
    out = investigate_leakage_bound(
        EVIDENCE,
        client=_submit(
            verdict="leakage_likely",
            confidence="high",
            columns_referenced=["log_target_v2"],
            claims=[{"column": "log_target_v2", "statistic": "correlation", "value": 1.0}],
            narration="The reported R2 of 0.13 is well within a plausible range.",
        ),
    )
    assert out["schema_rejections"] > 0 or out["had_unrecoverable_violation"] is True


def test_fabricated_quantities_reach_the_user_verbatim() -> None:
    """The measured consequence, stated as a passing assertion.

    The same two fabrications as above, shown reaching the returned response
    untouched with a clean contract record. This is the fact the manuscript's
    §V-C explanation ("copying them is easy") does not account for: the
    channel did not decline to fire, it could not.
    """
    out = investigate_leakage_bound(
        EVIDENCE,
        client=_submit(
            verdict="leakage_likely",
            confidence="high",
            columns_referenced=["log_target_v2"],
            claims=[{"column": "log_target_v2", "statistic": "correlation", "value": 1.0}],
            narration="Only 42% of rows match and the R2 is 0.13.",
        ),
    )
    assert out["schema_rejections"] == 0
    assert out["rejection_kinds"] == []
    assert out["had_unrecoverable_violation"] is False
    assert out["omitted_critical_evidence"] is False
    # Both fabricated numbers are in the string handed to the renderer.
    assert "42%" in out["primary_hypothesis"]
    assert "0.13" in out["primary_hypothesis"]
