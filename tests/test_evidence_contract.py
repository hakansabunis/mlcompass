"""The generic contract layer, and the promise that extracting it changed nothing.

Two jobs. The first is regression: the leakage path now emits its schema and
runs its three Tier B channels through :mod:`mlcompass.agents.evidence_contract`,
and every published measurement was taken under the old inline code. A schema
that differs by one enum value, or a corrective retry that differs by one word,
is a different treatment -- so these tests pin both against literals rather than
against a re-derivation that could drift with the code.

That is not hypothetical. The first draft of the contract spec restated the
verdict vocabulary instead of owning it, got two of the four values wrong, and
silently changed the shipped tool schema. All 38 tests over that path passed,
because none of them looked at the schema. ``test_verdict_enum_is_exact`` is
the one that would have caught it.

The second job is the thing the layer exists for: showing that a second
evidence shape binds to the same verifier. The profile contract carries a
dozen statistics per column where leakage carries one, which is precisely the
case an entity-keyed value table gets wrong -- comparing a claimed ``mean``
against a stored ``missing_pct`` and passing it.
"""

from __future__ import annotations

import pytest

from mlcompass.agents.evidence_contract import (
    LEAKAGE,
    PROFILE,
    BoundEvidence,
    build_payload_schema,
    correction_text,
    strip_unsound,
    verify,
)
from mlcompass.agents.leakage_investigator import (
    SUBMIT_TOOL_NAME,
    build_submit_investigation_tool,
    build_submit_investigation_tool_openai,
    evidence_allowed_columns,
    evidence_correlation_map,
    top_candidate,
)

# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

LEAK_EVIDENCE = {
    "ok": True,
    "row_count": 1000,
    "suspicious_metric": {"name": "r2", "value": 1.0},
    "candidate_leak_columns": ["log_target_v2", "near_target_proxy"],
    "target_feature_correlations": [
        {"feature": "log_target_v2", "correlation": 0.9990, "method": "spearman"},
        {"feature": "near_target_proxy", "correlation": 0.9987, "method": "pearson"},
        {"feature": "feature_4", "correlation": -0.0215, "method": "pearson"},
    ],
}

PROFILE_EVIDENCE = {
    "shape": {"rows": 1000, "cols": 3},
    "columns": [
        {
            "name": "age",
            "type": "numeric",
            "missing_count": 12,
            "missing_pct": 0.012,
            "zero_ratio": 0.0,
            "stats": {"mean": 38.5, "std": 13.2, "min": 18.0, "max": 92.0,
                      "q25": 28.0, "q50": 37.0, "q75": 48.0},
            "outliers": {"iqr_count": 7, "z_score_count": 3},
        },
        {
            "name": "city",
            "type": "categorical",
            "missing_count": 240,
            "missing_pct": 0.24,
            "cardinality": 14,
        },
        {
            "name": "income",
            "type": "numeric",
            "missing_count": 0,
            "missing_pct": 0.0,
            "zero_ratio": 0.08,
            "stats": {"mean": 51200.0, "std": 9100.0, "min": 0.0, "max": 250000.0,
                      "q25": 44000.0, "q50": 50000.0, "q75": 58000.0},
            "outliers": {"iqr_count": 31, "z_score_count": 12},
        },
    ],
}


@pytest.fixture
def leak_bound() -> BoundEvidence:
    return LEAKAGE.bind(LEAK_EVIDENCE)


@pytest.fixture
def profile_bound() -> BoundEvidence:
    return PROFILE.bind(PROFILE_EVIDENCE)


# --------------------------------------------------------------------------- #
# Regression: the leakage contract is unchanged                                #
# --------------------------------------------------------------------------- #


def test_verdict_enum_is_exact() -> None:
    """The four values, spelled out. This is the test the first draft needed."""
    assert set(LEAKAGE.verdict_values) == {
        "leakage_likely",
        "leakage_uncertain",
        "score_legitimate",
        "cannot_determine",
    }
    assert set(LEAKAGE.confidence_values) == {
        "high",
        "medium",
        "low",
        "cannot_determine",
    }


def test_binder_matches_the_shipped_helpers(leak_bound: BoundEvidence) -> None:
    """bind() agrees with the three functions callers have always used."""
    assert list(leak_bound.domains["columns_referenced"]) == evidence_allowed_columns(
        LEAK_EVIDENCE
    )
    assert {k[0]: v for k, v in leak_bound.values.items()} == evidence_correlation_map(
        LEAK_EVIDENCE
    )
    assert leak_bound.anchor == top_candidate(LEAK_EVIDENCE)
    assert all(stat == "correlation" for _, stat in leak_bound.values)


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("enforce_enum", [True, False])
def test_schema_shape_is_pinned(enforce_enum: bool, strict: bool) -> None:
    """Field names, required list and the statistic enum, against literals."""
    tool = build_submit_investigation_tool(
        ["a", "b"], enforce_enum=enforce_enum, strict=strict
    )
    schema = tool["input_schema"]
    props = schema["properties"]
    assert set(props) == {
        "verdict",
        "confidence",
        "columns_referenced",
        "claims",
        "narration",
        "recommended_checks",
    }
    # statistic keeps its enum in every configuration: Tier A gates the column
    # domain only, which is what the stress arm measured.
    assert props["claims"]["items"]["properties"]["statistic"]["enum"] == ["correlation"]
    if enforce_enum:
        assert props["columns_referenced"]["items"]["enum"] == ["a", "b"]
        assert props["claims"]["items"]["properties"]["column"]["enum"] == ["a", "b"]
    else:
        assert "enum" not in props["columns_referenced"]["items"]
        assert "enum" not in props["claims"]["items"]["properties"]["column"]
    if strict:
        assert schema["additionalProperties"] is False
        assert schema["required"] == sorted(props)
    else:
        assert schema["required"] == [
            "verdict",
            "confidence",
            "columns_referenced",
            "claims",
            "narration",
        ]


def test_openai_wrapper_carries_the_same_inner_schema() -> None:
    a = build_submit_investigation_tool(["a"])["input_schema"]
    b = build_submit_investigation_tool_openai(["a"])["function"]["parameters"]
    assert a == b


def test_corrective_retry_wording_is_pinned(leak_bound: BoundEvidence) -> None:
    """The exact sentence the published arms were measured with."""
    payload = {
        "verdict": "leakage_likely",
        "confidence": "high",
        "columns_referenced": ["r2"],
        "claims": [{"column": "feature_4", "statistic": "correlation", "value": 0.5}],
        "narration": "x",
    }
    violations = verify(payload, leak_bound, LEAKAGE)
    assert violations.kinds == "entity+value+omission"
    text = correction_text(violations, leak_bound, LEAKAGE, tool_name=SUBMIT_TOOL_NAME)
    assert text.startswith("\n\nYour previous answer violated the contract: ")
    assert "you cited columns NOT in the evidence dictionary: ['r2']" in text
    # One statistic in the domain, so the violation names the column alone --
    # exactly as the shipped code did before the layer was extracted.
    assert "feature_4: cited 0.5, evidence says -0.0215" in text
    assert "did not address the top-ranked candidate-leak column 'log_target_v2'" in text
    assert text.endswith(". Re-answer through submit_investigation.")


# --------------------------------------------------------------------------- #
# The three channels                                                           #
# --------------------------------------------------------------------------- #


def test_clean_payload_fires_nothing(leak_bound: BoundEvidence) -> None:
    payload = {
        "verdict": "leakage_likely",
        "confidence": "high",
        "columns_referenced": ["log_target_v2", "feature_4"],
        "claims": [
            {"column": "log_target_v2", "statistic": "correlation", "value": 0.9990}
        ],
        "narration": "x",
    }
    assert not verify(payload, leak_bound, LEAKAGE)


def test_abstention_is_not_an_omission(leak_bound: BoundEvidence) -> None:
    """A narrator that declines to commit has not skipped the anchor."""
    payload = {
        "verdict": "cannot_determine",
        "confidence": "low",
        "columns_referenced": [],
        "claims": [],
        "narration": "x",
    }
    assert verify(payload, leak_bound, LEAKAGE).omitted is False


def test_strip_deletes_and_never_inserts(leak_bound: BoundEvidence) -> None:
    """Deletion is the only safe repair; the anchor is never added back."""
    payload = {
        "verdict": "leakage_likely",
        "confidence": "high",
        "columns_referenced": ["r2", "feature_4"],
        "claims": [
            {"column": "feature_4", "statistic": "correlation", "value": 0.5},
            {"column": "feature_4", "statistic": "correlation", "value": -0.0215},
        ],
        "narration": "x",
    }
    cited, claims = strip_unsound(payload, leak_bound, LEAKAGE)
    assert cited == ["feature_4"]
    assert claims == [{"column": "feature_4", "statistic": "correlation", "value": -0.0215}]
    assert leak_bound.anchor not in cited


def test_booleans_are_not_numbers(leak_bound: BoundEvidence) -> None:
    """`True == 1` in Python, and a claim of True is not a measurement."""
    payload = {
        "verdict": "leakage_likely",
        "confidence": "high",
        "columns_referenced": ["log_target_v2"],
        "claims": [{"column": "feature_4", "statistic": "correlation", "value": True}],
        "narration": "x",
    }
    assert verify(payload, leak_bound, LEAKAGE).value


# --------------------------------------------------------------------------- #
# The second instance: a shape the entity-keyed check would get wrong          #
# --------------------------------------------------------------------------- #


def test_profile_binds_a_wide_domain_and_many_statistics(
    profile_bound: BoundEvidence,
) -> None:
    assert profile_bound.domains["columns_referenced"] == ("age", "city", "income")
    stats = set(profile_bound.domains["claims[].statistic"])
    assert {"mean", "std", "missing_pct", "cardinality", "iqr_outliers"} <= stats
    # A categorical column carries cardinality and no mean; the domain is the
    # union of what the columns actually carry, not a frozen list.
    assert ("city", "cardinality") in profile_bound.values
    assert ("city", "mean") not in profile_bound.values
    assert profile_bound.anchor == "city"  # the most missing data


def test_value_table_is_keyed_on_the_pair(profile_bound: BoundEvidence) -> None:
    """The reason the generic layer had to exist.

    ``age.mean`` is 38.5 and ``age.missing_pct`` is 0.012. A table keyed on the
    entity alone holds one of them, so a claim naming the other statistic is
    checked against the wrong number. Here the claim is a correct missing_pct
    reported under the wrong statistic name and it must not pass.
    """
    payload = {
        "verdict": "needs_cleaning",
        "confidence": "high",
        "columns_referenced": ["city"],
        "claims": [{"column": "age", "statistic": "mean", "value": 0.012}],
        "narration": "x",
    }
    violations = verify(payload, profile_bound, PROFILE)
    assert violations.value
    assert "age.mean: cited 0.012, evidence says 38.5000" in violations.value[0]


def test_profile_correction_names_the_statistic(profile_bound: BoundEvidence) -> None:
    """More than one statistic in the domain, so the message must disambiguate."""
    payload = {
        "verdict": "needs_cleaning",
        "confidence": "high",
        "columns_referenced": ["city"],
        "claims": [{"column": "income", "statistic": "std", "value": 1.0}],
        "narration": "x",
    }
    text = correction_text(
        verify(payload, profile_bound, PROFILE), profile_bound, PROFILE,
        tool_name="submit_profile",
    )
    assert "income.std: cited 1.0, evidence says 9100.0000" in text
    assert text.endswith(". Re-answer through submit_profile.")


def test_profile_tier_a_gates_the_statistic_enum(profile_bound: BoundEvidence) -> None:
    """Unlike leakage, the profile's statistic domain comes from E, so Tier A
    gates it: dropping the binding drops that enum too."""
    on = build_payload_schema(profile_bound, PROFILE, enforce_enum=True)
    off = build_payload_schema(profile_bound, PROFILE, enforce_enum=False)
    assert "enum" in on["properties"]["claims"]["items"]["properties"]["statistic"]
    assert "enum" not in off["properties"]["claims"]["items"]["properties"]["statistic"]
    assert "enum" not in off["properties"]["columns_referenced"]["items"]


def test_the_same_verifier_serves_both_contracts(
    leak_bound: BoundEvidence, profile_bound: BoundEvidence
) -> None:
    """No branch in verify() names either task."""
    bad_leak = {
        "verdict": "leakage_likely", "confidence": "high",
        "columns_referenced": ["nope"], "claims": [], "narration": "x",
    }
    bad_profile = {
        "verdict": "needs_cleaning", "confidence": "high",
        "columns_referenced": ["nope"], "claims": [], "narration": "x",
    }
    assert verify(bad_leak, leak_bound, LEAKAGE).entity == ("nope",)
    assert verify(bad_profile, profile_bound, PROFILE).entity == ("nope",)
