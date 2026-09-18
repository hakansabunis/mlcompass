"""What Tier B is for, when Tier A already caught everything.

The batteries make Tier B look redundant. `A-CONTRACT` carries both tiers and
reaches 0/200; `A-TIER-A-ONLY` carries the enum alone and also reaches 0/200.
A reader is entitled to ask what the verifier earns, and "it fired 12 times in
the stress arm" is an answer about one arm rather than an argument.

The argument is that **two of the three channels cannot be expressed in a
schema at all**, so no amount of Tier A closes them:

* *Entity* is enumerable, so a schema can express it. This is the one channel
  where the two tiers genuinely overlap, and the overlap is why the enum arm
  scores what the contract arm scores on this task.
* *Value* is not. The admissible set is "within 0.005 of a measured float",
  which is a predicate over a continuum. JSON Schema can say `"type":
  "number"`; it cannot say *which* number, because the answer depends on the
  entity named in a sibling field.
* *Completeness* is not. "A committed verdict must address the anchor" is a
  property of the whole response, not of any field. A schema constrains fields.

And even on the entity channel, Tier A is a request. The enum reaches the
provider as part of a tool definition, and whether it is honoured is the
provider's decision, unverifiable from outside. Tier B is the same check run
where we control it.

These are mutation tests: take a payload the contract accepts, change one thing
a schema cannot forbid, and require the change to be caught. They run offline
and cost nothing, which is the other reason they exist -- every other piece of
evidence in this study is a property of a model on a date.
"""

from __future__ import annotations

import pytest

from mlcompass.agents.evidence_contract import (
    LEAKAGE,
    PROFILE,
    BoundEvidence,
    build_payload_schema,
    strip_unsound,
    verify,
)

LEAK_EVIDENCE = {
    "suspicious_metric": {"name": "r2", "value": 1.0},
    "candidate_leak_columns": ["log_target_v2", "near_target_proxy"],
    "target_feature_correlations": [
        {"feature": "log_target_v2", "correlation": 0.9990},
        {"feature": "near_target_proxy", "correlation": 0.9987},
        {"feature": "feature_4", "correlation": -0.0215},
    ],
}

PROFILE_EVIDENCE = {
    "columns": [
        {
            "name": "age", "type": "numeric", "missing_count": 12,
            "missing_pct": 0.012,
            "stats": {"mean": 38.5, "std": 13.2, "min": 18.0, "max": 92.0,
                      "q25": 28.0, "q50": 37.0, "q75": 48.0},
            "outliers": {"iqr_count": 7, "z_score_count": 3},
        },
        {
            "name": "city", "type": "categorical", "missing_count": 240,
            "missing_pct": 0.24, "cardinality": 14,
        },
    ]
}


@pytest.fixture
def leak() -> BoundEvidence:
    return LEAKAGE.bind(LEAK_EVIDENCE)


@pytest.fixture
def prof() -> BoundEvidence:
    return PROFILE.bind(PROFILE_EVIDENCE)


def clean_leak() -> dict:
    """A payload the contract accepts. Every mutation below starts here."""
    return {
        "verdict": "leakage_likely",
        "confidence": "high",
        "columns_referenced": ["log_target_v2", "feature_4"],
        "claims": [
            {"column": "log_target_v2", "statistic": "correlation", "value": 0.9990},
            {"column": "feature_4", "statistic": "correlation", "value": -0.0215},
        ],
        "narration": "log_target_v2 is a transformed copy of the target.",
    }


def test_the_baseline_payload_is_actually_clean(leak: BoundEvidence) -> None:
    assert not verify(clean_leak(), leak, LEAKAGE)


# --------------------------------------------------------------------------- #
# The value channel: no schema can express this                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value,caught",
    [
        (0.9990, False),   # the measured value
        (0.9994, False),   # +0.0004, inside tolerance
        (0.9945, False),   # -0.0045, inside tolerance
        (0.9939, True),    # -0.0051, outside by 0.0001
        (0.99, True),      # plausible-looking and wrong
        (0.0, True),
        (-0.9990, True),   # right magnitude, wrong sign
        # The suspicious metric's own value. A narrator confusing R^2 = 1.0
        # with this column's correlation of 0.9990 is off by 0.001, which is
        # inside tolerance, so the value channel does not fire. That is not a
        # bug and it is worth stating: on evidence whose quantities sit this
        # close together, tolerance decides what counts as a misquote, and
        # 0.005 was chosen for float-formatting slack rather than from any
        # analysis of how far apart the measured values are. The entity channel
        # is what separates these two, because `r2` is not a column.
        (1.0, False),
    ],
)
def test_value_tolerance_boundary(leak: BoundEvidence, value: float, caught: bool) -> None:
    """A float is a continuum; an enum is a list. Only code can check this.

    The boundary cases matter because 0.005 is the contract's promise. A check
    that is loose by a thousandth silently widens what may be said -- and, as
    the last case shows, a tolerance wider than the gap between two measured
    quantities makes confusing them invisible on this channel.
    """
    p = clean_leak()
    p["claims"][0]["value"] = value
    assert bool(verify(p, leak, LEAKAGE).value) is caught


def test_the_schema_cannot_express_the_value_constraint(leak: BoundEvidence) -> None:
    """Stated as an assertion rather than as prose: `value` is an open number.

    This is the structural reason Tier A cannot close this channel, and it
    holds however the schema is configured.
    """
    for enforce in (True, False):
        schema = build_payload_schema(leak, LEAKAGE, enforce_enum=enforce)
        value_field = schema["properties"]["claims"]["items"]["properties"]["value"]
        assert value_field == {"type": "number"}
        assert "enum" not in value_field


def test_a_boolean_is_not_a_measurement(leak: BoundEvidence) -> None:
    """`True == 1` in Python, and `"type": "number"` admits neither."""
    p = clean_leak()
    p["claims"][0]["value"] = True
    assert verify(p, leak, LEAKAGE).value


@pytest.mark.parametrize("junk", [None, "0.9990", [0.9990], {"v": 0.9990}])
def test_non_numeric_values_are_caught(leak: BoundEvidence, junk: object) -> None:
    p = clean_leak()
    p["claims"][0]["value"] = junk
    assert verify(p, leak, LEAKAGE).value


# --------------------------------------------------------------------------- #
# Completeness: a property of the response, not of a field                     #
# --------------------------------------------------------------------------- #


def test_omission_fires_when_the_anchor_is_dropped(leak: BoundEvidence) -> None:
    p = clean_leak()
    p["columns_referenced"] = ["feature_4"]
    p["claims"] = [{"column": "feature_4", "statistic": "correlation", "value": -0.0215}]
    assert verify(p, leak, LEAKAGE).omitted


def test_abstaining_is_not_omitting(leak: BoundEvidence) -> None:
    """The rule applies to a commitment. Declining to commit is not a failure."""
    p = clean_leak()
    p["verdict"] = "cannot_determine"
    p["columns_referenced"] = []
    p["claims"] = []
    assert not verify(p, leak, LEAKAGE).omitted


def test_the_anchor_may_be_addressed_through_a_claim_alone(leak: BoundEvidence) -> None:
    p = clean_leak()
    p["columns_referenced"] = ["feature_4"]
    assert not verify(p, leak, LEAKAGE).omitted


def test_no_schema_field_carries_the_completeness_rule(leak: BoundEvidence) -> None:
    """`columns_referenced` is an array with no minimum and no required member.

    A schema can require the field to exist. It cannot require a particular
    value inside it, which is what completeness is.
    """
    schema = build_payload_schema(leak, LEAKAGE)
    arr = schema["properties"]["columns_referenced"]
    assert arr["type"] == "array"
    assert "minItems" not in arr
    assert "contains" not in arr


def test_stripping_can_create_an_omission(leak: BoundEvidence) -> None:
    """The case the repair asymmetry is about, as an executable example.

    The anchor is addressed only through a claim, and that claim is unsound.
    Deleting it is the only safe repair, and deleting it removes the only
    mention of the anchor. So a response can enter repair complete and leave it
    incomplete, which is why an omission is flagged rather than fixed: the
    alternative is inserting a reference the narrator never made.
    """
    p = clean_leak()
    p["columns_referenced"] = ["feature_4"]
    p["claims"] = [
        {"column": "log_target_v2", "statistic": "correlation", "value": 0.5},
        {"column": "feature_4", "statistic": "correlation", "value": -0.0215},
    ]
    before = verify(p, leak, LEAKAGE)
    assert not before.omitted          # the anchor is mentioned
    assert before.value                # but by an unsound claim

    cited, claims = strip_unsound(p, leak, LEAKAGE)
    survivors = set(cited) | {c["column"] for c in claims}
    assert leak.anchor not in survivors


# --------------------------------------------------------------------------- #
# The entity channel: where Tier A does overlap, and where it still does not   #
# --------------------------------------------------------------------------- #


def test_entity_is_the_one_channel_a_schema_can_express(leak: BoundEvidence) -> None:
    schema = build_payload_schema(leak, LEAKAGE, enforce_enum=True)
    assert schema["properties"]["columns_referenced"]["items"]["enum"] == list(
        leak.domains["columns_referenced"]
    )


@pytest.mark.parametrize(
    "name",
    [
        "r2",                    # in E, not a column: the misfiling failure
        "suspicious_metric",     # an evidence KEY, not a value
        "correlation",           # a statistic name
        "log_target_v3",         # one character from a real column
        "LOG_TARGET_V2",         # case
        " log_target_v2",        # leading space
        "log_target_v2 ",        # trailing space
        "",                      # empty
    ],
)
def test_entity_mutations_are_caught_without_the_enum(
    leak: BoundEvidence, name: str
) -> None:
    """The stress configuration: enum removed, so only Tier B stands between
    these and the user. Near-misses are included deliberately -- a check that
    catches `banana` and not `log_target_v3` is not doing the job."""
    p = clean_leak()
    p["columns_referenced"] = [name, "feature_4"]
    assert verify(p, leak, LEAKAGE).entity == (name,)


def test_a_provider_ignoring_the_enum_is_still_caught(leak: BoundEvidence) -> None:
    """Tier A is a request. This is the same payload a provider that honoured
    the enum could not have produced, and Tier B does not care which happened."""
    p = clean_leak()
    p["columns_referenced"] = ["r2"]
    v = verify(p, leak, LEAKAGE)
    assert v.entity == ("r2",)
    cited, _ = strip_unsound(p, leak, LEAKAGE)
    assert cited == []


# --------------------------------------------------------------------------- #
# The statistic channel, which only exists once a task has more than one       #
# --------------------------------------------------------------------------- #


def test_right_column_wrong_statistic_is_caught(prof: BoundEvidence) -> None:
    """`age.missing_pct` is 0.012 and `age.mean` is 38.5.

    The value is a real measurement of that column, under the wrong name. An
    entity-keyed table holds one number per column and would compare against
    whichever it kept, so this is the mutation that distinguishes a pair-keyed
    value table from an entity-keyed one.
    """
    p = {
        "verdict": "needs_cleaning", "confidence": "high",
        "columns_referenced": ["city"],
        "claims": [{"column": "age", "statistic": "mean", "value": 0.012}],
        "narration": "x",
    }
    assert verify(p, prof, PROFILE).value


def test_a_statistic_the_column_does_not_carry_is_caught(prof: BoundEvidence) -> None:
    """`city` is categorical: it has a cardinality and no mean. The statistic is
    in the domain for the evidence as a whole and not for this column."""
    p = {
        "verdict": "needs_cleaning", "confidence": "high",
        "columns_referenced": ["city"],
        "claims": [{"column": "city", "statistic": "mean", "value": 14.0}],
        "narration": "x",
    }
    assert verify(p, prof, PROFILE).value


def test_the_correct_pair_passes(prof: BoundEvidence) -> None:
    p = {
        "verdict": "needs_cleaning", "confidence": "high",
        "columns_referenced": ["city"],
        "claims": [{"column": "city", "statistic": "cardinality", "value": 14.0}],
        "narration": "x",
    }
    assert not verify(p, prof, PROFILE)


# --------------------------------------------------------------------------- #
# Composition                                                                  #
# --------------------------------------------------------------------------- #


def test_all_three_channels_fire_together(leak: BoundEvidence) -> None:
    p = {
        "verdict": "leakage_likely", "confidence": "high",
        "columns_referenced": ["r2"],
        "claims": [{"column": "feature_4", "statistic": "correlation", "value": 0.5}],
        "narration": "x",
    }
    v = verify(p, leak, LEAKAGE)
    assert v.entity and v.value and v.omitted
    assert v.kinds == "entity+value+omission"


def test_repair_is_idempotent(leak: BoundEvidence) -> None:
    """Stripping a stripped response changes nothing. A repair loop that could
    keep deleting would eventually empty a sound answer."""
    p = clean_leak()
    p["columns_referenced"].append("r2")
    cited, claims = strip_unsound(p, leak, LEAKAGE)
    again = strip_unsound(
        {**p, "columns_referenced": cited, "claims": claims}, leak, LEAKAGE
    )
    assert again == (cited, claims)


def test_an_empty_payload_is_not_a_violation(leak: BoundEvidence) -> None:
    """A declined tool call is a transport outcome, not a faithfulness failure.
    Scoring it as one would inflate every rate in the study."""
    assert not verify({}, leak, LEAKAGE)
