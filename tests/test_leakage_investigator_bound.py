"""Tests for the evidence-bound runtime-schema leakage contract.

These tests exercise the mechanism that is the paper's central claim — the
``columns_referenced`` enum generated at call time from the evidence dict
(Tier A) plus the deterministic re-validation + retry + strip (Tier B) — end
to end, with no real API call. They are the regression guard the earlier
review found missing: the prior suite covered the deterministic detector and
the prose narrator, but nothing exercised the schema-bound rejection path.
"""

from __future__ import annotations

import json

from agentlite.testing import MockClient, tool_use_response

from mlcompass.agents.leakage_investigator import (
    SUBMIT_TOOL_NAME,
    build_submit_investigation_tool,
    evidence_allowed_columns,
    investigate_leakage_bound,
)

# A representative evidence dict in the exact shape detect_leakage emits.
EVIDENCE: dict = {
    "row_count": 1000,
    "trustworthy_sample_size": True,
    "suspicious_metric": {"name": "r2", "value": 1.0},
    "target_feature_correlations": [
        {"feature": "log_target_v2", "correlation": 1.0, "method": "spearman"},
        {"feature": "near_target_proxy", "correlation": 0.95, "method": "pearson"},
        {"feature": "feature_3", "correlation": 0.21, "method": "pearson"},
    ],
    "perfect_match_rate": 0.987,
    "candidate_leak_columns": ["log_target_v2"],
    "notes": [],
}

# The runtime enum domain: candidate leaks ∪ every correlated feature.
ALLOWED = ["feature_3", "log_target_v2", "near_target_proxy"]


# --------------------------------------------------------------------------- #
# Enum-domain construction                                                    #
# --------------------------------------------------------------------------- #


def test_evidence_allowed_columns_is_union_sorted() -> None:
    assert evidence_allowed_columns(EVIDENCE) == ALLOWED


def test_evidence_allowed_columns_empty_when_no_evidence() -> None:
    empty = {"candidate_leak_columns": [], "target_feature_correlations": []}
    assert evidence_allowed_columns(empty) == []


def test_tool_schema_binds_enum_to_evidence() -> None:
    tool = build_submit_investigation_tool(ALLOWED)
    enum = tool["input_schema"]["properties"]["columns_referenced"]["items"]["enum"]
    assert enum == ALLOWED
    assert tool["name"] == SUBMIT_TOOL_NAME


# --------------------------------------------------------------------------- #
# Happy path — model cites only in-evidence columns                           #
# --------------------------------------------------------------------------- #


def test_clean_response_passes_without_rejection() -> None:
    client = MockClient(
        responses=[
            tool_use_response(
                SUBMIT_TOOL_NAME,
                {
                    "verdict": "leakage_likely",
                    "confidence": "high",
                    "columns_referenced": ["log_target_v2"],
                    "narration": "log_target_v2 has Spearman 1.00 with the target.",
                    "recommended_checks": ["Inspect the feature pipeline for a leak."],
                },
            )
        ]
    )
    out = investigate_leakage_bound(EVIDENCE, client=client)
    assert out["verdict"] == "leakage_likely"
    assert out["confidence"] == "high"
    assert out["columns_referenced"] == ["log_target_v2"]
    assert out["schema_rejections"] == 0
    assert out["had_unrecoverable_violation"] is False
    assert out["evidence_bound"] is True
    assert client.create_call_count == 1


def test_runtime_enum_is_sent_to_the_api() -> None:
    client = MockClient(
        responses=[
            tool_use_response(
                SUBMIT_TOOL_NAME,
                {
                    "verdict": "leakage_likely",
                    "confidence": "high",
                    "columns_referenced": ["log_target_v2"],
                    "narration": "ok",
                },
            )
        ]
    )
    investigate_leakage_bound(EVIDENCE, client=client)
    tools = client.last_create_kwargs["tools"]
    enum = tools[0]["input_schema"]["properties"]["columns_referenced"]["items"]["enum"]
    assert enum == ALLOWED
    # The contract forces the answer through the submit tool.
    assert client.last_create_kwargs["tool_choice"] == {"type": "tool", "name": SUBMIT_TOOL_NAME}


# --------------------------------------------------------------------------- #
# Phantom path — Tier B catches what the prompt/enum did not                   #
# --------------------------------------------------------------------------- #


def test_phantom_then_clean_triggers_one_retry() -> None:
    # First answer fabricates a column not in the evidence; the contract
    # rejects it deterministically and retries; the second answer is clean.
    client = MockClient(
        responses=[
            tool_use_response(
                SUBMIT_TOOL_NAME,
                {
                    "verdict": "leakage_likely",
                    "confidence": "high",
                    "columns_referenced": ["log_target_v2", "revenue"],  # "revenue" is phantom
                    "narration": "phantom attempt",
                },
            ),
            tool_use_response(
                SUBMIT_TOOL_NAME,
                {
                    "verdict": "leakage_likely",
                    "confidence": "high",
                    "columns_referenced": ["log_target_v2"],
                    "narration": "clean retry",
                },
            ),
        ]
    )
    out = investigate_leakage_bound(EVIDENCE, client=client)
    assert out["schema_rejections"] == 1
    assert out["columns_referenced"] == ["log_target_v2"]
    assert out["had_unrecoverable_violation"] is False
    assert client.create_call_count == 2


def test_persistent_phantom_is_stripped_after_retry_budget() -> None:
    # The model fabricates on every attempt. After the retry budget is spent,
    # the residual phantom is removed deterministically — the worst-case
    # guarantee: no out-of-evidence column ever reaches the caller.
    phantom = tool_use_response(
        SUBMIT_TOOL_NAME,
        {
            "verdict": "leakage_likely",
            "confidence": "high",
            "columns_referenced": ["log_target_v2", "shadow_target"],  # phantom every time
            "narration": "stubborn phantom",
        },
    )
    client = MockClient(responses=[phantom, _clone(phantom), _clone(phantom)])
    out = investigate_leakage_bound(EVIDENCE, client=client, max_retries=2)
    assert out["had_unrecoverable_violation"] is True
    assert "shadow_target" not in out["columns_referenced"]
    assert out["columns_referenced"] == ["log_target_v2"]
    assert out["schema_rejections"] == 3  # initial + 2 retries, each a violation


def test_invalid_verdict_is_clamped() -> None:
    client = MockClient(
        responses=[
            tool_use_response(
                SUBMIT_TOOL_NAME,
                {
                    "verdict": "definitely_leaking",  # not a valid enum value
                    "confidence": "extremely_high",  # not a valid enum value
                    "columns_referenced": ["log_target_v2"],
                    "narration": "ok",
                },
            )
        ]
    )
    out = investigate_leakage_bound(EVIDENCE, client=client)
    assert out["verdict"] == "cannot_determine"
    assert out["confidence"] == "cannot_determine"


def _clone(resp):  # type: ignore[no-untyped-def]
    """MockClient.create pops responses; re-use the same payload three times."""
    block = resp.content[-1]
    return tool_use_response(block.name, dict(block.input))


# --------------------------------------------------------------------------- #
# OpenAI-compatible provider path (DeepSeek, etc.)                             #
# --------------------------------------------------------------------------- #
#
# DeepSeek and other OpenAI-compatible endpoints do not strictly enforce the
# JSON-schema enum during decoding, so Tier B (deterministic validation) is the
# load-bearing guarantee there. These tests use a minimal fake OpenAI client.


class _FakeFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments  # JSON string, per the OpenAI schema


class _FakeToolCall:
    def __init__(self, name: str, args: dict) -> None:
        self.id = "call_1"
        self.function = _FakeFunction(name, json.dumps(args))


class _FakeMessage:
    def __init__(self, tool_calls: list) -> None:
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message: _FakeMessage) -> None:
        self.message = message


class _FakeOpenAIResponse:
    def __init__(self, choices: list) -> None:
        self.choices = choices


class FakeOpenAIClient:
    """Minimal stand-in for an OpenAI-compatible client (``chat.completions``)."""

    def __init__(self, payloads: list) -> None:
        self._payloads = list(payloads)  # each: dict of tool args, or None to decline
        self.create_call_count = 0
        self.last_create_kwargs: dict = {}
        self.chat = self
        self.completions = self

    def create(self, **kwargs):  # type: ignore[no-untyped-def]
        self.create_call_count += 1
        self.last_create_kwargs = kwargs
        payload = self._payloads.pop(0)
        calls = [] if payload is None else [_FakeToolCall(SUBMIT_TOOL_NAME, payload)]
        return _FakeOpenAIResponse([_FakeChoice(_FakeMessage(calls))])


def test_openai_clean_response_passes() -> None:
    client = FakeOpenAIClient(
        payloads=[
            {
                "verdict": "leakage_likely",
                "confidence": "high",
                "columns_referenced": ["log_target_v2"],
                "narration": "ok",
            }
        ]
    )
    out = investigate_leakage_bound(
        EVIDENCE, client=client, model="deepseek-chat", provider="openai"
    )
    assert out["columns_referenced"] == ["log_target_v2"]
    assert out["schema_rejections"] == 0
    assert out["had_unrecoverable_violation"] is False
    # OpenAI-format request shape: function tool + forced tool_choice.
    tools = client.last_create_kwargs["tools"]
    enum = tools[0]["function"]["parameters"]["properties"]["columns_referenced"]["items"]["enum"]
    assert enum == ALLOWED
    assert client.last_create_kwargs["tool_choice"] == "required"


def test_openai_phantom_then_clean_retries() -> None:
    client = FakeOpenAIClient(
        payloads=[
            {
                "verdict": "leakage_likely",
                "confidence": "high",
                "columns_referenced": ["log_target_v2", "revenue"],  # phantom
                "narration": "phantom",
            },
            {
                "verdict": "leakage_likely",
                "confidence": "high",
                "columns_referenced": ["log_target_v2"],
                "narration": "clean",
            },
        ]
    )
    out = investigate_leakage_bound(
        EVIDENCE, client=client, model="deepseek-chat", provider="openai"
    )
    assert out["schema_rejections"] == 1
    assert out["columns_referenced"] == ["log_target_v2"]
    assert out["had_unrecoverable_violation"] is False
    assert client.create_call_count == 2


def test_openai_persistent_phantom_is_stripped() -> None:
    payload = {
        "verdict": "leakage_likely",
        "confidence": "high",
        "columns_referenced": ["log_target_v2", "shadow_target"],  # phantom every time
        "narration": "stubborn",
    }
    client = FakeOpenAIClient(payloads=[dict(payload), dict(payload), dict(payload)])
    out = investigate_leakage_bound(
        EVIDENCE, client=client, model="deepseek-chat", provider="openai", max_retries=2
    )
    assert out["had_unrecoverable_violation"] is True
    assert "shadow_target" not in out["columns_referenced"]
    assert out["columns_referenced"] == ["log_target_v2"]
    assert out["schema_rejections"] == 3


def test_openai_declined_tool_call_is_safe() -> None:
    # If the model declines to call the tool, no column reaches the user.
    client = FakeOpenAIClient(payloads=[None])
    out = investigate_leakage_bound(
        EVIDENCE, client=client, model="deepseek-chat", provider="openai"
    )
    assert out["columns_referenced"] == []
    assert out["had_unrecoverable_violation"] is False


# --------------------------------------------------------------------------- #
# Verified claims (value soundness) + completeness (omission)                  #
# --------------------------------------------------------------------------- #
#
# The contract upgrade: quantitative claims are structured {column, statistic,
# value} triples checked against the measured correlations (VALUE_TOLERANCE),
# and a committed verdict must address the top-ranked candidate column.


def _resp(verdict="leakage_likely", cols=None, claims=None, narration="ok"):
    payload = {
        "verdict": verdict,
        "confidence": "high",
        "columns_referenced": cols if cols is not None else ["log_target_v2"],
        "narration": narration,
    }
    if claims is not None:
        payload["claims"] = claims
    return tool_use_response(SUBMIT_TOOL_NAME, payload)


def test_claim_with_exact_value_passes() -> None:
    client = MockClient(
        responses=[
            _resp(claims=[{"column": "log_target_v2", "statistic": "correlation", "value": 1.0}])
        ]
    )
    out = investigate_leakage_bound(EVIDENCE, client=client)
    assert out["schema_rejections"] == 0
    assert out["claims"] == [{"column": "log_target_v2", "statistic": "correlation", "value": 1.0}]


def test_claim_rounded_to_two_decimals_passes_tolerance() -> None:
    # Evidence says near_target_proxy = 0.95; citing 0.95 (or 0.9512-style
    # rounding within 0.005) must pass — honest rounding is not fabrication.
    client = MockClient(
        responses=[
            _resp(
                cols=["log_target_v2", "near_target_proxy"],
                claims=[
                    {"column": "near_target_proxy", "statistic": "correlation", "value": 0.948}
                ],
            )
        ]
    )
    out = investigate_leakage_bound(EVIDENCE, client=client)
    assert out["schema_rejections"] == 0
    assert len(out["claims"]) == 1


def test_misquoted_value_triggers_retry_then_clean() -> None:
    # First answer misquotes the correlation (0.85 vs measured 1.0) — Tier B
    # rejects deterministically and retries; second answer is exact.
    client = MockClient(
        responses=[
            _resp(claims=[{"column": "log_target_v2", "statistic": "correlation", "value": 0.85}]),
            _resp(claims=[{"column": "log_target_v2", "statistic": "correlation", "value": 1.0}]),
        ]
    )
    out = investigate_leakage_bound(EVIDENCE, client=client)
    assert out["schema_rejections"] == 1
    assert out["claims"][0]["value"] == 1.0
    assert out["had_unrecoverable_violation"] is False
    assert client.create_call_count == 2


def test_persistent_misquote_is_stripped() -> None:
    bad = _resp(claims=[{"column": "log_target_v2", "statistic": "correlation", "value": 0.5}])
    client = MockClient(responses=[bad, _clone(bad), _clone(bad)])
    out = investigate_leakage_bound(EVIDENCE, client=client, max_retries=2)
    assert out["had_unrecoverable_violation"] is True
    assert out["claims"] == []  # the unsound claim never reaches the user
    assert out["columns_referenced"] == ["log_target_v2"]  # entity channel intact


def test_omitting_top_candidate_triggers_retry_then_flag() -> None:
    # The narrator commits to leakage_likely but never addresses the top
    # candidate (log_target_v2) — completeness violation every attempt.
    evasive = _resp(cols=["near_target_proxy"])
    client = MockClient(responses=[evasive, _clone(evasive), _clone(evasive)])
    out = investigate_leakage_bound(EVIDENCE, client=client, max_retries=2)
    assert out["schema_rejections"] == 3
    assert out["omitted_critical_evidence"] is True
    assert out["had_unrecoverable_violation"] is False  # nothing unsound, just incomplete


def test_cannot_determine_is_not_an_omission() -> None:
    # An explicit abstention with empty citations is allowed — completeness
    # only binds committed verdicts.
    client = MockClient(responses=[_resp(verdict="cannot_determine", cols=[])])
    out = investigate_leakage_bound(EVIDENCE, client=client)
    assert out["schema_rejections"] == 0
    assert out["omitted_critical_evidence"] is False
    assert client.create_call_count == 1


def test_claim_column_counts_as_addressing_anchor() -> None:
    # Addressing the top candidate via a claim (not columns_referenced) is
    # sufficient for completeness.
    client = MockClient(
        responses=[
            _resp(
                cols=["near_target_proxy"],
                claims=[{"column": "log_target_v2", "statistic": "correlation", "value": 1.0}],
            )
        ]
    )
    out = investigate_leakage_bound(EVIDENCE, client=client)
    assert out["schema_rejections"] == 0
    assert out["omitted_critical_evidence"] is False


def test_claims_enum_present_in_both_tool_formats() -> None:
    from mlcompass.agents.leakage_investigator import build_submit_investigation_tool_openai

    ant = build_submit_investigation_tool(ALLOWED)
    oai = build_submit_investigation_tool_openai(ALLOWED)
    ant_enum = ant["input_schema"]["properties"]["claims"]["items"]["properties"]["column"]["enum"]
    oai_enum = oai["function"]["parameters"]["properties"]["claims"]["items"]["properties"][
        "column"
    ]["enum"]
    assert ant_enum == ALLOWED
    assert oai_enum == ALLOWED
