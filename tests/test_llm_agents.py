"""Unit tests for the Faz 2.1 optional LLM layers.

Covers the three agents (``audit``, ``watch``, ``compare``) plus the
shared ``_common.parse_json_response`` helper. All tests use
``agentlite.testing.MockClient`` so no real API call is made.
"""

from __future__ import annotations

import json

import pytest
from agentlite.testing import MockClient, text_response

from mlcompass.agents._common import AgentResponseError, parse_json_response
from mlcompass.agents.audit import (
    AuditAgentError,
    build_audit_agent,
    prioritize_findings,
)
from mlcompass.agents.compare import (
    CompareAgentError,
    build_compare_agent,
    hypothesize_comparison,
)
from mlcompass.agents.watch import (
    WatchAgentError,
    build_watch_agent,
    diagnose_findings,
)


# --------------------------------------------------------------------------- #
# Shared parser                                                               #
# --------------------------------------------------------------------------- #


def test_parser_clean_object() -> None:
    out = parse_json_response('{"a": 1, "b": 2}', required_keys=("a", "b"))
    assert out == {"a": 1, "b": 2}


def test_parser_strips_markdown_code_fence() -> None:
    text = '```json\n{"a": 1, "b": 2}\n```'
    out = parse_json_response(text, required_keys=("a",))
    assert out["a"] == 1


def test_parser_raises_on_invalid_json() -> None:
    with pytest.raises(AgentResponseError):
        parse_json_response("not json", required_keys=())


def test_parser_raises_on_non_object() -> None:
    with pytest.raises(AgentResponseError, match="not an object"):
        parse_json_response("[1, 2, 3]", required_keys=())


def test_parser_raises_on_missing_required_key() -> None:
    with pytest.raises(AgentResponseError, match="b"):
        parse_json_response('{"a": 1}', required_keys=("a", "b"))


def test_parser_uses_custom_error_class() -> None:
    class MyError(AgentResponseError):
        pass

    with pytest.raises(MyError):
        parse_json_response("not json", error_class=MyError)


# --------------------------------------------------------------------------- #
# Fixtures: audit                                                             #
# --------------------------------------------------------------------------- #


SAMPLE_AUDIT_RESULT: dict = {
    "path": "train.py",
    "lines": 50,
    "frameworks": ["torch"],
    "findings": [
        {
            "rule_id": "seed",
            "severity": "error",
            "message": "No random seed is set anywhere in the script.",
            "suggestion": "Add torch.manual_seed(42).",
            "line": None,
        },
        {
            "rule_id": "optimizer",
            "severity": "error",
            "message": "Adam does not accept a momentum= argument.",
            "suggestion": "Remove momentum= or switch to SGD.",
            "line": 17,
        },
    ],
}


VALID_AUDIT_PRIORITIES_JSON = json.dumps(
    {
        "priorities": [
            {
                "rule_id": "seed",
                "priority_rank": 1,
                "blast_radius": "Results cannot be reproduced.",
            },
            {
                "rule_id": "optimizer",
                "priority_rank": 2,
                "blast_radius": "Momentum kwarg silently ignored.",
            },
        ],
        "synthesis": "Fix the seed first, then the Adam momentum kwarg.",
    }
)


# --------------------------------------------------------------------------- #
# audit prioritizer                                                           #
# --------------------------------------------------------------------------- #


def test_build_audit_agent_has_no_tools() -> None:
    client = MockClient(responses=[text_response(VALID_AUDIT_PRIORITIES_JSON)])
    agent = build_audit_agent(client=client)
    assert agent.tools == [] or len(agent.tools) == 0


def test_prioritize_findings_parses_response() -> None:
    client = MockClient(responses=[text_response(VALID_AUDIT_PRIORITIES_JSON)])
    out = prioritize_findings(SAMPLE_AUDIT_RESULT, client=client)
    assert len(out["priorities"]) == 2
    assert out["priorities"][0]["rule_id"] == "seed"
    assert "synthesis" in out


def test_prioritize_findings_short_circuits_on_empty() -> None:
    # Should not invoke the agent at all when there are no findings.
    client = MockClient(responses=[])
    out = prioritize_findings({"findings": []}, client=client)
    assert out["priorities"] == []
    assert client.create_call_count == 0


def test_prioritize_findings_raises_on_bad_json() -> None:
    client = MockClient(responses=[text_response("nonsense")])
    with pytest.raises(AuditAgentError):
        prioritize_findings(SAMPLE_AUDIT_RESULT, client=client)


def test_prioritize_findings_raises_when_missing_priorities() -> None:
    bad = json.dumps({"synthesis": "x"})
    client = MockClient(responses=[text_response(bad)])
    with pytest.raises(AuditAgentError, match="priorities"):
        prioritize_findings(SAMPLE_AUDIT_RESULT, client=client)


# --------------------------------------------------------------------------- #
# watch diagnostician                                                         #
# --------------------------------------------------------------------------- #


SAMPLE_SNAPSHOTS: list[dict] = [
    {"epoch": 0, "step": None, "metrics": {"train_loss": 0.5, "val_loss": 0.5}},
    {"epoch": 5, "step": None, "metrics": {"train_loss": 0.1, "val_loss": 0.55}},
]

SAMPLE_WATCH_FINDINGS: list[dict] = [
    {
        "rule_id": "overfitting",
        "severity": "warning",
        "message": "train down, val up.",
        "suggestion": "regularize.",
        "epoch": 5,
    }
]

VALID_DIAGNOSIS_JSON = json.dumps(
    {
        "diagnosis": [
            {
                "finding_rule_id": "overfitting",
                "hypothesis": "Capacity too high for dataset.",
                "recommended_action": "Increase dropout to 0.3.",
                "confidence": "high",
            }
        ],
        "summary": "Stop training, regularize, restart.",
    }
)


def test_build_watch_agent_has_no_tools() -> None:
    client = MockClient(responses=[text_response(VALID_DIAGNOSIS_JSON)])
    agent = build_watch_agent(client=client)
    assert agent.tools == [] or len(agent.tools) == 0


def test_diagnose_findings_parses_response() -> None:
    client = MockClient(responses=[text_response(VALID_DIAGNOSIS_JSON)])
    out = diagnose_findings(SAMPLE_SNAPSHOTS, SAMPLE_WATCH_FINDINGS, client=client)
    assert out["diagnosis"][0]["finding_rule_id"] == "overfitting"
    assert out["summary"]


def test_diagnose_findings_short_circuits_on_empty() -> None:
    client = MockClient(responses=[])
    out = diagnose_findings(SAMPLE_SNAPSHOTS, [], client=client)
    assert out["diagnosis"] == []
    assert client.create_call_count == 0


def test_diagnose_findings_raises_on_bad_json() -> None:
    client = MockClient(responses=[text_response("nope")])
    with pytest.raises(WatchAgentError):
        diagnose_findings(SAMPLE_SNAPSHOTS, SAMPLE_WATCH_FINDINGS, client=client)


def test_diagnose_findings_raises_when_missing_summary() -> None:
    bad = json.dumps({"diagnosis": []})
    client = MockClient(responses=[text_response(bad)])
    with pytest.raises(WatchAgentError, match="summary"):
        diagnose_findings(SAMPLE_SNAPSHOTS, SAMPLE_WATCH_FINDINGS, client=client)


# --------------------------------------------------------------------------- #
# compare hypothesizer                                                        #
# --------------------------------------------------------------------------- #


SAMPLE_COMPARISON: dict = {
    "run_a": {"id": "run-3", "name": "baseline", "config": {"lr": 1e-3}},
    "run_b": {"id": "run-7", "name": "lower-lr", "config": {"lr": 3e-4}},
    "config_diff": [{"key": "lr", "a_value": 1e-3, "b_value": 3e-4}],
    "metric_comparison": [
        {
            "name": "val_loss",
            "a_value": 0.5,
            "b_value": 0.3,
            "delta": -0.2,
            "better": "b",
        }
    ],
    "verdict": "b_better",
    "verdict_explanation": "Run B wins on val_loss.",
}

VALID_HYPOTHESIS_JSON = json.dumps(
    {
        "hypothesis": "Run B's lower lr stabilised late training.",
        "key_factors": [
            {
                "config_key": "lr",
                "impact": "high",
                "reason": "Lower lr is the only changed knob.",
            }
        ],
        "next_experiment": "Try lr=1e-4 on the same architecture.",
    }
)


def test_build_compare_agent_has_no_tools() -> None:
    client = MockClient(responses=[text_response(VALID_HYPOTHESIS_JSON)])
    agent = build_compare_agent(client=client)
    assert agent.tools == [] or len(agent.tools) == 0


def test_hypothesize_comparison_parses_response() -> None:
    client = MockClient(responses=[text_response(VALID_HYPOTHESIS_JSON)])
    out = hypothesize_comparison(SAMPLE_COMPARISON, client=client)
    assert "hypothesis" in out
    assert out["key_factors"][0]["config_key"] == "lr"
    assert out["next_experiment"]


def test_hypothesize_comparison_raises_on_bad_json() -> None:
    client = MockClient(responses=[text_response("garbage")])
    with pytest.raises(CompareAgentError):
        hypothesize_comparison(SAMPLE_COMPARISON, client=client)


def test_hypothesize_comparison_raises_when_missing_key() -> None:
    bad = json.dumps({"hypothesis": "x", "key_factors": []})
    client = MockClient(responses=[text_response(bad)])
    with pytest.raises(CompareAgentError, match="next_experiment"):
        hypothesize_comparison(SAMPLE_COMPARISON, client=client)


def test_hypothesize_comparison_sends_comparison_payload() -> None:
    client = MockClient(responses=[text_response(VALID_HYPOTHESIS_JSON)])
    hypothesize_comparison(SAMPLE_COMPARISON, client=client)
    messages = client.last_create_kwargs.get("messages", [])
    serialised = json.dumps(messages[-1])
    # The user message should contain the verdict the analyzer assigned.
    assert "b_better" in serialised
