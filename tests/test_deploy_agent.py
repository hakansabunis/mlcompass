"""Unit tests for the deploy LLM advisor (``agents.deploy``)."""

from __future__ import annotations

import json

import pytest
from agentlite.testing import MockClient, text_response

from mlcompass.agents.deploy import (
    DeployAgentError,
    advise_deployment,
    build_deploy_agent,
)

SAMPLE_REPORT: dict = {
    "model": {"format": "pytorch", "size_class": "small", "size_pretty": "195 KB"},
    "dependencies": {
        "manifest": "requirements.txt",
        "pinned": ["torch==2.1.0"],
        "unpinned": ["requests"],
        "ml_packages": ["torch"],
    },
    "target": "lambda",
    "checklist": [
        {"item": "Model file format identified", "status": "ok", "detail": "pytorch"},
    ],
    "warnings": ["1 dependency(ies) are unpinned."],
}


VALID_ADVICE_JSON = json.dumps(
    {
        "verdict": "Ready for canary on Lambda after one fix.",
        "blockers": ["Pin requests before shipping."],
        "next_steps": ["Lock all deps", "Measure cold start"],
        "rollout_strategy": "Canary 1% -> 10% -> 50% -> 100%.",
    }
)


def test_build_deploy_agent_has_no_tools() -> None:
    client = MockClient(responses=[text_response(VALID_ADVICE_JSON)])
    agent = build_deploy_agent(client=client)
    assert agent.tools == [] or len(agent.tools) == 0


def test_advise_deployment_returns_parsed_dict() -> None:
    client = MockClient(responses=[text_response(VALID_ADVICE_JSON)])
    out = advise_deployment(SAMPLE_REPORT, client=client)
    for key in ("verdict", "blockers", "next_steps", "rollout_strategy"):
        assert key in out
    assert out["blockers"][0].startswith("Pin")


def test_advise_deployment_payload_includes_report_fields() -> None:
    client = MockClient(responses=[text_response(VALID_ADVICE_JSON)])
    advise_deployment(SAMPLE_REPORT, client=client)
    messages = client.last_create_kwargs.get("messages", [])
    serialized = json.dumps(messages[-1])
    assert "lambda" in serialized.lower()
    assert "pytorch" in serialized.lower()


def test_advise_deployment_raises_on_invalid_json() -> None:
    client = MockClient(responses=[text_response("not json")])
    with pytest.raises(DeployAgentError):
        advise_deployment(SAMPLE_REPORT, client=client)


def test_advise_deployment_raises_when_missing_required_key() -> None:
    bad = json.dumps({"verdict": "x", "blockers": [], "next_steps": []})
    client = MockClient(responses=[text_response(bad)])
    with pytest.raises(DeployAgentError, match="rollout_strategy"):
        advise_deployment(SAMPLE_REPORT, client=client)
