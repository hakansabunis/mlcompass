"""Unit tests for the evaluate LLM interpreter (``agents.evaluate``)."""

from __future__ import annotations

import json

import pytest
from agentlite.testing import MockClient, text_response

from mlcompass.agents.evaluate import (
    EvaluateAgentError,
    build_evaluate_agent,
    interpret_evaluation,
)

SAMPLE_EVALUATION: dict = {
    "task": "binary_classification",
    "rows": 200,
    "columns": {"y_true": "y_true", "y_pred": "y_pred", "y_prob": "y_prob"},
    "metrics": {"accuracy": 0.93, "precision": 0.78, "recall": 0.92, "f1": 0.85, "auc": 0.96},
    "confusion_matrix": {"tp": 92, "fp": 26, "fn": 8, "tn": 74},
    "threshold_sweep": [
        {"threshold": 0.5, "precision": 0.78, "recall": 0.92, "f1": 0.84},
        {"threshold": 0.65, "precision": 0.91, "recall": 0.86, "f1": 0.88},
    ],
    "best_threshold": {"threshold": 0.65, "precision": 0.91, "recall": 0.86, "f1": 0.88},
    "hard_examples": [],
    "warnings": [],
}


VALID_INTERPRETATION_JSON = json.dumps(
    {
        "assessment": "Strong overall performance.",
        "strengths": ["AUC 0.96 is excellent."],
        "weaknesses": ["Precision lags recall at threshold 0.5."],
        "next_steps": ["Ship threshold 0.65 instead."],
    }
)


def test_build_evaluate_agent_has_no_tools() -> None:
    client = MockClient(responses=[text_response(VALID_INTERPRETATION_JSON)])
    agent = build_evaluate_agent(client=client)
    assert agent.tools == [] or len(agent.tools) == 0


def test_interpret_evaluation_returns_parsed_dict() -> None:
    client = MockClient(responses=[text_response(VALID_INTERPRETATION_JSON)])
    out = interpret_evaluation(SAMPLE_EVALUATION, client=client)
    for key in ("assessment", "strengths", "weaknesses", "next_steps"):
        assert key in out
    assert out["strengths"][0].startswith("AUC")


def test_interpret_evaluation_sends_evaluation_payload() -> None:
    client = MockClient(responses=[text_response(VALID_INTERPRETATION_JSON)])
    interpret_evaluation(SAMPLE_EVALUATION, client=client)
    messages = client.last_create_kwargs.get("messages", [])
    serialized = json.dumps(messages[-1])
    assert "binary_classification" in serialized
    assert "auc" in serialized.lower()


def test_interpret_evaluation_raises_on_invalid_json() -> None:
    client = MockClient(responses=[text_response("not really json")])
    with pytest.raises(EvaluateAgentError):
        interpret_evaluation(SAMPLE_EVALUATION, client=client)


def test_interpret_evaluation_raises_when_missing_required_keys() -> None:
    bad = json.dumps({"assessment": "x", "strengths": [], "weaknesses": []})
    client = MockClient(responses=[text_response(bad)])
    with pytest.raises(EvaluateAgentError, match="next_steps"):
        interpret_evaluation(SAMPLE_EVALUATION, client=client)
