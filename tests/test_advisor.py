"""Tests for the model + feature engineering advisor."""

from __future__ import annotations

import json

import pytest
from agentlite.testing import MockClient, text_response

from ml_copilot.agents.advise import (
    AdvisorParseError,
    _parse_advisor_response,
    build_advisor_agent,
    get_recommendation,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

SAMPLE_ANALYSIS: dict = {
    "path": "data.csv",
    "format": "csv",
    "shape": {"rows": 10000, "cols": 5},
    "columns": [
        {
            "name": "age",
            "type": "numeric",
            "stats": {"mean": 35.4},
            "outliers": {"iqr_count": 23},
        },
        {
            "name": "churn",
            "type": "categorical",
            "cardinality": 2,
        },
    ],
    "target_hint": {"column": "churn", "confidence": "high"},
    "task_hint": {
        "type": "binary_classification",
        "class_balance": {"0": 0.88, "1": 0.12},
    },
    "warnings": ["Class imbalance detected (12% minority class)."],
}

VALID_REC_JSON = json.dumps(
    {
        "models": [
            {
                "name": "XGBoost",
                "reason": "Strong tabular baseline",
                "expected_metric": "AUC 0.82 - 0.87",
            },
            {
                "name": "Logistic Regression",
                "reason": "Interpretable baseline",
                "expected_metric": "AUC 0.76 - 0.80",
            },
        ],
        "features": [
            {
                "column": "age",
                "suggestion": "winsorize at 99th percentile",
                "reason": "23 IQR outliers reported",
            }
        ],
        "pitfalls": [
            {
                "issue": "Class imbalance (12% positive)",
                "mitigation": "Use AUC/F1, class_weight='balanced', or focal loss",
            }
        ],
    }
)


# --------------------------------------------------------------------------- #
# build_advisor_agent                                                         #
# --------------------------------------------------------------------------- #


def test_build_advisor_agent_has_no_tools() -> None:
    """The advisor should not need any tools — analysis is pre-computed."""
    client = MockClient(responses=[text_response(VALID_REC_JSON)])
    agent = build_advisor_agent(client=client)
    assert agent.tools == [] or len(agent.tools) == 0


def test_build_advisor_agent_uses_opus_by_default() -> None:
    client = MockClient(responses=[text_response(VALID_REC_JSON)])
    agent = build_advisor_agent(client=client)
    assert "opus" in agent.model.lower()


def test_build_advisor_agent_respects_custom_model() -> None:
    client = MockClient(responses=[text_response(VALID_REC_JSON)])
    agent = build_advisor_agent(client=client, model="claude-haiku-4-5")
    assert agent.model == "claude-haiku-4-5"


# --------------------------------------------------------------------------- #
# get_recommendation (full path)                                              #
# --------------------------------------------------------------------------- #


def test_get_recommendation_returns_parsed_dict() -> None:
    client = MockClient(responses=[text_response(VALID_REC_JSON)])
    result = get_recommendation(SAMPLE_ANALYSIS, client=client)

    assert "models" in result
    assert "features" in result
    assert "pitfalls" in result
    assert len(result["models"]) == 2
    assert result["models"][0]["name"] == "XGBoost"


def test_get_recommendation_sends_analysis_to_agent() -> None:
    client = MockClient(responses=[text_response(VALID_REC_JSON)])
    get_recommendation(SAMPLE_ANALYSIS, client=client)

    # The user message should embed the analysis JSON
    last_messages = client.last_create_kwargs.get("messages", [])
    assert len(last_messages) >= 1
    user_text = json.dumps(last_messages[-1])
    assert "churn" in user_text  # column name should be in the prompt
    assert "binary_classification" in user_text


def test_get_recommendation_raises_on_invalid_json() -> None:
    client = MockClient(responses=[text_response("not actually json")])
    with pytest.raises(AdvisorParseError):
        get_recommendation(SAMPLE_ANALYSIS, client=client)


def test_get_recommendation_raises_when_missing_required_keys() -> None:
    # Valid JSON but missing 'pitfalls'
    bad_response = json.dumps({"models": [], "features": []})
    client = MockClient(responses=[text_response(bad_response)])
    with pytest.raises(AdvisorParseError, match="pitfalls"):
        get_recommendation(SAMPLE_ANALYSIS, client=client)


# --------------------------------------------------------------------------- #
# Parser-only tests                                                           #
# --------------------------------------------------------------------------- #


def test_parser_handles_clean_json() -> None:
    result = _parse_advisor_response(VALID_REC_JSON)
    assert "models" in result


def test_parser_strips_markdown_code_fence() -> None:
    wrapped = f"```json\n{VALID_REC_JSON}\n```"
    result = _parse_advisor_response(wrapped)
    assert "models" in result


def test_parser_strips_plain_fence() -> None:
    wrapped = f"```\n{VALID_REC_JSON}\n```"
    result = _parse_advisor_response(wrapped)
    assert "models" in result


def test_parser_tolerates_leading_trailing_whitespace() -> None:
    padded = f"\n\n   {VALID_REC_JSON}   \n\n"
    result = _parse_advisor_response(padded)
    assert "models" in result


def test_parser_raises_on_non_json_text() -> None:
    with pytest.raises(AdvisorParseError):
        _parse_advisor_response("Just plain prose.")


def test_parser_raises_when_response_is_array_not_object() -> None:
    with pytest.raises(AdvisorParseError, match="not an object"):
        _parse_advisor_response("[1, 2, 3]")


def test_parser_raises_when_missing_models_key() -> None:
    response = json.dumps({"features": [], "pitfalls": []})
    with pytest.raises(AdvisorParseError, match="models"):
        _parse_advisor_response(response)
