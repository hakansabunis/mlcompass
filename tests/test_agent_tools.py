"""Tests for the shared agent tool registry."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlcompass.agent.tools import (
    BY_NAME,
    TOOL_REGISTRY,
    anthropic_tool_specs,
    call_tool,
)


def test_eight_tools_registered() -> None:
    assert len(TOOL_REGISTRY) == 8
    names = {spec.name for spec in TOOL_REGISTRY}
    assert names == {
        "mlcompass_init",
        "mlcompass_status",
        "mlcompass_advise",
        "mlcompass_audit",
        "mlcompass_watch",
        "mlcompass_compare",
        "mlcompass_evaluate",
        "mlcompass_deploy",
    }


def test_only_init_mutates() -> None:
    mutating = {spec.name for spec in TOOL_REGISTRY if spec.mutates}
    assert mutating == {"mlcompass_init"}


def test_each_spec_has_schema_with_object_type() -> None:
    for spec in TOOL_REGISTRY:
        schema = spec.input_schema
        assert schema["type"] == "object"
        assert "properties" in schema


def test_anthropic_tool_specs_strips_dispatcher() -> None:
    """The Anthropic API payload must NOT carry our private dispatcher."""
    specs = anthropic_tool_specs()
    assert len(specs) == 8
    for entry in specs:
        assert set(entry.keys()) == {"name", "description", "input_schema"}


def test_by_name_lookup_matches_registry() -> None:
    for spec in TOOL_REGISTRY:
        assert BY_NAME[spec.name] is spec


def test_call_tool_dispatches_successful_advise(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    pd.DataFrame({"x": [1, 2, 3], "churn": [0, 1, 0]}).to_csv(csv, index=False)

    result = call_tool("mlcompass_advise", {"dataset_path": str(csv)})
    assert result["shape"]["rows"] == 3
    assert result["target_hint"]["column"] == "churn"


def test_call_tool_unknown_returns_error_envelope() -> None:
    result = call_tool("mlcompass_unknown", {})
    assert result["ok"] is False
    assert result["error"] == "UnknownTool"
    assert "mlcompass_advise" in result["message"]


def test_call_tool_bad_arguments_returns_error_envelope() -> None:
    result = call_tool("mlcompass_advise", {"wrong_kwarg": 1})
    assert result["ok"] is False
    assert result["error"] == "BadArguments"
