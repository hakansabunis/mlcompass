"""Tests for the shared agent tool registry."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

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


def test_non_mutating_tools_do_not_touch_a_project_they_merely_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``mutates=False`` has to mean something on disk, not just in the dataclass.

    The declared flag is what the permission gate reads, so a tool that
    claims to be read-only is never prompted for. That claim is only
    honest if naming a path inside *someone else's* project leaves that
    project byte-identical. (Each tool does still append to the agent's
    own ledger — that is the audit trail, and it is tested in
    ``tests/test_agent_ledger_boundary.py``.)
    """
    from mlcompass.context import ProjectContext

    active = tmp_path / "active"
    bystander = tmp_path / "bystander"
    active.mkdir()
    bystander.mkdir()
    ProjectContext.init("active", parent_dir=active)
    ProjectContext.init("bystander", parent_dir=bystander)
    monkeypatch.chdir(active)

    csv = bystander / "data.csv"
    pd.DataFrame({"age": list(range(20)), "churn": [0, 1] * 10}).to_csv(csv, index=False)
    script = bystander / "train.py"
    script.write_text("import torch\nm = torch.nn.Linear(1, 1)\n", encoding="utf-8")

    def _snapshot() -> dict[str, bytes]:
        root = bystander / ".mlcompass"
        return {
            str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
        }

    before = _snapshot()
    call_tool("mlcompass_advise", {"dataset_path": str(csv)})
    call_tool("mlcompass_audit", {"script_path": str(script)})
    after = _snapshot()

    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    assert changed == [], f"non-mutating tools wrote into {bystander}: {changed}"


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
