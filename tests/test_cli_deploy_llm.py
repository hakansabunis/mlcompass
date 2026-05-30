"""CLI tests for ``mlcompass deploy --llm``."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from mlcompass import cli as cli_module
from mlcompass.agents.deploy import DeployAgentError
from mlcompass.cli import cli
from mlcompass.context import ProjectContext


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def pytorch_model(tmp_path: Path) -> Path:
    p = tmp_path / "model.pt"
    p.write_bytes(b"PK\x03\x04" + b"\x00" * 200)
    return p


@pytest.fixture
def with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")


@pytest.fixture
def without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture
def fake_advice() -> dict[str, Any]:
    return {
        "verdict": "Ready for canary deploy.",
        "blockers": [],
        "next_steps": ["Set up monitoring."],
        "rollout_strategy": "Canary 5% then 25% then 100%.",
    }


# --------------------------------------------------------------------------- #
# Happy path                                                                  #
# --------------------------------------------------------------------------- #


def test_deploy_llm_calls_advisor(
    pytorch_model: Path,
    tmp_path: Path,
    with_api_key: None,
    fake_advice: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {"called": False, "target": None}

    def fake(report: dict[str, Any], *, model: str = "claude-opus-4-7") -> dict[str, Any]:
        captured["called"] = True
        captured["target"] = report.get("target")
        return fake_advice

    monkeypatch.setattr(cli_module, "_deploy_advisor_callable", fake)

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["deploy", str(pytorch_model), "--llm"])

    assert result.exit_code == 0, result.output
    assert captured["called"]
    assert captured["target"] == "local"
    assert "ready for canary deploy" in result.output.lower()


def test_deploy_llm_skips_when_no_api_key(
    pytorch_model: Path,
    tmp_path: Path,
    without_api_key: None,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["deploy", str(pytorch_model), "--llm"])
    assert result.exit_code == 0
    assert "anthropic_api_key" in result.output.lower()


def test_deploy_llm_handles_agent_error(
    pytorch_model: Path,
    tmp_path: Path,
    with_api_key: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*_a: Any, **_kw: Any) -> dict[str, Any]:
        raise DeployAgentError("bad json")

    monkeypatch.setattr(cli_module, "_deploy_advisor_callable", broken)

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["deploy", str(pytorch_model), "--llm"])
    assert result.exit_code == 0
    assert "bad response" in result.output.lower() or "bad json" in result.output.lower()


def test_deploy_llm_persists_advice_to_project(
    pytorch_model: Path,
    tmp_path: Path,
    with_api_key: None,
    fake_advice: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = ProjectContext.init("test-proj", parent_dir=tmp_path)

    monkeypatch.setattr(
        cli_module,
        "_deploy_advisor_callable",
        lambda report, *, model="claude-opus-4-7": fake_advice,
    )

    runner = CliRunner()
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(cli, ["deploy", str(pytorch_model), "--llm"])
    finally:
        os.chdir(cwd)

    assert result.exit_code == 0, result.output

    log_path = project.path / "advice.log"
    entries = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert entries
    entry = entries[0]
    assert "advice" in entry
    assert entry["advice"]["rollout_strategy"].startswith("Canary")
