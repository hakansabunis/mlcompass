"""CLI integration tests for ``mlcompass agent``.

The orchestrator is monkeypatched via ``cli._agent_runner`` so we
don't need a live model or the claude-agent-sdk to verify the CLI's
contract: argument parsing, exit codes, paths recorded in the summary
panel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

import mlcompass.cli as cli_module
from mlcompass.agent.backends import AgentResult
from mlcompass.agent.orchestrator import AgentRunSummary
from mlcompass.cli import cli


def _summary_factory(
    *,
    ok: bool = True,
    turns: int = 2,
    stop_reason: str = "end_turn",
    project_path: Path,
) -> AgentRunSummary:
    run_dir = project_path / ".mlcompass" / "agent_runs" / "demo"
    run_dir.mkdir(parents=True, exist_ok=True)
    return AgentRunSummary(
        result=AgentResult(
            ok=ok,
            turns=turns,
            final_text="Done." if ok else "",
            stop_reason=stop_reason,
            error=None if ok else "boom",
        ),
        run_dir=run_dir,
        transcript_path=run_dir / "transcript.jsonl",
        summary_path=run_dir / "summary.md",
    )


@pytest.fixture
def stub_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def _stub(**kwargs: Any) -> AgentRunSummary:
        calls.append(kwargs)
        return _summary_factory(project_path=tmp_path)

    monkeypatch.setattr(cli_module, "_agent_runner", _stub)
    return calls


def test_agent_default_backend_is_api(stub_runner: list[dict[str, Any]], tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["agent", "Recommend a model", "--project-path", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert len(stub_runner) == 1
    assert stub_runner[0]["backend"] == "api"
    assert stub_runner[0]["task"] == "Recommend a model"
    assert "Agent run" in result.output
    assert "transcript" in result.output.lower()


def test_agent_claude_code_backend_dispatched(
    stub_runner: list[dict[str, Any]], tmp_path: Path
) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "agent",
            "Check this data",
            "--backend",
            "claude-code",
            "--project-path",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert stub_runner[0]["backend"] == "claude-code"


def test_agent_auto_approve_flag_propagates(
    stub_runner: list[dict[str, Any]], tmp_path: Path
) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "agent",
            "Init a project",
            "--project-path",
            str(tmp_path),
            "--auto-approve",
        ],
    )
    assert result.exit_code == 0, result.output
    assert stub_runner[0]["auto_approve_mutations"] is True


def test_agent_max_turns_passes_through(stub_runner: list[dict[str, Any]], tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "agent",
            "Run forever",
            "--project-path",
            str(tmp_path),
            "--max-turns",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    assert stub_runner[0]["max_turns"] == 3


def test_agent_nonzero_exit_when_backend_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _stub(**_: Any) -> AgentRunSummary:
        return _summary_factory(ok=False, stop_reason="max_turns", project_path=tmp_path)

    monkeypatch.setattr(cli_module, "_agent_runner", _stub)

    runner = CliRunner()
    result = runner.invoke(cli, ["agent", "loop", "--project-path", str(tmp_path)])
    # Exit code 1 on failure, but the summary panel should still print.
    assert result.exit_code == 1
    assert "max_turns" in result.output


def test_root_help_lists_agent() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "agent" in result.output
    assert "self-driving" in result.output.lower()


def test_agent_help_lists_all_options() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["agent", "--help"])
    assert result.exit_code == 0
    for needle in (
        "--backend",
        "--project-path",
        "--max-turns",
        "--auto-approve",
        "--model",
    ):
        assert needle in result.output
