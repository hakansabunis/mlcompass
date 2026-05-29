"""Tests for ``mlcompass advise`` CLI command."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from click.testing import CliRunner

from mlcompass import cli as cli_module
from mlcompass.agents.advise import AdvisorParseError
from mlcompass.cli import cli
from mlcompass.context import DEFAULT_PROJECT_DIR, ProjectContext


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """A small clean CSV with a clear target column."""
    df = pd.DataFrame(
        {
            "age": list(range(20, 120)),
            "income": [i * 1000 for i in range(100)],
            "churn": [i % 2 for i in range(100)],
        }
    )
    p = tmp_path / "customers.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def fake_recommendation() -> dict[str, Any]:
    """A canned advisor response used to bypass real LLM calls."""
    return {
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
                "suggestion": "bin into quartiles",
                "reason": "small numeric range",
            }
        ],
        "pitfalls": [
            {
                "issue": "Small dataset (100 rows)",
                "mitigation": "Use cross-validation, prefer simpler models",
            }
        ],
    }


@pytest.fixture
def fake_advisor(
    monkeypatch: pytest.MonkeyPatch,
    fake_recommendation: dict[str, Any],
) -> dict[str, Any]:
    """Replace cli._advisor_callable so tests don't hit a real API."""

    captured: dict[str, Any] = {"called": False, "analysis": None, "model": None}

    def _fake(analysis: dict[str, Any], *, model: str = "claude-opus-4-7") -> dict[str, Any]:
        captured["called"] = True
        captured["analysis"] = analysis
        captured["model"] = model
        return fake_recommendation

    monkeypatch.setattr(cli_module, "_advisor_callable", _fake)
    return captured


@pytest.fixture
def with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")


@pytest.fixture
def without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


# --------------------------------------------------------------------------- #
# Standalone (no project context)                                             #
# --------------------------------------------------------------------------- #


def test_advise_runs_standalone_without_project(
    sample_csv: Path,
    fake_advisor: dict[str, Any],
    with_api_key: None,
    tmp_path: Path,
) -> None:
    """advise should work without a .mlcompass/ project; it just warns."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["advise", str(sample_csv)])

    assert result.exit_code == 0, result.output
    assert "no .mlcompass" in result.output.lower() or "standalone" in result.output.lower()
    assert fake_advisor["called"]
    assert "XGBoost" in result.output


def test_advise_skips_llm_when_api_key_missing(
    sample_csv: Path,
    without_api_key: None,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["advise", str(sample_csv)])

    assert result.exit_code == 0
    assert "anthropic_api_key" in result.output.lower()


def test_advise_skips_llm_with_no_llm_flag(
    sample_csv: Path,
    with_api_key: None,
    fake_advisor: dict[str, Any],
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["advise", str(sample_csv), "--no-llm"])

    assert result.exit_code == 0
    assert not fake_advisor["called"], "advisor should be skipped when --no-llm"
    assert "skipping advisor" in result.output.lower()


# --------------------------------------------------------------------------- #
# With project context                                                        #
# --------------------------------------------------------------------------- #


def test_advise_persists_to_project_context(
    sample_csv: Path,
    fake_advisor: dict[str, Any],
    with_api_key: None,
    tmp_path: Path,
) -> None:
    # Initialise project where the CSV lives so .mlcompass/ is discoverable
    project_root = sample_csv.parent
    project = ProjectContext.init("test-proj", parent_dir=project_root)

    runner = CliRunner()
    # Use Click's working_dir to mimic invoking mlcompass from project_root
    cwd = os.getcwd()
    os.chdir(project_root)
    try:
        result = runner.invoke(cli, ["advise", str(sample_csv)])
    finally:
        os.chdir(cwd)

    assert result.exit_code == 0, result.output

    state = project.read_context()
    assert state["active_dataset"] is not None
    assert state["active_dataset"].startswith("datasets/")
    assert state["project_type"] == "binary_classification"
    assert state["target_column"] == "churn"

    # One decision recorded
    assert len(state["decisions"]) == 1
    decision = state["decisions"][0]
    assert decision["command"] == "advise"
    assert "XGBoost" in decision["summary"]

    # advice.log should contain one JSON line
    log_path = project.path / "advice.log"
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    log_entry = json.loads(lines[0])
    assert log_entry["dataset"] == str(sample_csv)
    assert log_entry["task_type"] == "binary_classification"
    assert log_entry["recommendation"]["models"][0]["name"] == "XGBoost"


def test_advise_passes_target_override_to_analysis(
    sample_csv: Path,
    fake_advisor: dict[str, Any],
    with_api_key: None,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            cli,
            ["advise", str(sample_csv), "--target", "income"],
        )

    assert result.exit_code == 0
    # The advisor should have received the analysis with the override applied
    analysis = fake_advisor["analysis"]
    assert analysis["target_hint"]["column"] == "income"
    assert analysis["target_hint"]["confidence"] == "explicit"


def test_advise_passes_sample_rows(
    sample_csv: Path,
    fake_advisor: dict[str, Any],
    with_api_key: None,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            cli,
            ["advise", str(sample_csv), "--sample-rows", "10"],
        )

    assert result.exit_code == 0
    analysis = fake_advisor["analysis"]
    assert analysis["shape"]["rows"] == 10


def test_advise_passes_advisor_model(
    sample_csv: Path,
    fake_advisor: dict[str, Any],
    with_api_key: None,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            cli,
            ["advise", str(sample_csv), "--model", "claude-haiku-4-5"],
        )

    assert result.exit_code == 0
    assert fake_advisor["model"] == "claude-haiku-4-5"


# --------------------------------------------------------------------------- #
# Error handling                                                              #
# --------------------------------------------------------------------------- #


def test_advise_handles_advisor_parse_error_gracefully(
    sample_csv: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_api_key: None,
    tmp_path: Path,
) -> None:
    def _broken_advisor(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AdvisorParseError("Advisor returned garbage")

    monkeypatch.setattr(cli_module, "_advisor_callable", _broken_advisor)

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["advise", str(sample_csv)])

    # Should exit zero (analysis still rendered) but report the error
    assert result.exit_code == 0
    assert "invalid response" in result.output.lower() or "garbage" in result.output.lower()


def test_advise_fails_when_dataset_missing(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["advise", str(tmp_path / "nope.csv")])
    assert result.exit_code != 0


# --------------------------------------------------------------------------- #
# Help text                                                                   #
# --------------------------------------------------------------------------- #


def test_advise_help_shows_options() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["advise", "--help"])
    assert result.exit_code == 0
    assert "--target" in result.output
    assert "--no-llm" in result.output
    assert "--sample-rows" in result.output


def test_root_help_lists_advise() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "advise" in result.output
