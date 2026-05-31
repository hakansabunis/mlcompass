"""CLI integration tests for ``mlcompass status``."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner

from mlcompass.cli import cli
from mlcompass.context import ProjectContext

# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def project(tmp_path: Path) -> ProjectContext:
    return ProjectContext.init("demo-project", parent_dir=tmp_path)


def _populate_state(project: ProjectContext) -> None:
    project.write_context(
        {
            "project_type": "binary_classification",
            "target_column": "churn",
            "active_dataset": "datasets/abc123.json",
            "preferred_models": ["XGBoost", "LightGBM"],
        }
    )
    project.append_decision(
        command="advise",
        summary="Recommended XGBoost as baseline",
        reasoning="500-row binary classification",
    )
    project.append_decision(
        command="audit",
        summary="Audit: 2 error, 4 warning",
        reasoning="script: train.py",
    )


def _write_log_lines(project: ProjectContext, commands: list[str]) -> None:
    log_path = project.path / "advice.log"
    with log_path.open("a", encoding="utf-8") as out:
        for cmd in commands:
            out.write(
                json.dumps(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "command": cmd,
                    }
                )
                + "\n"
            )


# --------------------------------------------------------------------------- #
# Behaviour                                                                   #
# --------------------------------------------------------------------------- #


def test_status_fresh_project_renders_panels(project: ProjectContext, tmp_path: Path) -> None:
    runner = CliRunner()
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(cli, ["status"])
    finally:
        os.chdir(cwd)

    assert result.exit_code == 0, result.output
    text = result.output.lower()
    assert "project" in text
    assert "active state" in text
    # Fresh project: no decisions yet.
    assert "no decisions" in text or "nothing logged" in text


def test_status_populated_project_shows_recent_decisions(
    project: ProjectContext, tmp_path: Path
) -> None:
    _populate_state(project)

    runner = CliRunner()
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(cli, ["status"])
    finally:
        os.chdir(cwd)

    assert result.exit_code == 0
    # Active-state panel surfaces the recorded fields.
    assert "binary_classification" in result.output.lower()
    assert "churn" in result.output.lower()
    assert "xgboost" in result.output.lower()
    # Decisions table includes both commands.
    assert "advise" in result.output
    assert "audit" in result.output


def test_status_command_counts_table_aggregates_log_entries(
    project: ProjectContext, tmp_path: Path
) -> None:
    _write_log_lines(project, ["advise", "advise", "audit", "watch"])

    runner = CliRunner()
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(cli, ["status"])
    finally:
        os.chdir(cwd)

    assert result.exit_code == 0
    assert "command activity" in result.output.lower()
    # Tally formatting: command name on the left, integer count on the right.
    assert "advise" in result.output
    assert "2" in result.output


def test_status_recent_flag_caps_decisions(project: ProjectContext, tmp_path: Path) -> None:
    for i in range(6):
        project.append_decision(
            command="audit",
            summary=f"audit {i}",
        )

    runner = CliRunner()
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(cli, ["status", "--recent", "3"])
    finally:
        os.chdir(cwd)

    assert result.exit_code == 0
    # Only the last three audit messages should be on the screen.
    assert "audit 5" in result.output
    assert "audit 4" in result.output
    assert "audit 3" in result.output
    assert "audit 0" not in result.output


def test_status_exits_nonzero_when_no_project(tmp_path: Path) -> None:
    runner = CliRunner()
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(cli, ["status"])
    finally:
        os.chdir(cwd)

    assert result.exit_code != 0
    assert "no .mlcompass" in result.output.lower()


# --------------------------------------------------------------------------- #
# Version-key regression (field-test bug #7)                                  #
# --------------------------------------------------------------------------- #


def test_status_renders_mlcompass_version_from_current_key(
    project: ProjectContext, tmp_path: Path
) -> None:
    """``mlcompass init`` writes ``mlcompass_version`` in project.yaml.

    Pre-v0.6.1 the status panel looked for ``ml_compass_version``
    (with the underscore) — a leftover from the v0.4 ml-copilot →
    mlcompass rename — and silently rendered "—" instead. This test
    pins the current key as the primary lookup.
    """
    import yaml

    project_yaml = project.path / "project.yaml"
    meta = yaml.safe_load(project_yaml.read_text(encoding="utf-8"))
    meta["mlcompass_version"] = "0.6.1"
    project_yaml.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")

    runner = CliRunner()
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(cli, ["status"])
    finally:
        os.chdir(cwd)

    assert result.exit_code == 0, result.output
    assert "0.6.1" in result.output
    assert "mlcompass ver" in result.output


def test_status_falls_back_to_legacy_version_keys(project: ProjectContext, tmp_path: Path) -> None:
    """Projects pre-dating the v0.4 rename still render their version."""
    import yaml

    project_yaml = project.path / "project.yaml"
    meta = yaml.safe_load(project_yaml.read_text(encoding="utf-8"))
    # Simulate an old project: drop the new key, set the legacy one.
    meta.pop("mlcompass_version", None)
    meta["ml_copilot_version"] = "0.1.0"
    project_yaml.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")

    runner = CliRunner()
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(cli, ["status"])
    finally:
        os.chdir(cwd)

    assert result.exit_code == 0
    assert "0.1.0" in result.output


# --------------------------------------------------------------------------- #
# Help / discovery                                                            #
# --------------------------------------------------------------------------- #


def test_root_help_lists_status() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "status" in result.output


def test_status_help_lists_recent_flag() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["status", "--help"])
    assert result.exit_code == 0
    assert "--recent" in result.output
