"""CLI integration tests for ``mlcompass deploy``."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from mlcompass.cli import cli
from mlcompass.context import ProjectContext

# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def pytorch_model(tmp_path: Path) -> Path:
    p = tmp_path / "model.pt"
    p.write_bytes(b"PK\x03\x04" + b"\x00" * 60)
    return p


@pytest.fixture
def pickle_model(tmp_path: Path) -> Path:
    p = tmp_path / "model.pkl"
    p.write_bytes(b"\x80\x04" + b"\x00" * 60)
    return p


@pytest.fixture
def requirements(tmp_path: Path) -> Path:
    p = tmp_path / "requirements.txt"
    p.write_text(
        "torch==2.1.0\nscikit-learn==1.3.0\nrequests\n",
        encoding="utf-8",
    )
    return p


# --------------------------------------------------------------------------- #
# Smoke / behaviour                                                           #
# --------------------------------------------------------------------------- #


def test_deploy_pytorch_local_succeeds(pytorch_model: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["deploy", str(pytorch_model)])
    assert result.exit_code == 0, result.output
    text = result.output.lower()
    assert "deployment readiness" in text
    assert "pytorch" in text


def test_deploy_pickle_warns_about_security(pickle_model: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["deploy", str(pickle_model)])
    assert result.exit_code == 0
    assert "pickle" in result.output.lower()


def test_deploy_with_requirements_renders_dep_table(
    pytorch_model: Path, requirements: Path, tmp_path: Path
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            cli,
            ["deploy", str(pytorch_model), "--requirements", str(requirements)],
        )
    assert result.exit_code == 0
    text = result.output.lower()
    assert "dependencies" in text
    assert "torch" in text


def test_deploy_unpinned_dependency_warning_appears(
    pytorch_model: Path, requirements: Path, tmp_path: Path
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            cli,
            ["deploy", str(pytorch_model), "--requirements", str(requirements)],
        )
    assert result.exit_code == 0
    # ``requests`` is unpinned in the fixture.
    assert "unpinned" in result.output.lower()


def test_deploy_lambda_target_surfaces_target_specific_warnings(
    pickle_model: Path, tmp_path: Path
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["deploy", str(pickle_model), "--target", "lambda"])
    assert result.exit_code == 0
    text = result.output.lower()
    assert "lambda" in text or "cold start" in text


def test_deploy_checklist_visible(pytorch_model: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["deploy", str(pytorch_model)])
    assert result.exit_code == 0
    text = result.output.lower()
    assert "production checklist" in text
    assert "drift" in text or "rollback" in text


# --------------------------------------------------------------------------- #
# Error paths                                                                 #
# --------------------------------------------------------------------------- #


def test_deploy_missing_model_exits_nonzero(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["deploy", str(tmp_path / "nope.pt")])
    assert result.exit_code != 0


def test_deploy_invalid_target_exits_nonzero(pytorch_model: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["deploy", str(pytorch_model), "--target", "mars"])
    assert result.exit_code != 0


# --------------------------------------------------------------------------- #
# Help / discovery                                                            #
# --------------------------------------------------------------------------- #


def test_root_help_lists_deploy() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "deploy" in result.output


def test_deploy_help_lists_options() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["deploy", "--help"])
    assert result.exit_code == 0
    for opt in ("--requirements", "--target"):
        assert opt in result.output


# --------------------------------------------------------------------------- #
# Persistence                                                                 #
# --------------------------------------------------------------------------- #


def test_deploy_persists_to_project_context(pytorch_model: Path, tmp_path: Path) -> None:
    project = ProjectContext.init("test-proj", parent_dir=tmp_path)

    runner = CliRunner()
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(cli, ["deploy", str(pytorch_model)])
    finally:
        os.chdir(cwd)

    assert result.exit_code == 0, result.output

    state = project.read_context()
    assert len(state["decisions"]) == 1
    decision = state["decisions"][0]
    assert decision["command"] == "deploy"

    log_path = project.path / "advice.log"
    entries = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert entries
    entry = entries[0]
    assert entry["command"] == "deploy"
    assert entry["target"] == "local"
    assert entry["format"] == "pytorch"
    assert "checklist" in entry
