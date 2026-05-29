"""Tests for ``mlcompass init`` CLI command."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from mlcompass.cli import cli
from mlcompass.context import DEFAULT_PROJECT_DIR


def test_init_creates_project(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["init", "my-project", "--path", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / DEFAULT_PROJECT_DIR).is_dir()


def test_init_prints_next_step_hint(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["init", "my-project", "--path", str(tmp_path)],
    )
    assert result.exit_code == 0
    # The output should hint at the next command the user should run.
    assert "advise" in result.output.lower()


def test_init_fails_when_project_already_exists(tmp_path: Path) -> None:
    runner = CliRunner()
    first = runner.invoke(cli, ["init", "first", "--path", str(tmp_path)])
    assert first.exit_code == 0

    second = runner.invoke(cli, ["init", "second", "--path", str(tmp_path)])
    assert second.exit_code != 0


def test_init_accepts_default_model_option(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "init",
            "my-project",
            "--path",
            str(tmp_path),
            "--default-model",
            "claude-haiku-4-5",
        ],
    )
    assert result.exit_code == 0, result.output


def test_root_help_lists_init(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "init" in result.output


def test_version_flag_works() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "mlcompass" in result.output
