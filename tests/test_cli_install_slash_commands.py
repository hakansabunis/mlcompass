"""CLI integration tests for ``mlcompass install-slash-commands``."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from mlcompass.cli import cli

# --------------------------------------------------------------------------- #
# Project scope                                                               #
# --------------------------------------------------------------------------- #


def test_project_scope_writes_to_cwd_dot_claude(tmp_path: Path) -> None:
    """Default scope = project ⇒ writes to .claude/commands/ in cwd."""
    runner = CliRunner()
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(cli, ["install-slash-commands"])
    finally:
        os.chdir(cwd)

    assert result.exit_code == 0, result.output
    target = tmp_path / ".claude" / "commands"
    assert target.is_dir()
    installed = sorted(p.name for p in target.glob("*.md"))
    # All 11 shipped slash commands must land.
    assert len(installed) >= 11
    # All names follow the mlc- prefix convention.
    assert all(name.startswith("mlc-") for name in installed)
    # Sanity-check a known command body.
    advise = (target / "mlc-advise.md").read_text(encoding="utf-8")
    assert "mlcompass_advise" in advise


def test_project_scope_renders_summary_panel(tmp_path: Path) -> None:
    runner = CliRunner()
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(cli, ["install-slash-commands"])
    finally:
        os.chdir(cwd)

    assert result.exit_code == 0
    # Panel mentions number installed + a slash-prefixed sample command name.
    assert "installed" in result.output.lower()
    assert "/mlc-advise" in result.output


# --------------------------------------------------------------------------- #
# Idempotency + --force                                                       #
# --------------------------------------------------------------------------- #


def test_second_run_skips_existing_files_by_default(tmp_path: Path) -> None:
    runner = CliRunner()
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        # First install.
        runner.invoke(cli, ["install-slash-commands"])
        # Tamper with one file so we can detect overwrites.
        marker = tmp_path / ".claude" / "commands" / "mlc-advise.md"
        marker.write_text("LOCAL EDIT", encoding="utf-8")
        # Second install — should NOT overwrite the local edit.
        result = runner.invoke(cli, ["install-slash-commands"])
    finally:
        os.chdir(cwd)

    assert result.exit_code == 0
    assert "skipped" in result.output.lower()
    assert marker.read_text(encoding="utf-8") == "LOCAL EDIT"


def test_force_flag_overwrites_existing_files(tmp_path: Path) -> None:
    runner = CliRunner()
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        runner.invoke(cli, ["install-slash-commands"])
        marker = tmp_path / ".claude" / "commands" / "mlc-advise.md"
        marker.write_text("LOCAL EDIT", encoding="utf-8")
        result = runner.invoke(cli, ["install-slash-commands", "--force"])
    finally:
        os.chdir(cwd)

    assert result.exit_code == 0
    # File is now back to the shipped content.
    contents = marker.read_text(encoding="utf-8")
    assert "LOCAL EDIT" not in contents
    assert "mlcompass_advise" in contents


# --------------------------------------------------------------------------- #
# User scope                                                                  #
# --------------------------------------------------------------------------- #


def test_user_scope_writes_to_home_dot_claude(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--scope user resolves to ~/.claude/commands/ — redirect HOME."""
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
    monkeypatch.setenv("HOME", str(tmp_path))  # POSIX

    runner = CliRunner()
    result = runner.invoke(cli, ["install-slash-commands", "--scope", "user"])

    assert result.exit_code == 0, result.output
    target = tmp_path / ".claude" / "commands"
    assert target.is_dir()
    installed = sorted(p.name for p in target.glob("*.md"))
    assert len(installed) >= 11


# --------------------------------------------------------------------------- #
# Discovery                                                                   #
# --------------------------------------------------------------------------- #


def test_root_help_lists_install_slash_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "install-slash-commands" in result.output
