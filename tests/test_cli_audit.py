"""Tests for ``mlcompass audit`` CLI command."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from mlcompass.cli import cli
from mlcompass.context import ProjectContext


@pytest.fixture
def clean_script(tmp_path: Path) -> Path:
    """A small training script that passes every rule."""
    code = """
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

torch.manual_seed(42)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
model = nn.Linear(10, 1)
model.train()
model.eval()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
"""
    p = tmp_path / "train_clean.py"
    p.write_text(code, encoding="utf-8")
    return p


@pytest.fixture
def buggy_script(tmp_path: Path) -> Path:
    """A training script that should hit several rules."""
    code = """
import torch
import torch.nn as nn

# No random seed — error
model = nn.LSTM(10, 20)        # RNN without grad-clipping warning
model.train()                   # train() with no eval() — warning
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, momentum=0.9)  # error
loader = torch.utils.data.DataLoader(ds, batch_size=1)  # info + dataloader warning
y = torch.log(x)               # loss_stability warning
"""
    p = tmp_path / "train_buggy.py"
    p.write_text(code, encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# Smoke / output                                                              #
# --------------------------------------------------------------------------- #


def test_audit_clean_script_exits_zero(clean_script: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["audit", str(clean_script)])
    assert result.exit_code == 0, result.output
    assert "no issues" in result.output.lower() or "all clear" in result.output.lower()


def test_audit_buggy_script_reports_findings(
    buggy_script: Path,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["audit", str(buggy_script)])
    assert result.exit_code == 0, result.output
    # The output should mention at least the seed and momentum issues.
    text = result.output.lower()
    assert "seed" in text
    assert "momentum" in text


def test_audit_shows_summary_panel(buggy_script: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["audit", str(buggy_script)])
    assert result.exit_code == 0
    assert "script audit" in result.output.lower()


# --------------------------------------------------------------------------- #
# Flags                                                                       #
# --------------------------------------------------------------------------- #


def test_audit_skip_flag_filters_rules(
    buggy_script: Path,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        full = runner.invoke(cli, ["audit", str(buggy_script)])
        skipped = runner.invoke(
            cli, ["audit", str(buggy_script), "--skip", "seed"]
        )

    assert full.exit_code == 0
    assert skipped.exit_code == 0
    # The "no random seed" message should disappear when we skip the rule.
    assert "no random seed" in full.output.lower()
    assert "no random seed" not in skipped.output.lower()


def test_audit_skip_multiple_rules(buggy_script: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            cli,
            [
                "audit",
                str(buggy_script),
                "--skip",
                "seed",
                "--skip",
                "optimizer",
            ],
        )
    assert result.exit_code == 0
    assert "no random seed" not in result.output.lower()
    assert "momentum" not in result.output.lower()


# --------------------------------------------------------------------------- #
# Error paths                                                                 #
# --------------------------------------------------------------------------- #


def test_audit_missing_script_exits_nonzero(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["audit", str(tmp_path / "nope.py")])
    assert result.exit_code != 0


def test_audit_help_lists_skip_option() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["audit", "--help"])
    assert result.exit_code == 0
    assert "--skip" in result.output


def test_root_help_lists_audit() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "audit" in result.output


# --------------------------------------------------------------------------- #
# Persistence                                                                 #
# --------------------------------------------------------------------------- #


def test_audit_persists_to_project_context(
    buggy_script: Path,
    tmp_path: Path,
) -> None:
    project_root = buggy_script.parent
    project = ProjectContext.init("test-proj", parent_dir=project_root)

    runner = CliRunner()
    cwd = os.getcwd()
    os.chdir(project_root)
    try:
        result = runner.invoke(cli, ["audit", str(buggy_script)])
    finally:
        os.chdir(cwd)

    assert result.exit_code == 0, result.output

    state = project.read_context()
    assert len(state["decisions"]) == 1
    decision = state["decisions"][0]
    assert decision["command"] == "audit"
    assert "audit" in decision["summary"].lower()

    log_path = project.path / "advice.log"
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    log_entry = json.loads(lines[0])
    assert log_entry["command"] == "audit"
    assert log_entry["script"] == str(buggy_script)
    assert isinstance(log_entry["findings"], list)
    assert any(f["rule_id"] == "seed" for f in log_entry["findings"])
