"""Tests for ``mlcompass watch`` CLI command (one-shot mode)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from mlcompass.cli import cli
from mlcompass.context import ProjectContext


@pytest.fixture
def clean_log(tmp_path: Path) -> Path:
    """Healthy training log — both losses decreasing steadily."""
    lines = []
    for epoch in range(8):
        train = 0.8 - epoch * 0.08
        val = 0.85 - epoch * 0.07
        lines.append(f"Epoch {epoch} train_loss={train:.4f} val_loss={val:.4f}")
    p = tmp_path / "train_clean.log"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


@pytest.fixture
def overfitting_log(tmp_path: Path) -> Path:
    """Classic overfitting signature."""
    lines = [
        "Epoch 0 train_loss=0.50 val_loss=0.50",
        "Epoch 1 train_loss=0.40 val_loss=0.48",
        "Epoch 2 train_loss=0.30 val_loss=0.50",
        "Epoch 3 train_loss=0.20 val_loss=0.55",
        "Epoch 4 train_loss=0.10 val_loss=0.60",
    ]
    p = tmp_path / "train_overfit.log"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


@pytest.fixture
def nan_log(tmp_path: Path) -> Path:
    """Loss exploded into NaN at epoch 5."""
    lines = [
        "Epoch 0 train_loss=0.80",
        "Epoch 1 train_loss=0.70",
        "Epoch 2 train_loss=0.65",
        "Epoch 3 train_loss=0.60",
        "Epoch 4 train_loss=0.55",
        "Epoch 5 train_loss=nan",
    ]
    p = tmp_path / "train_nan.log"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# Smoke / behaviour                                                           #
# --------------------------------------------------------------------------- #


def test_watch_clean_log_reports_no_anomalies(clean_log: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["watch", str(clean_log)])
    assert result.exit_code == 0, result.output
    assert "no anomalies" in result.output.lower()


def test_watch_overfitting_log_surfaces_overfitting(overfitting_log: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["watch", str(overfitting_log)])
    assert result.exit_code == 0
    assert "overfitting" in result.output.lower()


def test_watch_nan_log_surfaces_nan(nan_log: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["watch", str(nan_log)])
    assert result.exit_code == 0
    assert "nan" in result.output.lower()


def test_watch_shows_recent_metrics_table(clean_log: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["watch", str(clean_log)])
    assert result.exit_code == 0
    assert "recent metrics" in result.output.lower()
    # train_loss and val_loss should be column headers in the table.
    assert "train_loss" in result.output
    assert "val_loss" in result.output


def test_watch_shows_overview_panel(clean_log: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["watch", str(clean_log)])
    assert result.exit_code == 0
    assert "watch report" in result.output.lower()


# --------------------------------------------------------------------------- #
# Error / help                                                                #
# --------------------------------------------------------------------------- #


def test_watch_missing_log_exits_nonzero(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["watch", str(tmp_path / "nope.log")])
    assert result.exit_code != 0


def test_root_help_lists_watch() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "watch" in result.output


def test_watch_help_lists_follow_flag() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["watch", "--help"])
    assert result.exit_code == 0
    assert "--follow" in result.output


# --------------------------------------------------------------------------- #
# Persistence                                                                 #
# --------------------------------------------------------------------------- #


def test_watch_persists_to_project_context(overfitting_log: Path, tmp_path: Path) -> None:
    project_root = overfitting_log.parent
    project = ProjectContext.init("test-proj", parent_dir=project_root)

    runner = CliRunner()
    cwd = os.getcwd()
    os.chdir(project_root)
    try:
        result = runner.invoke(cli, ["watch", str(overfitting_log)])
    finally:
        os.chdir(cwd)

    assert result.exit_code == 0, result.output

    state = project.read_context()
    assert len(state["decisions"]) == 1
    decision = state["decisions"][0]
    assert decision["command"] == "watch"
    assert "watch" in decision["summary"].lower()

    log_path = project.path / "advice.log"
    entries = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").strip().splitlines()
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["command"] == "watch"
    assert entry["log"] == str(overfitting_log)
    assert any(f["rule_id"] == "overfitting" for f in entry["findings"])
