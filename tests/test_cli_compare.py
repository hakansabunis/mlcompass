"""Tests for ``mlcompass compare`` CLI command."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from mlcompass.cli import cli
from mlcompass.context import ProjectContext

# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


def _make_run(
    parent: Path,
    *,
    run_id: str,
    config: dict,
    metrics: list[dict],
    name: str | None = None,
) -> Path:
    run_dir = parent / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    meta: dict = {}
    if name is not None:
        meta["name"] = name
    meta["created"] = "2026-05-29T15:00:00Z"
    meta["config"] = config
    (run_dir / "config.yaml").write_text(yaml.safe_dump(meta), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps({"metrics": metrics}), encoding="utf-8")
    return run_dir


@pytest.fixture
def two_runs_by_path(tmp_path: Path) -> tuple[Path, Path]:
    a = _make_run(
        tmp_path,
        run_id="run-3",
        name="baseline",
        config={"lr": 1e-3, "batch_size": 64},
        metrics=[{"epoch": 0, "val_loss": 0.5, "val_acc": 0.78}],
    )
    b = _make_run(
        tmp_path,
        run_id="run-7",
        name="lower-lr",
        config={"lr": 3e-4, "batch_size": 64, "dropout": 0.3},
        metrics=[{"epoch": 0, "val_loss": 0.3, "val_acc": 0.85}],
    )
    return a, b


# --------------------------------------------------------------------------- #
# Smoke / output                                                              #
# --------------------------------------------------------------------------- #


def test_compare_by_paths_succeeds(two_runs_by_path: tuple[Path, Path], tmp_path: Path) -> None:
    a, b = two_runs_by_path
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["compare", str(a), str(b)])

    assert result.exit_code == 0, result.output
    text = result.output.lower()
    assert "run comparison" in text
    assert "val_loss" in text
    assert "val_acc" in text


def test_compare_reports_winner_in_output(
    two_runs_by_path: tuple[Path, Path], tmp_path: Path
) -> None:
    a, b = two_runs_by_path
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["compare", str(a), str(b)])

    assert result.exit_code == 0
    # Run B is better on both lower val_loss and higher val_acc
    text = result.output.lower()
    assert "run b wins" in text


def test_compare_shows_config_diff(two_runs_by_path: tuple[Path, Path], tmp_path: Path) -> None:
    a, b = two_runs_by_path
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["compare", str(a), str(b)])
    assert result.exit_code == 0
    text = result.output.lower()
    # lr differs; dropout is only in B.
    assert "lr" in text
    assert "dropout" in text


# --------------------------------------------------------------------------- #
# Project-relative lookup                                                     #
# --------------------------------------------------------------------------- #


def test_compare_resolves_run_ids_through_project_context(tmp_path: Path) -> None:
    project = ProjectContext.init("test-proj", parent_dir=tmp_path)
    runs_root = project.path / "runs"
    _make_run(
        runs_root,
        run_id="run-3",
        config={"lr": 1e-3},
        metrics=[{"epoch": 0, "val_loss": 0.5}],
    )
    _make_run(
        runs_root,
        run_id="run-7",
        config={"lr": 3e-4},
        metrics=[{"epoch": 0, "val_loss": 0.3}],
    )

    runner = CliRunner()
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(cli, ["compare", "run-3", "run-7"])
    finally:
        os.chdir(cwd)

    assert result.exit_code == 0, result.output
    assert "run-3" in result.output
    assert "run-7" in result.output


# --------------------------------------------------------------------------- #
# Error / help                                                                #
# --------------------------------------------------------------------------- #


def test_compare_missing_run_exits_nonzero(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["compare", "nope", "still-nope"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_root_help_lists_compare() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "compare" in result.output


def test_compare_help_describes_arguments() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["compare", "--help"])
    assert result.exit_code == 0
    assert "run_a" in result.output.lower() or "run a" in result.output.lower()


# --------------------------------------------------------------------------- #
# Persistence                                                                 #
# --------------------------------------------------------------------------- #


def test_compare_persists_to_project_context(tmp_path: Path) -> None:
    project = ProjectContext.init("test-proj", parent_dir=tmp_path)
    runs_root = project.path / "runs"
    _make_run(
        runs_root,
        run_id="run-3",
        config={"lr": 1e-3},
        metrics=[{"epoch": 0, "val_loss": 0.5}],
    )
    _make_run(
        runs_root,
        run_id="run-7",
        config={"lr": 3e-4},
        metrics=[{"epoch": 0, "val_loss": 0.3}],
    )

    runner = CliRunner()
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(cli, ["compare", "run-3", "run-7"])
    finally:
        os.chdir(cwd)

    assert result.exit_code == 0, result.output

    state = project.read_context()
    assert len(state["decisions"]) == 1
    decision = state["decisions"][0]
    assert decision["command"] == "compare"
    assert "run-3" in decision["summary"]
    assert "run-7" in decision["summary"]

    log_path = project.path / "advice.log"
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    log_entry = json.loads(lines[0])
    assert log_entry["command"] == "compare"
    assert log_entry["run_a"] == "run-3"
    assert log_entry["run_b"] == "run-7"
    assert log_entry["verdict"] == "b_better"
