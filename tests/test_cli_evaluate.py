"""CLI integration tests for ``mlcompass evaluate``."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from mlcompass.cli import cli
from mlcompass.context import ProjectContext

# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def binary_csv(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    n = 100
    y_true = rng.integers(0, 2, size=n)
    y_prob = np.where(
        y_true == 1,
        rng.uniform(0.55, 0.95, size=n),
        rng.uniform(0.05, 0.45, size=n),
    )
    y_pred = (y_prob >= 0.5).astype(int)
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob})
    p = tmp_path / "preds.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def regression_csv(tmp_path: Path) -> Path:
    rng = np.random.default_rng(1)
    x = np.linspace(0, 10, 50)
    y_true = 2 * x + 1
    y_pred = y_true + rng.normal(0, 0.5, size=50)
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    p = tmp_path / "preds.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def multiclass_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "y_true": ["a"] * 10 + ["b"] * 10 + ["c"] * 10,
            "y_pred": ["a"] * 8 + ["b"] * 2 + ["b"] * 10 + ["c"] * 10,
        }
    )
    p = tmp_path / "preds.csv"
    df.to_csv(p, index=False)
    return p


# --------------------------------------------------------------------------- #
# Smoke / behaviour                                                           #
# --------------------------------------------------------------------------- #


def test_evaluate_binary_csv_succeeds(binary_csv: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["evaluate", str(binary_csv)])
    assert result.exit_code == 0, result.output
    text = result.output.lower()
    assert "evaluation" in text
    assert "binary" in text or "auc" in text


def test_evaluate_regression_csv_succeeds(regression_csv: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["evaluate", str(regression_csv)])
    assert result.exit_code == 0, result.output
    text = result.output.lower()
    assert "regression" in text
    assert "rmse" in text


def test_evaluate_multiclass_csv_succeeds(multiclass_csv: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["evaluate", str(multiclass_csv)])
    assert result.exit_code == 0, result.output
    text = result.output.lower()
    assert "multiclass" in text
    assert "per-class" in text or "per class" in text


def test_evaluate_shows_threshold_sweep_for_binary(binary_csv: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["evaluate", str(binary_csv)])
    assert result.exit_code == 0
    assert "threshold sweep" in result.output.lower()


# --------------------------------------------------------------------------- #
# Flags                                                                       #
# --------------------------------------------------------------------------- #


def test_evaluate_explicit_columns_pass_through(
    tmp_path: Path,
) -> None:
    df = pd.DataFrame(
        {
            "actual": [0, 1, 0, 1, 0],
            "predicted": [0, 1, 1, 1, 0],
            "score": [0.1, 0.9, 0.6, 0.8, 0.2],
        }
    )
    p = tmp_path / "renamed.csv"
    df.to_csv(p, index=False)

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            cli,
            [
                "evaluate",
                str(p),
                "--y-true",
                "actual",
                "--y-pred",
                "predicted",
                "--y-prob",
                "score",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "actual" in result.output


def test_evaluate_task_override(regression_csv: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            cli,
            ["evaluate", str(regression_csv), "--task", "regression"],
        )
    assert result.exit_code == 0


def test_evaluate_hard_examples_count_flag(binary_csv: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["evaluate", str(binary_csv), "--hard-examples", "2"])
    assert result.exit_code == 0


# --------------------------------------------------------------------------- #
# Error paths                                                                 #
# --------------------------------------------------------------------------- #


def test_evaluate_missing_file_exits_nonzero(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["evaluate", str(tmp_path / "nope.csv")])
    assert result.exit_code != 0


def test_evaluate_no_truth_column_fails_cleanly(tmp_path: Path) -> None:
    df = pd.DataFrame({"foo": [1, 2, 3]})
    p = tmp_path / "bad.csv"
    df.to_csv(p, index=False)
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["evaluate", str(p)])
    assert result.exit_code != 0
    assert "ground-truth" in result.output.lower()


# --------------------------------------------------------------------------- #
# Help / discovery                                                            #
# --------------------------------------------------------------------------- #


def test_root_help_lists_evaluate() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "evaluate" in result.output


def test_evaluate_help_lists_options() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["evaluate", "--help"])
    assert result.exit_code == 0
    for opt in ("--y-true", "--y-pred", "--y-prob", "--task", "--hard-examples"):
        assert opt in result.output


# --------------------------------------------------------------------------- #
# Persistence                                                                 #
# --------------------------------------------------------------------------- #


def test_evaluate_persists_to_project_context(binary_csv: Path, tmp_path: Path) -> None:
    project = ProjectContext.init("test-proj", parent_dir=tmp_path)

    runner = CliRunner()
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(cli, ["evaluate", str(binary_csv)])
    finally:
        os.chdir(cwd)

    assert result.exit_code == 0, result.output

    state = project.read_context()
    assert len(state["decisions"]) == 1
    decision = state["decisions"][0]
    assert decision["command"] == "evaluate"

    log_path = project.path / "advice.log"
    entries = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert entries
    entry = entries[0]
    assert entry["command"] == "evaluate"
    assert entry["task"] == "binary_classification"
    assert "metrics" in entry
