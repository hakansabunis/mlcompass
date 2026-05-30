"""CLI integration tests for ``mlcompass optimize``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

import mlcompass.cli as cli_module
from mlcompass.cli import cli
from mlcompass.context import ProjectContext

# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


def _make_run(runs_dir: Path, name: str, config: dict, metrics: list[dict]) -> None:
    run = runs_dir / name
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.yaml").write_text(
        yaml.safe_dump({"name": name, "config": config}), encoding="utf-8"
    )
    (run / "metrics.json").write_text(json.dumps({"metrics": metrics}), encoding="utf-8")


@pytest.fixture
def project_with_runs(tmp_path: Path) -> tuple[ProjectContext, Path]:
    """An mlcompass project with three run subdirectories."""
    project = ProjectContext.init("demo", parent_dir=tmp_path)
    runs_dir = project.path / "runs"
    _make_run(runs_dir, "r1", {"lr": 0.01}, [{"epoch": 0, "val_acc": 0.6}])
    _make_run(runs_dir, "r2", {"lr": 0.001}, [{"epoch": 0, "val_acc": 0.75}])
    _make_run(runs_dir, "r3", {"lr": 0.0001}, [{"epoch": 0, "val_acc": 0.85}])
    return project, tmp_path


# --------------------------------------------------------------------------- #
# Behaviour                                                                   #
# --------------------------------------------------------------------------- #


def test_optimize_default_runs_dir_from_project(
    project_with_runs: tuple[ProjectContext, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, tmp_path = project_with_runs
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["optimize", "--metric", "val_acc"])
    assert result.exit_code == 0, result.output
    assert "r3" in result.output  # leaderboard winner


def test_optimize_explicit_runs_dir(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _make_run(runs_dir, "a", {"lr": 0.01}, [{"epoch": 0, "auc": 0.7}])
    _make_run(runs_dir, "b", {"lr": 0.001}, [{"epoch": 0, "auc": 0.82}])
    _make_run(runs_dir, "c", {"lr": 0.0001}, [{"epoch": 0, "auc": 0.88}])
    runner = CliRunner()
    result = runner.invoke(cli, ["optimize", "--runs-dir", str(runs_dir), "--metric", "auc"])
    assert result.exit_code == 0, result.output
    assert "auc" in result.output.lower()


def test_optimize_no_project_no_runs_dir_exits_2(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["optimize", "--metric", "val_acc"], catch_exceptions=False)
    # No project AND no --runs-dir ⇒ exit 2.
    assert result.exit_code == 2


def test_optimize_bad_metric_exits_2(
    project_with_runs: tuple[ProjectContext, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, tmp_path = project_with_runs
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["optimize", "--metric", "nonexistent_metric"])
    assert result.exit_code == 2
    assert "nonexistent_metric" in result.output


def test_optimize_constraints_passthrough(
    project_with_runs: tuple[ProjectContext, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, tmp_path = project_with_runs
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "optimize",
            "--metric",
            "val_acc",
            "--constraints",
            "lr:0.0001-0.1",
            "--suggestions",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output


def test_optimize_bad_constraints_exits_2(
    project_with_runs: tuple[ProjectContext, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, tmp_path = project_with_runs
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["optimize", "--metric", "val_acc", "--constraints", "lr_no_colon"])
    assert result.exit_code == 2


def test_optimize_llm_strategist_renders_when_stubbed(
    project_with_runs: tuple[ProjectContext, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, tmp_path = project_with_runs
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")

    def _stub(_result: dict[str, Any], *, model: str) -> dict[str, Any]:
        return {
            "headline": "lr is the dominant lever.",
            "pattern": "Lower learning rate monotonically improves val_acc.",
            "next_plan": [
                "Try lr=3e-5 with the same batch size.",
                "Add early stopping to avoid overshooting.",
                "Sweep batch_size after lr is locked.",
            ],
        }

    monkeypatch.setattr(cli_module, "_optimize_strategist_callable", _stub)

    runner = CliRunner()
    result = runner.invoke(cli, ["optimize", "--metric", "val_acc", "--llm"])
    assert result.exit_code == 0, result.output
    assert "HPO strategist" in result.output
    assert "lr is the dominant lever" in result.output


def test_optimize_llm_skipped_without_api_key(
    project_with_runs: tuple[ProjectContext, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, tmp_path = project_with_runs
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    runner = CliRunner()
    result = runner.invoke(cli, ["optimize", "--metric", "val_acc", "--llm"])
    assert result.exit_code == 0
    assert "ANTHROPIC_API_KEY" in result.output


def test_root_help_lists_optimize() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "optimize" in result.output
