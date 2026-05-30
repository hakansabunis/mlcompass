"""CLI integration tests for ``mlcompass watch --apply``."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from mlcompass import cli as cli_module
from mlcompass.cli import cli
from mlcompass.context import ProjectContext

# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def overfitting_log(tmp_path: Path) -> Path:
    lines = [
        "Epoch 0 train_loss=0.50 val_loss=0.50",
        "Epoch 1 train_loss=0.40 val_loss=0.48",
        "Epoch 2 train_loss=0.30 val_loss=0.50",
        "Epoch 3 train_loss=0.20 val_loss=0.55",
        "Epoch 4 train_loss=0.10 val_loss=0.60",
    ]
    p = tmp_path / "train.log"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    p = tmp_path / "train.yaml"
    p.write_text("dropout: 0.1\nlr: 0.001\n", encoding="utf-8")
    return p


@pytest.fixture
def diagnosis_with_edit() -> dict[str, Any]:
    return {
        "diagnosis": [
            {
                "finding_rule_id": "overfitting",
                "hypothesis": "Capacity too high for dataset size.",
                "recommended_action": "Increase dropout to 0.3.",
                "confidence": "high",
            }
        ],
        "summary": "Stop training, regularize, restart.",
        "suggested_edits": [
            {
                "key": "dropout",
                "current_value": 0.1,
                "proposed_value": 0.3,
                "rationale": "train/val gap widened to 0.50 at epoch 4",
            }
        ],
    }


@pytest.fixture
def with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")


# --------------------------------------------------------------------------- #
# Happy path                                                                  #
# --------------------------------------------------------------------------- #


def test_watch_apply_applies_edit_with_yes(
    overfitting_log: Path,
    config_file: Path,
    tmp_path: Path,
    with_api_key: None,
    diagnosis_with_edit: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli_module,
        "_watch_diagnostician_callable",
        lambda snaps, finds, *, model="claude-opus-4-7": diagnosis_with_edit,
    )

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            cli,
            [
                "watch",
                str(overfitting_log),
                "--llm",
                "--apply",
                "--config",
                str(config_file),
                "--yes",
            ],
        )

    assert result.exit_code == 0, result.output
    data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert data["dropout"] == 0.3
    assert "applied" in result.output.lower()


def test_watch_apply_records_rejection_with_interactive_no(
    overfitting_log: Path,
    config_file: Path,
    tmp_path: Path,
    with_api_key: None,
    diagnosis_with_edit: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli_module,
        "_watch_diagnostician_callable",
        lambda snaps, finds, *, model="claude-opus-4-7": diagnosis_with_edit,
    )

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            cli,
            [
                "watch",
                str(overfitting_log),
                "--llm",
                "--apply",
                "--config",
                str(config_file),
            ],
            input="n\n",  # Click confirm default-False; "n" makes that explicit.
        )

    assert result.exit_code == 0, result.output
    data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert data["dropout"] == 0.1, "no edit should have been applied"
    assert "rejected" in result.output.lower()


# --------------------------------------------------------------------------- #
# Skips and guards                                                            #
# --------------------------------------------------------------------------- #


def test_watch_apply_without_llm_warns(
    overfitting_log: Path,
    config_file: Path,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            cli,
            ["watch", str(overfitting_log), "--apply", "--config", str(config_file)],
        )
    assert result.exit_code == 0
    assert "--llm" in result.output


def test_watch_apply_without_config_warns(
    overfitting_log: Path,
    tmp_path: Path,
    with_api_key: None,
    diagnosis_with_edit: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli_module,
        "_watch_diagnostician_callable",
        lambda snaps, finds, *, model="claude-opus-4-7": diagnosis_with_edit,
    )

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["watch", str(overfitting_log), "--llm", "--apply"])
    assert result.exit_code == 0
    assert "--config" in result.output


def test_watch_apply_skips_when_no_suggested_edits(
    overfitting_log: Path,
    config_file: Path,
    tmp_path: Path,
    with_api_key: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Diagnosis with no suggested_edits field.
    no_edits = {
        "diagnosis": [
            {
                "finding_rule_id": "overfitting",
                "hypothesis": "x",
                "recommended_action": "y",
                "confidence": "high",
            }
        ],
        "summary": "x",
    }
    monkeypatch.setattr(
        cli_module,
        "_watch_diagnostician_callable",
        lambda snaps, finds, *, model="claude-opus-4-7": no_edits,
    )

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            cli,
            [
                "watch",
                str(overfitting_log),
                "--llm",
                "--apply",
                "--config",
                str(config_file),
            ],
        )
    assert result.exit_code == 0
    assert "nothing to do" in result.output.lower() or "suggested_edits" in result.output.lower()


# --------------------------------------------------------------------------- #
# Persistence                                                                 #
# --------------------------------------------------------------------------- #


def test_watch_apply_persists_to_project_advice_log(
    overfitting_log: Path,
    config_file: Path,
    tmp_path: Path,
    with_api_key: None,
    diagnosis_with_edit: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = ProjectContext.init("test-proj", parent_dir=tmp_path)

    monkeypatch.setattr(
        cli_module,
        "_watch_diagnostician_callable",
        lambda snaps, finds, *, model="claude-opus-4-7": diagnosis_with_edit,
    )

    runner = CliRunner()
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(
            cli,
            [
                "watch",
                str(overfitting_log),
                "--llm",
                "--apply",
                "--config",
                str(config_file),
                "--yes",
            ],
        )
    finally:
        os.chdir(cwd)

    assert result.exit_code == 0, result.output

    entries = [
        json.loads(line)
        for line in (project.path / "advice.log").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert entries
    entry = entries[0]
    assert entry["command"] == "watch"
    assert entry["applied_edits"][0]["key"] == "dropout"
    assert entry["applied_edits"][0]["to"] == 0.3
    assert "backup" in entry
