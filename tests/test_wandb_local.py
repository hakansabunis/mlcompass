"""Tests for the W&B local-cache parser and source auto-detection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from mlcompass.cli import cli
from mlcompass.tools.logs import detect_source, load_snapshots
from mlcompass.tools.wandb_local import (
    WandbRunNotFoundError,
    parse_wandb_run,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _write_history(parent: Path, rows: list[dict]) -> Path:
    """Write a ``wandb-history.jsonl`` file containing ``rows``."""
    parent.mkdir(parents=True, exist_ok=True)
    history = parent / "wandb-history.jsonl"
    history.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    return history


# --------------------------------------------------------------------------- #
# detect_source                                                               #
# --------------------------------------------------------------------------- #


def test_detect_source_wandb_at_top_level(tmp_path: Path) -> None:
    _write_history(tmp_path, [])
    assert detect_source(tmp_path) == "wandb"


def test_detect_source_wandb_under_files_subdir(tmp_path: Path) -> None:
    _write_history(tmp_path / "files", [])
    assert detect_source(tmp_path) == "wandb"


def test_detect_source_wandb_history_file_directly(tmp_path: Path) -> None:
    history = _write_history(tmp_path, [])
    assert detect_source(history) == "wandb"


def test_detect_source_wandb_summary_file_directly(tmp_path: Path) -> None:
    p = tmp_path / "wandb-summary.json"
    p.write_text("{}", encoding="utf-8")
    assert detect_source(p) == "wandb"


# --------------------------------------------------------------------------- #
# parse_wandb_run                                                             #
# --------------------------------------------------------------------------- #


def test_parse_wandb_run_from_history_file(tmp_path: Path) -> None:
    history = _write_history(
        tmp_path,
        [
            {"_step": 0, "train_loss": 0.8, "val_loss": 0.75, "epoch": 0},
            {"_step": 1, "train_loss": 0.6, "val_loss": 0.65, "epoch": 1},
        ],
    )
    snaps = parse_wandb_run(history)
    assert len(snaps) == 2
    assert snaps[0].step == 0
    assert snaps[0].epoch == 0
    assert snaps[0].metrics == {"train_loss": 0.8, "val_loss": 0.75}
    assert snaps[1].metrics["val_loss"] == 0.65


def test_parse_wandb_run_from_run_directory(tmp_path: Path) -> None:
    _write_history(
        tmp_path / "files",
        [{"_step": 0, "train_loss": 0.5}],
    )
    snaps = parse_wandb_run(tmp_path)
    assert len(snaps) == 1
    assert snaps[0].metrics == {"train_loss": 0.5}


def test_parse_wandb_run_skips_wandb_internal_keys(tmp_path: Path) -> None:
    history = _write_history(
        tmp_path,
        [
            {
                "_step": 5,
                "_runtime": 12.4,
                "_timestamp": 1716998400,
                "_wandb": {"foo": "bar"},
                "train_loss": 0.3,
            }
        ],
    )
    snap = parse_wandb_run(history)[0]
    assert snap.step == 5
    assert snap.metrics == {"train_loss": 0.3}  # underscore keys excluded


def test_parse_wandb_run_drops_non_numeric_values(tmp_path: Path) -> None:
    history = _write_history(
        tmp_path,
        [
            {
                "_step": 0,
                "train_loss": 0.5,
                "label": "good",
                "config": {"nested": 1},
                "flag": True,  # bools are intentionally ignored
            }
        ],
    )
    snap = parse_wandb_run(history)[0]
    assert snap.metrics == {"train_loss": 0.5}


def test_parse_wandb_run_skips_invalid_json_lines(tmp_path: Path) -> None:
    history = tmp_path / "wandb-history.jsonl"
    history.write_text(
        '{"_step": 0, "train_loss": 0.5}\n'
        "not a json line at all\n"
        '{"_step": 1, "train_loss": 0.4}\n',
        encoding="utf-8",
    )
    snaps = parse_wandb_run(history)
    assert [s.step for s in snaps] == [0, 1]


def test_parse_wandb_run_skips_empty_lines(tmp_path: Path) -> None:
    history = tmp_path / "wandb-history.jsonl"
    history.write_text(
        '{"_step": 0, "train_loss": 0.5}\n\n\n{"_step": 1, "train_loss": 0.4}\n',
        encoding="utf-8",
    )
    snaps = parse_wandb_run(history)
    assert len(snaps) == 2


def test_parse_wandb_run_drops_rows_with_no_signal(tmp_path: Path) -> None:
    history = _write_history(
        tmp_path,
        [
            {"label": "first"},  # nothing numeric, no step, no epoch
            {"_step": 0, "train_loss": 0.5},
        ],
    )
    snaps = parse_wandb_run(history)
    assert len(snaps) == 1
    assert snaps[0].step == 0


def test_parse_wandb_run_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(WandbRunNotFoundError):
        parse_wandb_run(tmp_path)


# --------------------------------------------------------------------------- #
# load_snapshots dispatcher                                                   #
# --------------------------------------------------------------------------- #


def test_load_snapshots_dispatches_to_wandb(tmp_path: Path) -> None:
    _write_history(
        tmp_path / "files",
        [
            {"_step": 0, "train_loss": 0.5, "val_loss": 0.6},
            {"_step": 1, "train_loss": 0.4, "val_loss": 0.55},
        ],
    )
    source, snaps = load_snapshots(tmp_path)
    assert source == "wandb"
    assert len(snaps) == 2
    assert snaps[1].metrics["val_loss"] == 0.55


# --------------------------------------------------------------------------- #
# CLI integration                                                             #
# --------------------------------------------------------------------------- #


def test_watch_consumes_wandb_run_directory(tmp_path: Path) -> None:
    # Build an overfitting-shaped W&B run.
    rows = [
        {
            "_step": i,
            "epoch": i,
            "train_loss": 0.5 - i * 0.10,
            "val_loss": 0.50 + i * 0.025,
        }
        for i in range(5)
    ]
    _write_history(tmp_path / "files", rows)

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["watch", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "w&b" in result.output.lower() or "wandb" in result.output.lower()
    assert "overfitting" in result.output.lower()


def test_watch_follow_warns_for_wandb_source(tmp_path: Path) -> None:
    _write_history(tmp_path, [{"_step": 0, "train_loss": 0.5}])

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["watch", str(tmp_path), "--follow"])

    assert result.exit_code == 0
    assert "follow" in result.output.lower() and "plain-text" in result.output.lower()
