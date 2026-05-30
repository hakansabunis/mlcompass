"""CLI integration test for the watch command with a TensorBoard source."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pandas as pd
from click.testing import CliRunner

from mlcompass.cli import cli


def _make_fake_event_file(tmp_path: Path) -> Path:
    p = tmp_path / "events.out.tfevents.1716998400.host"
    p.write_bytes(b"")
    return p


def _patched_reader(scalars_df: pd.DataFrame) -> mock.MagicMock:
    reader = mock.MagicMock()
    reader.scalars = scalars_df
    return reader


def test_watch_consumes_tensorboard_directory(tmp_path: Path) -> None:
    _make_fake_event_file(tmp_path)
    # An overfitting-shaped scalar series so the watcher emits one finding.
    df = pd.DataFrame(
        {
            "step": list(range(5)),
            "train_loss": [0.50, 0.40, 0.30, 0.20, 0.10],
            "val_loss": [0.50, 0.48, 0.50, 0.55, 0.60],
        }
    )

    runner = CliRunner()
    with (
        mock.patch("tbparse.SummaryReader", return_value=_patched_reader(df)),
        runner.isolated_filesystem(temp_dir=tmp_path),
    ):
        result = runner.invoke(cli, ["watch", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "tensorboard" in result.output.lower()
    # The overfitting rule should still fire on TB-sourced metrics.
    assert "overfitting" in result.output.lower()


def test_watch_follow_warns_for_tensorboard_source(tmp_path: Path) -> None:
    _make_fake_event_file(tmp_path)
    df = pd.DataFrame({"step": [0], "train_loss": [0.5]})

    runner = CliRunner()
    with (
        mock.patch("tbparse.SummaryReader", return_value=_patched_reader(df)),
        runner.isolated_filesystem(temp_dir=tmp_path),
    ):
        result = runner.invoke(cli, ["watch", str(tmp_path), "--follow"])

    assert result.exit_code == 0
    # The follow loop should refuse but not crash.
    assert "follow" in result.output.lower() and "plain-text" in result.output.lower()
