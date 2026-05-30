"""Tests for the TensorBoard parser and source auto-detection."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from mlcompass.tools.logs import (
    detect_source,
    load_snapshots,
)
from mlcompass.tools.tensorboard import (
    TensorBoardImportError,
    parse_tb_events,
)

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _make_fake_event_file(tmp_path: Path) -> Path:
    """Create an empty file with the TB events.out.tfevents.* name pattern.

    The file content is irrelevant for detect_source; the parser tests
    mock out tbparse so they never read it.
    """
    p = tmp_path / "events.out.tfevents.1716998400.host"
    p.write_bytes(b"")
    return p


def _patched_reader(scalars_df: pd.DataFrame) -> mock.MagicMock:
    """Return a mock object that mimics tbparse.SummaryReader."""
    reader = mock.MagicMock()
    reader.scalars = scalars_df
    return reader


# --------------------------------------------------------------------------- #
# detect_source                                                               #
# --------------------------------------------------------------------------- #


def test_detect_source_plain_text_file(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text("Epoch 0 train_loss=0.5\n", encoding="utf-8")
    assert detect_source(log) == "plain_text"


def test_detect_source_tb_file(tmp_path: Path) -> None:
    event = _make_fake_event_file(tmp_path)
    assert detect_source(event) == "tensorboard"


def test_detect_source_tb_directory(tmp_path: Path) -> None:
    _make_fake_event_file(tmp_path)
    assert detect_source(tmp_path) == "tensorboard"


def test_detect_source_empty_directory(tmp_path: Path) -> None:
    # No TB events and no W&B markers → falls through to plain_text.
    assert detect_source(tmp_path) == "plain_text"


def test_detect_source_wandb_directory(tmp_path: Path) -> None:
    (tmp_path / "wandb-summary.json").write_text("{}", encoding="utf-8")
    assert detect_source(tmp_path) == "wandb"


def test_detect_source_wandb_history_marker(tmp_path: Path) -> None:
    (tmp_path / "wandb-history.jsonl").write_text("", encoding="utf-8")
    assert detect_source(tmp_path) == "wandb"


# --------------------------------------------------------------------------- #
# parse_tb_events                                                             #
# --------------------------------------------------------------------------- #


def test_parse_tb_events_returns_snapshots_per_step(tmp_path: Path) -> None:
    event = _make_fake_event_file(tmp_path)
    df = pd.DataFrame(
        {
            "step": [0, 1, 2],
            "train_loss": [0.8, 0.6, 0.4],
            "val_loss": [0.85, 0.7, 0.55],
        }
    )
    with mock.patch("tbparse.SummaryReader", return_value=_patched_reader(df)):
        snapshots = parse_tb_events(event)

    assert len(snapshots) == 3
    assert snapshots[0].step == 0
    assert snapshots[0].metrics == {"train_loss": 0.8, "val_loss": 0.85}
    assert snapshots[2].metrics["val_loss"] == 0.55


def test_parse_tb_events_picks_up_epoch_column(tmp_path: Path) -> None:
    event = _make_fake_event_file(tmp_path)
    df = pd.DataFrame({"step": [10], "epoch": [3], "train_loss": [0.4]})
    with mock.patch("tbparse.SummaryReader", return_value=_patched_reader(df)):
        snapshots = parse_tb_events(event)

    assert snapshots[0].step == 10
    assert snapshots[0].epoch == 3
    assert snapshots[0].metrics == {"train_loss": 0.4}


def test_parse_tb_events_skips_nan_cells(tmp_path: Path) -> None:
    event = _make_fake_event_file(tmp_path)
    df = pd.DataFrame(
        {
            "step": [0, 1],
            "train_loss": [0.8, None],
            "val_loss": [None, 0.6],
        }
    )
    with mock.patch("tbparse.SummaryReader", return_value=_patched_reader(df)):
        snapshots = parse_tb_events(event)

    assert snapshots[0].metrics == {"train_loss": 0.8}
    assert snapshots[1].metrics == {"val_loss": 0.6}


def test_parse_tb_events_empty_returns_empty_list(tmp_path: Path) -> None:
    event = _make_fake_event_file(tmp_path)
    df = pd.DataFrame(columns=["step", "train_loss"])
    with mock.patch("tbparse.SummaryReader", return_value=_patched_reader(df)):
        snapshots = parse_tb_events(event)
    assert snapshots == []


def test_parse_tb_events_handles_none_scalars(tmp_path: Path) -> None:
    event = _make_fake_event_file(tmp_path)
    reader = mock.MagicMock()
    reader.scalars = None
    with mock.patch("tbparse.SummaryReader", return_value=reader):
        snapshots = parse_tb_events(event)
    assert snapshots == []


def test_parse_tb_events_raises_when_tbparse_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = _make_fake_event_file(tmp_path)

    # Block tbparse from importing for the duration of the test.
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

    def fake_import(name: str, *args, **kwargs):
        if name == "tbparse":
            raise ImportError("simulated missing tbparse")
        return real_import(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "tbparse", None)
    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(TensorBoardImportError, match="tbparse"):
        parse_tb_events(event)


# --------------------------------------------------------------------------- #
# load_snapshots dispatcher                                                   #
# --------------------------------------------------------------------------- #


def test_load_snapshots_plain_text_path(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text(
        "Epoch 0 train_loss=0.5\nEpoch 1 train_loss=0.4\n",
        encoding="utf-8",
    )
    source, snaps = load_snapshots(log)
    assert source == "plain_text"
    assert len(snaps) == 2


def test_load_snapshots_tensorboard_dispatches(tmp_path: Path) -> None:
    event = _make_fake_event_file(tmp_path)
    df = pd.DataFrame({"step": [0], "train_loss": [0.5]})
    with mock.patch("tbparse.SummaryReader", return_value=_patched_reader(df)):
        source, snaps = load_snapshots(event)
    assert source == "tensorboard"
    assert len(snaps) == 1
    assert snaps[0].step == 0
