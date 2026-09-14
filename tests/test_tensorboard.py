"""Tests for the TensorBoard parser and source auto-detection."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from mlcompass.tools.anomaly import detect_nan
from mlcompass.tools.logs import (
    detect_source,
    has_invalid_loss,
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


def _write_event_file(directory: Path, records: list[tuple[int, str, float]]) -> Path:
    """Write a *real* TensorBoard event file containing ``(step, tag, value)``.

    Mocking ``tbparse.SummaryReader`` cannot express the distinction this
    module has to get right — in a pivoted frame a NaN *value* and an
    absent *cell* are the same object — so these tests go through
    tensorboard's own writer and tbparse's own reader end to end.
    """
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    directory.mkdir(parents=True, exist_ok=True)
    writer = EventFileWriter(str(directory))
    for step, tag, value in records:
        writer.add_event(
            Event(
                step=step,
                wall_time=1716998400.0 + step,
                summary=Summary(value=[Summary.Value(tag=tag, simple_value=value)]),
            )
        )
    writer.flush()
    writer.close()
    return next(directory.glob("events.out.tfevents.*"))


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
    event = _write_event_file(
        tmp_path / "run",
        [
            (0, "train_loss", 0.8),
            (0, "val_loss", 0.85),
            (1, "train_loss", 0.6),
            (1, "val_loss", 0.7),
            (2, "train_loss", 0.4),
            (2, "val_loss", 0.55),
        ],
    )
    snapshots = parse_tb_events(event)

    assert len(snapshots) == 3
    assert [s.step for s in snapshots] == [0, 1, 2]
    assert sorted(snapshots[0].metrics) == ["train_loss", "val_loss"]
    assert snapshots[0].metrics["train_loss"] == pytest.approx(0.8)
    assert snapshots[0].metrics["val_loss"] == pytest.approx(0.85)
    assert snapshots[2].metrics["val_loss"] == pytest.approx(0.55)


def test_parse_tb_events_picks_up_epoch_column(tmp_path: Path) -> None:
    event = _write_event_file(
        tmp_path / "run",
        [(10, "epoch", 3), (10, "train_loss", 0.4)],
    )
    snapshots = parse_tb_events(event)

    assert snapshots[0].step == 10
    assert snapshots[0].epoch == 3
    assert list(snapshots[0].metrics) == ["train_loss"]
    assert snapshots[0].metrics["train_loss"] == pytest.approx(0.4)


def test_parse_tb_events_skips_missing_cells(tmp_path: Path) -> None:
    # A tag that was never written at this step is genuinely absent — it
    # must not appear in the snapshot's metrics. Written through the real
    # writer, because a mocked pivoted frame cannot express "absent" as
    # anything other than NaN, which is the very conflation under test.
    event = _write_event_file(
        tmp_path / "run",
        [(0, "train_loss", 0.8), (1, "val_loss", 0.6)],
    )
    snapshots = parse_tb_events(event)

    # ``simple_value`` is a float32 on the wire, so compare approximately.
    assert list(snapshots[0].metrics) == ["train_loss"]
    assert snapshots[0].metrics["train_loss"] == pytest.approx(0.8)
    assert list(snapshots[1].metrics) == ["val_loss"]
    assert snapshots[1].metrics["val_loss"] == pytest.approx(0.6)


def test_parse_tb_events_keeps_nan_loss_value(tmp_path: Path) -> None:
    # A loss that *went* NaN is a value, not a missing cell. Dropping it
    # leaves detect_nan with an empty metrics dict and the only
    # error-severity rule cannot fire on a TensorBoard source at all.
    event = _write_event_file(
        tmp_path / "run",
        [(0, "train_loss", 0.8), (1, "train_loss", float("nan"))],
    )
    snapshots = parse_tb_events(event)

    assert len(snapshots) == 2
    assert "train_loss" in snapshots[1].metrics
    assert math.isnan(snapshots[1].metrics["train_loss"])
    assert has_invalid_loss(snapshots[1])
    findings = detect_nan(snapshots)
    assert [f.rule_id for f in findings] == ["nan"]
    assert findings[0].severity == "error"


def test_parse_tb_events_keeps_inf_loss_value(tmp_path: Path) -> None:
    # The pre-fix asymmetry: ``_is_missing`` tested isnan but not isinf,
    # so Inf already survived where NaN did not. Pinned so the fix keeps
    # both, rather than making the two consistent in the wrong direction.
    event = _write_event_file(
        tmp_path / "run",
        [(0, "train_loss", 0.8), (1, "train_loss", float("inf"))],
    )
    snapshots = parse_tb_events(event)

    assert math.isinf(snapshots[1].metrics["train_loss"])
    assert [f.rule_id for f in detect_nan(snapshots)] == ["nan"]


def test_parse_tb_events_empty_returns_empty_list(tmp_path: Path) -> None:
    # Long-form (un-pivoted) columns, matching what parse_tb_events now asks
    # tbparse for; an empty frame short-circuits before any column is read.
    event = _make_fake_event_file(tmp_path)
    df = pd.DataFrame(columns=["step", "tag", "value"])
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
    event = _write_event_file(tmp_path / "run", [(0, "train_loss", 0.5)])
    source, snaps = load_snapshots(event)
    assert source == "tensorboard"
    assert len(snaps) == 1
    assert snaps[0].step == 0
    assert snaps[0].metrics["train_loss"] == pytest.approx(0.5)
