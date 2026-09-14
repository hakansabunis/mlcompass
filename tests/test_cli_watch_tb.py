"""CLI integration test for the watch command with a TensorBoard source."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from mlcompass.cli import cli


def _write_event_file(directory: Path, records: list[tuple[int, str, float]]) -> Path:
    """Write a *real* TensorBoard event file into ``directory``.

    Deliberately not a mocked ``tbparse.SummaryReader``: the mock has to
    commit to a frame shape, and a mock committed to the pivoted shape is
    what let a NaN loss silently vanish between the writer and the rules.
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
    return directory


def test_watch_consumes_tensorboard_directory(tmp_path: Path) -> None:
    # An overfitting-shaped scalar series so the watcher emits one finding.
    train = [0.50, 0.40, 0.30, 0.20, 0.10]
    val = [0.50, 0.48, 0.50, 0.55, 0.60]
    _write_event_file(
        tmp_path,
        [(i, "train_loss", train[i]) for i in range(5)]
        + [(i, "val_loss", val[i]) for i in range(5)],
    )

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["watch", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "tensorboard" in result.output.lower()
    # The overfitting rule should still fire on TB-sourced metrics.
    assert "overfitting" in result.output.lower()


def test_watch_reports_nan_from_tensorboard_source(tmp_path: Path) -> None:
    # The README advertises the `nan` rule for TensorBoard sources. End to
    # end, through the real writer and the real reader, it has to arrive.
    _write_event_file(
        tmp_path,
        [(0, "train_loss", 0.5), (1, "train_loss", 0.4), (2, "train_loss", float("nan"))],
    )

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["watch", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "nan" in result.output.lower()
    assert "no anomalies detected" not in result.output.lower()


def test_watch_follow_warns_for_tensorboard_source(tmp_path: Path) -> None:
    _write_event_file(tmp_path, [(0, "train_loss", 0.5)])

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["watch", str(tmp_path), "--follow"])

    assert result.exit_code == 0
    # The follow loop should refuse but not crash.
    assert "follow" in result.output.lower() and "plain-text" in result.output.lower()
