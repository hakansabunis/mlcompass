"""Tests for ``tools.logs`` (plain-text log parser)."""

from __future__ import annotations

import math
from pathlib import Path

from mlcompass.tools.logs import (
    MetricSnapshot,
    has_invalid_loss,
    merge_consecutive_same_epoch,
    parse_log_file,
    parse_log_line,
    parse_log_text,
)

# --------------------------------------------------------------------------- #
# parse_log_line                                                              #
# --------------------------------------------------------------------------- #


def test_parse_simple_kv_line() -> None:
    snap = parse_log_line("Epoch 5 train_loss=0.456 val_loss=0.523")
    assert snap is not None
    assert snap.epoch == 5
    assert snap.metrics == {"train_loss": 0.456, "val_loss": 0.523}


def test_parse_keras_style_line() -> None:
    snap = parse_log_line("Epoch 5/100 - loss: 0.456 - val_loss: 0.523 - val_acc: 0.85")
    assert snap is not None
    assert snap.epoch == 5
    assert snap.metrics == {"loss": 0.456, "val_loss": 0.523, "val_acc": 0.85}


def test_parse_step_only_line() -> None:
    snap = parse_log_line("step=1234 loss=0.4")
    assert snap is not None
    assert snap.step == 1234
    assert snap.metrics == {"loss": 0.4}


def test_parse_scientific_notation() -> None:
    snap = parse_log_line("Epoch 1 lr=1e-4 val_loss=2.5e-3")
    assert snap is not None
    assert snap.metrics["lr"] == 1e-4
    assert snap.metrics["val_loss"] == 2.5e-3


def test_parse_nan_value() -> None:
    snap = parse_log_line("Epoch 14 train_loss=nan")
    assert snap is not None
    assert math.isnan(snap.metrics["train_loss"])


def test_parse_inf_value() -> None:
    snap = parse_log_line("Epoch 14 train_loss=inf")
    assert snap is not None
    assert math.isinf(snap.metrics["train_loss"])


def test_parse_negative_value() -> None:
    snap = parse_log_line("Epoch 1 reward=-0.5 score=-2.7e-3")
    assert snap is not None
    assert snap.metrics["reward"] == -0.5
    assert snap.metrics["score"] == -2.7e-3


def test_blank_line_returns_none() -> None:
    assert parse_log_line("") is None
    assert parse_log_line("   ") is None


def test_unrelated_line_returns_none() -> None:
    assert parse_log_line("loading checkpoint /tmp/model.pt") is None


def test_epoch_only_line_is_kept() -> None:
    # Even without metrics, an epoch marker is useful context.
    snap = parse_log_line("Epoch 5/100")
    assert snap is not None
    assert snap.epoch == 5
    assert snap.metrics == {}


def test_epoch_marker_is_not_a_metric() -> None:
    snap = parse_log_line("Epoch=5 loss=0.4")
    assert snap is not None
    assert snap.epoch == 5
    assert "epoch" not in snap.metrics


def test_single_letter_keys_are_ignored() -> None:
    # Loop variables like "i=5" shouldn't pollute the metrics.
    snap = parse_log_line("i=5 j=2 loss=0.4")
    assert snap is not None
    assert snap.metrics == {"loss": 0.4}


# --------------------------------------------------------------------------- #
# parse_log_text                                                              #
# --------------------------------------------------------------------------- #


def test_parse_log_text_multiple_lines() -> None:
    text = """
Starting training
Epoch 1 train_loss=0.8 val_loss=0.75
Epoch 2 train_loss=0.6 val_loss=0.7
Saving checkpoint to /tmp/ckpt.pt
Epoch 3 train_loss=0.5 val_loss=0.65
"""
    snapshots = parse_log_text(text)
    assert len(snapshots) == 3
    assert snapshots[0].epoch == 1
    assert snapshots[-1].epoch == 3


# --------------------------------------------------------------------------- #
# parse_log_file                                                              #
# --------------------------------------------------------------------------- #


def test_parse_log_file_reads_disk(tmp_path: Path) -> None:
    log_path = tmp_path / "train.log"
    log_path.write_text(
        "Epoch 1 train_loss=0.8 val_loss=0.9\nEpoch 2 train_loss=0.6 val_loss=0.8\n",
        encoding="utf-8",
    )
    snapshots = parse_log_file(log_path)
    assert len(snapshots) == 2


# --------------------------------------------------------------------------- #
# merge_consecutive_same_epoch                                                #
# --------------------------------------------------------------------------- #


def test_merge_collapses_same_epoch() -> None:
    snapshots = [
        MetricSnapshot(epoch=1, metrics={"train_loss": 0.8}),
        MetricSnapshot(epoch=1, metrics={"val_loss": 0.9}),
        MetricSnapshot(epoch=2, metrics={"train_loss": 0.6}),
    ]
    merged = merge_consecutive_same_epoch(snapshots)
    assert len(merged) == 2
    assert merged[0].metrics == {"train_loss": 0.8, "val_loss": 0.9}
    assert merged[1].metrics == {"train_loss": 0.6}


def test_merge_does_not_merge_across_other_epochs() -> None:
    snapshots = [
        MetricSnapshot(epoch=1, metrics={"train_loss": 0.8}),
        MetricSnapshot(epoch=2, metrics={"train_loss": 0.6}),
        MetricSnapshot(epoch=1, metrics={"val_loss": 0.9}),
    ]
    merged = merge_consecutive_same_epoch(snapshots)
    assert len(merged) == 3


# --------------------------------------------------------------------------- #
# has_invalid_loss                                                            #
# --------------------------------------------------------------------------- #


def test_has_invalid_loss_with_nan() -> None:
    snap = MetricSnapshot(metrics={"train_loss": float("nan")})
    assert has_invalid_loss(snap)


def test_has_invalid_loss_with_inf() -> None:
    snap = MetricSnapshot(metrics={"val_loss": float("inf")})
    assert has_invalid_loss(snap)


def test_has_invalid_loss_ignores_non_loss_nans() -> None:
    snap = MetricSnapshot(metrics={"weird_metric": float("nan"), "train_loss": 0.5})
    assert not has_invalid_loss(snap)


def test_has_invalid_loss_is_false_for_clean_metrics() -> None:
    snap = MetricSnapshot(metrics={"train_loss": 0.3, "val_loss": 0.4})
    assert not has_invalid_loss(snap)
