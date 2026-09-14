"""Tests for ``tools.logs`` (plain-text log parser)."""

from __future__ import annotations

import math
from pathlib import Path

from mlcompass.tools.anomaly import detect_overfitting
from mlcompass.tools.logs import (
    MetricSnapshot,
    has_invalid_loss,
    is_loss_key,
    merge_consecutive_same_epoch,
    normalise_metric_key,
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


def test_slash_separated_keys_do_not_collapse() -> None:
    # The W&B default naming, both series on one line. The key group could
    # not cross "/", so both names collapsed to "loss" and the train value
    # was silently overwritten by the val value — after which _train_loss
    # returned the val loss to every detector downstream.
    snap = parse_log_line("train/loss: 0.42 val/loss: 0.55")
    assert snap is not None
    assert snap.metrics == {"train/loss": 0.42, "val/loss": 0.55}


def test_pytorch_tutorial_keys_do_not_collapse() -> None:
    snap = parse_log_line("Loss/train: 0.42 Loss/val: 0.55")
    assert snap is not None
    assert snap.metrics == {"Loss/train": 0.42, "Loss/val": 0.55}


def test_dotted_keys_do_not_collapse() -> None:
    snap = parse_log_line("train.loss=0.42 val.loss=0.55")
    assert snap is not None
    assert snap.metrics == {"train.loss": 0.42, "val.loss": 0.55}


def test_dashed_keys_do_not_collapse() -> None:
    snap = parse_log_line("train-loss=0.42 val-loss=0.55")
    assert snap is not None
    assert snap.metrics == {"train-loss": 0.42, "val-loss": 0.55}


def test_separated_keys_reach_the_right_detector_series() -> None:
    # The consequence the collapse actually had: an overfitting signature
    # written in W&B naming looked like a single flat series.
    text = "\n".join(
        f"Epoch {i} train/loss: {t} val/loss: {v}"
        for i, (t, v) in enumerate([(0.50, 0.50), (0.40, 0.55), (0.30, 0.60), (0.10, 0.70)])
    )
    snapshots = parse_log_text(text)
    assert [s.metrics["train/loss"] for s in snapshots] == [0.50, 0.40, 0.30, 0.10]
    assert [s.metrics["val/loss"] for s in snapshots] == [0.50, 0.55, 0.60, 0.70]
    assert [f.rule_id for f in detect_overfitting(snapshots)] == ["overfitting"]


def test_thousands_separator_is_not_truncated() -> None:
    # ``f"{loss:,.2f}"`` is an ordinary format string. The value pattern
    # stopped at the comma, so 1,000.0 parsed as 1.0 — wrong by 1000x,
    # silently, feeding every threshold downstream.
    snap = parse_log_line("Epoch 1 train_loss=1,000.0")
    assert snap is not None
    assert snap.metrics["train_loss"] == 1000.0


def test_multi_group_thousands_separator_is_not_truncated() -> None:
    snap = parse_log_line("Epoch 1 train_loss=12,345.67 total_tokens=1,234,567")
    assert snap is not None
    assert snap.metrics["train_loss"] == 12345.67
    assert snap.metrics["total_tokens"] == 1234567.0


def test_uppercase_nan_is_parsed() -> None:
    # NumPy and several loggers print NAN; the pattern matched nan/NaN
    # only, so the pair was dropped entirely and the NaN rule never saw it.
    snap = parse_log_line("Epoch 14 train_loss=NAN")
    assert snap is not None
    assert math.isnan(snap.metrics["train_loss"])


def test_long_form_infinity_is_parsed() -> None:
    snap = parse_log_line("Epoch 14 train_loss=-Infinity val_loss=Infinity")
    assert snap is not None
    assert math.isinf(snap.metrics["train_loss"])
    assert snap.metrics["train_loss"] < 0
    assert math.isinf(snap.metrics["val_loss"])
    assert snap.metrics["val_loss"] > 0


def test_infinity_spelling_variants_are_parsed() -> None:
    for spelling, sign in (
        ("INF", 1),
        ("-INF", -1),
        ("Infinity", 1),
        ("-inf", -1),
        ("NaN", 0),
        ("nan", 0),
    ):
        snap = parse_log_line(f"Epoch 1 train_loss={spelling}")
        assert snap is not None, spelling
        value = snap.metrics["train_loss"]
        if sign == 0:
            assert math.isnan(value), spelling
        else:
            assert math.isinf(value) and (value > 0) == (sign > 0), spelling


def test_bad_values_still_dropped() -> None:
    # A comma that is a field delimiter, not a thousands separator, must
    # not glue two metrics together.
    snap = parse_log_line("Epoch 1 train_loss=0.42, val_loss=0.55")
    assert snap is not None
    assert snap.metrics == {"train_loss": 0.42, "val_loss": 0.55}


def test_step_only_line_is_not_a_snapshot() -> None:
    # The decision, made explicit: a line carrying a step marker but no
    # measurement is not a snapshot. It was already dropped, but by a
    # *second* guard after the first one had admitted it — so the
    # "step_match is None" term in the first guard was dead code, and the
    # intent was unreadable from either one.
    assert parse_log_line("step=1234") is None
    assert parse_log_line("iter 4000") is None


def test_nanogpt_style_line_yields_no_snapshot() -> None:
    # nanoGPT prints whitespace-separated pairs. The parser requires a
    # ":" or "=" between key and value, so it genuinely cannot read this
    # line — and must say so by producing nothing, not by producing an
    # empty-but-present snapshot that looks like successfully-read data.
    assert parse_log_line("iter 0: loss 4.0000, time 12.30ms") is None
    assert parse_log_text("iter 0: loss 4.0000, time 12.30ms\n") == []


def test_epoch_only_line_is_still_a_snapshot() -> None:
    # Unchanged: an epoch marker alone is useful context for the report.
    snap = parse_log_line("Epoch 5/100")
    assert snap is not None
    assert snap.epoch == 5


def test_path_like_line_is_still_not_a_metric() -> None:
    # Widening the key charset must not start matching file paths.
    assert parse_log_line("loading checkpoint /tmp/model.pt") is None
    assert parse_log_line("Saving to ./runs/exp-1/model.pt") is None


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


# --------------------------------------------------------------------------- #
# normalise_metric_key                                                        #
# --------------------------------------------------------------------------- #


def test_normalise_metric_key_folds_separators_and_case() -> None:
    # One spelling for every separator convention in the wild, so the
    # detectors' key vocabulary and has_invalid_loss can share a notion of
    # what a metric is called.
    for spelling in (
        "train_loss",
        "train/loss",
        "train.loss",
        "train-loss",
        "train loss",
        "Train Loss",
        "TRAIN_LOSS",
        "train__loss",
    ):
        assert normalise_metric_key(spelling) == "train_loss", spelling


def test_normalise_metric_key_preserves_word_order() -> None:
    # Order is not normalised away — ``Loss/train`` is a different string
    # from ``train/loss``. Matching them to the same metric is the key
    # vocabulary's job, not this function's.
    assert normalise_metric_key("Loss/train") == "loss_train"


def test_normalise_metric_key_strips_edge_separators() -> None:
    assert normalise_metric_key("  /train/loss/  ") == "train_loss"


def test_is_loss_key_matches_separator_variants() -> None:
    for spelling in ("train_loss", "Loss/train", "val/loss", "Total Loss"):
        assert is_loss_key(spelling), spelling


def test_is_loss_key_stays_a_substring_test() -> None:
    # Unchanged semantics: a plural or suffixed loss name still counts.
    assert is_loss_key("total_losses")
    assert is_loss_key("mse_loss_value")
    assert not is_loss_key("accuracy")
