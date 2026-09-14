"""Tests for ``tools.anomaly`` (plateau / overfit / NaN / divergence)."""

from __future__ import annotations

from mlcompass.tools.anomaly import (
    detect_divergence,
    detect_nan,
    detect_overfitting,
    detect_plateau,
    run_all_detectors,
)
from mlcompass.tools.logs import MetricSnapshot


def _snap(epoch: int, **metrics: float) -> MetricSnapshot:
    return MetricSnapshot(epoch=epoch, metrics=dict(metrics))


def _snap_named(epoch: int, metrics: dict[str, float]) -> MetricSnapshot:
    """Like ``_snap`` but for metric names that aren't Python identifiers."""
    return MetricSnapshot(epoch=epoch, metrics=dict(metrics))


# --------------------------------------------------------------------------- #
# detect_nan                                                                  #
# --------------------------------------------------------------------------- #


def test_detect_nan_fires_on_nan_loss() -> None:
    metrics = [
        _snap(0, train_loss=0.5),
        _snap(1, train_loss=float("nan")),
    ]
    findings = detect_nan(metrics)
    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert findings[0].rule_id == "nan"
    assert findings[0].epoch == 1


def test_detect_nan_fires_on_inf_loss() -> None:
    metrics = [_snap(0, val_loss=float("inf"))]
    findings = detect_nan(metrics)
    assert len(findings) == 1


def test_detect_nan_silent_on_clean_history() -> None:
    metrics = [_snap(0, train_loss=0.5), _snap(1, train_loss=0.4)]
    assert detect_nan(metrics) == []


def test_detect_nan_ignores_nan_in_non_loss_metric() -> None:
    metrics = [_snap(0, train_loss=0.4, weird_metric=float("nan"))]
    assert detect_nan(metrics) == []


def test_detect_nan_silent_when_nan_recovered() -> None:
    # A NaN followed by a real, finite loss is a run that recovered. The
    # rule stays quiet — this is the case the old latest-only design
    # reasoned about, and it still holds.
    metrics = [
        _snap(0, train_loss=float("nan")),
        _snap(1, train_loss=0.4),
    ]
    assert detect_nan(metrics) == []


def test_detect_nan_fires_when_trailing_summary_line_carries_no_loss() -> None:
    # The case latest-only got wrong: the run never recovered, and the
    # script printed a summary line afterwards. ``Total time: 412.30``
    # parses as a metric-free snapshot, and inspecting only metrics[-1]
    # silenced the only error-severity rule in the suite.
    nan_run = [_snap(i, train_loss=float("nan")) for i in range(5)]
    tail = MetricSnapshot(metrics={"time": 412.30}, source_line="Total time: 412.30")

    assert len(detect_nan(nan_run)) == 1, "sanity: fires without the tail"
    findings = detect_nan([*nan_run, tail])
    assert len(findings) == 1
    assert findings[0].rule_id == "nan"
    assert findings[0].severity == "error"
    # Reported at the epoch the loss actually went bad, not the tail.
    assert findings[0].epoch == 0


def test_detect_nan_fires_under_every_confirmed_silencing_tail() -> None:
    # Each of these tails was confirmed to silence the rule: they parse to
    # a snapshot with metrics but no loss-like key.
    nan_run = [_snap(i, train_loss=float("nan")) for i in range(5)]
    tails = [
        MetricSnapshot(metrics={"time": 412.30}, source_line="Total time: 412.30"),
        MetricSnapshot(metrics={"Elapsed": 93.5}, source_line="Elapsed: 93.5"),
        MetricSnapshot(metrics={"best_val": 0.4321}, source_line="best_val: 0.4321"),
        MetricSnapshot(metrics={"lr": 0.001}, source_line="lr: 0.001"),
    ]
    for tail in tails:
        findings = detect_nan([*nan_run, tail])
        assert len(findings) == 1, f"silenced by tail {tail.source_line!r}"
        assert findings[0].rule_id == "nan"


def test_detect_nan_reports_first_nan_epoch() -> None:
    # Scanning history means reporting where the run actually broke.
    metrics = [
        _snap(0, train_loss=0.5),
        _snap(1, train_loss=0.4),
        _snap(2, train_loss=float("nan")),
        _snap(3, train_loss=float("nan")),
    ]
    findings = detect_nan(metrics)
    assert len(findings) == 1
    assert findings[0].epoch == 2


def test_detect_nan_fires_once_not_per_bad_snapshot() -> None:
    # Six NaN epochs are one diagnosis, not six findings.
    metrics = [_snap(i, train_loss=float("nan")) for i in range(6)]
    assert len(detect_nan(metrics)) == 1


# --------------------------------------------------------------------------- #
# detect_divergence                                                           #
# --------------------------------------------------------------------------- #


def test_detect_divergence_fires_on_10x_jump() -> None:
    metrics = [_snap(0, train_loss=0.1), _snap(1, train_loss=2.0)]
    findings = detect_divergence(metrics)
    assert len(findings) == 1
    assert findings[0].severity == "error"


def test_detect_divergence_silent_for_normal_changes() -> None:
    metrics = [_snap(0, train_loss=0.3), _snap(1, train_loss=0.45)]
    assert detect_divergence(metrics) == []


def test_detect_divergence_requires_two_points() -> None:
    assert detect_divergence([_snap(0, train_loss=1.0)]) == []


def test_detect_divergence_handles_zero_prev_loss() -> None:
    metrics = [_snap(0, train_loss=0.0), _snap(1, train_loss=5.0)]
    # Avoid div-by-zero by reporting nothing rather than crashing.
    assert detect_divergence(metrics) == []


def test_detect_divergence_finds_a_mid_history_jump() -> None:
    # A 167x blow-up at epoch 4 of 8. Comparing only the last two points
    # read this run as clean: by the end the loss is high but no longer
    # jumping, which is exactly what a diverged run looks like.
    metrics = [
        _snap(0, train_loss=0.30),
        _snap(1, train_loss=0.24),
        _snap(2, train_loss=0.21),
        _snap(3, train_loss=0.18),
        _snap(4, train_loss=30.06),  # 167x
        _snap(5, train_loss=31.0),
        _snap(6, train_loss=32.5),
        _snap(7, train_loss=33.0),
    ]
    findings = detect_divergence(metrics)
    assert len(findings) == 1
    assert findings[0].rule_id == "divergence"
    assert findings[0].severity == "error"
    # The Finding already carries an epoch field; report where it jumped.
    assert findings[0].epoch == 4
    assert "167" in findings[0].message


def test_detect_divergence_reports_the_first_jump_when_several() -> None:
    metrics = [
        _snap(0, train_loss=0.1),
        _snap(1, train_loss=5.0),  # 50x, epoch 1
        _snap(2, train_loss=6.0),
        _snap(3, train_loss=600.0),  # 100x, epoch 3
    ]
    findings = detect_divergence(metrics)
    assert len(findings) == 1
    assert findings[0].epoch == 1


def test_detect_divergence_fires_on_negative_losses() -> None:
    # Log-likelihood, contrastive and ELBO losses are routinely negative.
    # Exempting non-positive losses made this run read clean.
    metrics = [
        _snap(0, train_loss=-2.0),
        _snap(1, train_loss=-2.1),
        _snap(2, train_loss=-2.2),
        _snap(3, train_loss=-500.0),
    ]
    findings = detect_divergence(metrics)
    assert len(findings) == 1
    assert findings[0].rule_id == "divergence"
    assert findings[0].epoch == 3


def test_detect_divergence_silent_when_negative_loss_improves() -> None:
    # A negative loss growing *more* negative in the normal way is the
    # healthy direction for these objectives and must stay quiet.
    metrics = [
        _snap(0, train_loss=-2.0),
        _snap(1, train_loss=-2.4),
        _snap(2, train_loss=-2.9),
        _snap(3, train_loss=-3.3),
    ]
    assert detect_divergence(metrics) == []


def test_detect_divergence_fires_when_negative_loss_crosses_to_positive() -> None:
    # -2.0 -> 400.0 is a blow-up whatever the sign convention.
    metrics = [_snap(0, train_loss=-2.0), _snap(1, train_loss=400.0)]
    findings = detect_divergence(metrics)
    assert len(findings) == 1
    assert findings[0].epoch == 1


def test_detect_divergence_still_silent_on_a_healthy_run() -> None:
    metrics = [_snap(i, train_loss=0.5 - 0.05 * i) for i in range(8)]
    assert detect_divergence(metrics) == []


def test_detect_divergence_does_not_fire_on_small_absolute_noise() -> None:
    # A loss legitimately near zero can produce a huge ratio from noise
    # alone; the rule must not turn that into an error-severity finding.
    metrics = [
        _snap(0, train_loss=1e-9),
        _snap(1, train_loss=1e-7),
        _snap(2, train_loss=1e-8),
    ]
    assert detect_divergence(metrics) == []


# --------------------------------------------------------------------------- #
# detect_plateau                                                              #
# --------------------------------------------------------------------------- #


def test_detect_plateau_fires_on_flat_val_loss() -> None:
    metrics = [_snap(i, train_loss=0.3, val_loss=0.5) for i in range(8)]
    findings = detect_plateau(metrics)
    assert len(findings) == 1
    assert findings[0].rule_id == "plateau"


def test_detect_plateau_silent_when_loss_still_moving() -> None:
    metrics = [_snap(i, val_loss=0.5 - 0.05 * i) for i in range(8)]
    assert detect_plateau(metrics) == []


def test_detect_plateau_requires_window_length() -> None:
    metrics = [_snap(i, val_loss=0.5) for i in range(2)]
    assert detect_plateau(metrics, window=5) == []


def test_detect_plateau_falls_back_to_train_loss() -> None:
    # No val_loss anywhere — train_loss should drive the rule.
    metrics = [_snap(i, train_loss=0.5) for i in range(8)]
    findings = detect_plateau(metrics)
    assert len(findings) == 1


# --------------------------------------------------------------------------- #
# detect_overfitting                                                          #
# --------------------------------------------------------------------------- #


def test_detect_overfitting_fires_on_classic_signature() -> None:
    # train_loss decreases, val_loss increases, big gap at the end
    metrics = [
        _snap(0, train_loss=0.5, val_loss=0.5),
        _snap(1, train_loss=0.4, val_loss=0.48),
        _snap(2, train_loss=0.3, val_loss=0.50),
        _snap(3, train_loss=0.2, val_loss=0.55),
        _snap(4, train_loss=0.1, val_loss=0.60),
    ]
    findings = detect_overfitting(metrics)
    assert len(findings) == 1
    assert findings[0].rule_id == "overfitting"


def test_detect_overfitting_silent_when_both_decrease() -> None:
    metrics = [
        _snap(0, train_loss=0.5, val_loss=0.6),
        _snap(1, train_loss=0.4, val_loss=0.5),
        _snap(2, train_loss=0.3, val_loss=0.4),
        _snap(3, train_loss=0.2, val_loss=0.3),
    ]
    assert detect_overfitting(metrics) == []


def test_detect_overfitting_silent_when_gap_tiny() -> None:
    # Train down, val up — but gap still negligible.
    metrics = [
        _snap(0, train_loss=0.50, val_loss=0.51),
        _snap(1, train_loss=0.49, val_loss=0.51),
        _snap(2, train_loss=0.48, val_loss=0.51),
        _snap(3, train_loss=0.47, val_loss=0.515),
    ]
    assert detect_overfitting(metrics) == []


def test_detect_overfitting_requires_enough_history() -> None:
    metrics = [
        _snap(0, train_loss=0.4, val_loss=0.5),
        _snap(1, train_loss=0.3, val_loss=0.6),
    ]
    assert detect_overfitting(metrics, window=3) == []


# --------------------------------------------------------------------------- #
# run_all_detectors                                                           #
# --------------------------------------------------------------------------- #


def test_run_all_detectors_aggregates() -> None:
    # Plateau-then-NaN: should surface both detectors.
    metrics = [_snap(i, train_loss=0.3, val_loss=0.5) for i in range(8)]
    metrics.append(_snap(8, train_loss=float("nan"), val_loss=0.5))
    findings = run_all_detectors(metrics)
    rule_ids = [f.rule_id for f in findings]
    assert "nan" in rule_ids
    # NaN should be reported first (it's the most urgent).
    assert findings[0].rule_id == "nan"


def test_run_all_detectors_empty_history() -> None:
    assert run_all_detectors([]) == []


# --------------------------------------------------------------------------- #
# Separator variants in loss key names                                        #
# --------------------------------------------------------------------------- #
#
# ``Loss/train`` is what PyTorch's own tutorial writes to TensorBoard, and
# ``train/loss`` is the W&B default. Both are real, common spellings. The
# key tuples matched them exactly and so matched neither, which left
# plateau, divergence and overfitting blind to an entire naming
# convention — while ``has_invalid_loss``, which tests for the substring,
# saw them fine. The two notions of "a loss metric" have to agree.


# A flat series (plateau), a 20x jump (divergence), and an overfitting
# signature, each expressed in one of the naming conventions.
_FLAT = [0.5000, 0.5001, 0.5000, 0.5001, 0.5000, 0.5000]
_DIVERGING = [0.10, 0.11, 0.12, 2.40]
_TRAIN_FALLING = [0.50, 0.40, 0.30, 0.10]
_VAL_RISING = [0.50, 0.55, 0.60, 0.70]


def test_detect_plateau_sees_pytorch_tutorial_naming() -> None:
    metrics = [_snap_named(i, {"Loss/train": v}) for i, v in enumerate(_FLAT)]
    findings = detect_plateau(metrics)
    assert [f.rule_id for f in findings] == ["plateau"]


def test_detect_plateau_sees_wandb_default_naming() -> None:
    metrics = [_snap_named(i, {"train/loss": v}) for i, v in enumerate(_FLAT)]
    findings = detect_plateau(metrics)
    assert [f.rule_id for f in findings] == ["plateau"]


def test_detect_divergence_sees_pytorch_tutorial_naming() -> None:
    metrics = [_snap_named(i, {"Loss/train": v}) for i, v in enumerate(_DIVERGING)]
    findings = detect_divergence(metrics)
    assert [f.rule_id for f in findings] == ["divergence"]


def test_detect_divergence_sees_wandb_default_naming() -> None:
    metrics = [_snap_named(i, {"train/loss": v}) for i, v in enumerate(_DIVERGING)]
    findings = detect_divergence(metrics)
    assert [f.rule_id for f in findings] == ["divergence"]


def test_detect_overfitting_sees_pytorch_tutorial_naming() -> None:
    metrics = [
        _snap_named(i, {"Loss/train": t, "Loss/val": v})
        for i, (t, v) in enumerate(zip(_TRAIN_FALLING, _VAL_RISING, strict=True))
    ]
    findings = detect_overfitting(metrics)
    assert [f.rule_id for f in findings] == ["overfitting"]


def test_detect_overfitting_sees_wandb_default_naming() -> None:
    metrics = [
        _snap_named(i, {"train/loss": t, "val/loss": v})
        for i, (t, v) in enumerate(zip(_TRAIN_FALLING, _VAL_RISING, strict=True))
    ]
    findings = detect_overfitting(metrics)
    assert [f.rule_id for f in findings] == ["overfitting"]


def test_detect_nan_already_saw_separator_variants() -> None:
    # The asymmetry that made the defect visible: the NaN rule's substring
    # test always matched these names. Pinned so the shared normalisation
    # does not "fix" the tuples by breaking this one.
    metrics = [_snap_named(0, {"Loss/train": float("nan")})]
    assert [f.rule_id for f in detect_nan(metrics)] == ["nan"]


def test_separator_variants_do_not_cross_train_and_val() -> None:
    # Normalising separators must not blur the train/val distinction:
    # a val-only history has no train series, so overfitting cannot fire.
    metrics = [_snap_named(i, {"Loss/val": v}) for i, v in enumerate(_VAL_RISING)]
    assert detect_overfitting(metrics) == []
    assert detect_divergence(metrics) == []


def test_dotted_and_dashed_variants_also_match() -> None:
    for name in ("train.loss", "train-loss", "Train Loss", "TRAIN_LOSS"):
        metrics = [_snap_named(i, {name: v}) for i, v in enumerate(_DIVERGING)]
        assert [f.rule_id for f in detect_divergence(metrics)] == ["divergence"], (
            f"missed spelling {name!r}"
        )
