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


def test_detect_nan_only_inspects_latest_snapshot() -> None:
    # An earlier NaN that recovered should still be visible to a watch
    # session that re-reads history? We chose latest-only by design;
    # this test pins that decision.
    metrics = [
        _snap(0, train_loss=float("nan")),
        _snap(1, train_loss=0.4),
    ]
    assert detect_nan(metrics) == []


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
