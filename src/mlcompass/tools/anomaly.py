"""Anomaly detection over a list of ``MetricSnapshot``.

Pure deterministic functions, no LLM. Each detector takes the metric
history and returns a list of :class:`Finding` for the *current end of
history* — i.e., they are stateless and idempotent, designed to be
re-called whenever a new snapshot arrives during ``watch --follow``.

The four v0.2 detectors:

* :func:`detect_nan` — any loss-like metric is NaN or ±Inf
* :func:`detect_divergence` — train loss grows >10× between consecutive epochs
* :func:`detect_plateau` — primary metric flat across the last N epochs
* :func:`detect_overfitting` — train loss falling while val loss rising,
  with a meaningful train/val gap
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .logs import MetricSnapshot, has_invalid_loss

# --------------------------------------------------------------------------- #
# Public types                                                                #
# --------------------------------------------------------------------------- #


@dataclass
class Finding:
    """A single live-watch anomaly hit."""

    rule_id: str
    severity: str  # "error" | "warning" | "info"
    message: str
    suggestion: str
    epoch: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_TRAIN_LOSS_KEYS = ("train_loss", "loss", "training_loss", "tr_loss")
_VAL_LOSS_KEYS = ("val_loss", "valid_loss", "validation_loss")


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _first_present(snap: MetricSnapshot, keys: Iterable[str]) -> float | None:
    for key in keys:
        if key in snap.metrics:
            return snap.metrics[key]
    return None


def _train_loss(snap: MetricSnapshot) -> float | None:
    return _first_present(snap, _TRAIN_LOSS_KEYS)


def _val_loss(snap: MetricSnapshot) -> float | None:
    return _first_present(snap, _VAL_LOSS_KEYS)


def _series(metrics: list[MetricSnapshot], key_getter) -> list[tuple[int, float]]:
    """Pull out an (epoch_index, value) series for snapshots that have the value."""
    out: list[tuple[int, float]] = []
    for idx, snap in enumerate(metrics):
        value = key_getter(snap)
        if value is None or math.isnan(value) or math.isinf(value):
            continue
        out.append((idx, value))
    return out


# --------------------------------------------------------------------------- #
# Detectors                                                                   #
# --------------------------------------------------------------------------- #


def detect_nan(metrics: list[MetricSnapshot]) -> list[Finding]:
    """Flag the *latest* snapshot if any loss-like value is NaN or Inf."""
    if not metrics:
        return []
    latest = metrics[-1]
    if not has_invalid_loss(latest):
        return []
    bad_keys = [
        key
        for key, value in latest.metrics.items()
        if "loss" in key.lower() and (math.isnan(value) or math.isinf(value))
    ]
    return [
        Finding(
            rule_id="nan",
            severity="error",
            message=(
                "Loss became NaN or Inf: "
                + ", ".join(f"{k}={latest.metrics[k]!r}" for k in bad_keys)
            ),
            suggestion=(
                "Stop training. Look for log(0), division by zero, exploding "
                "gradients (try lower lr or gradient clipping), or numerical "
                "instability in your loss formula."
            ),
            epoch=latest.epoch,
        )
    ]


def detect_divergence(
    metrics: list[MetricSnapshot],
    *,
    ratio_threshold: float = 10.0,
) -> list[Finding]:
    """Flag when train loss grows by ``>= ratio_threshold×`` between epochs."""
    series = _series(metrics, _train_loss)
    if len(series) < 2:
        return []

    last_idx, last_val = series[-1]
    prev_idx, prev_val = series[-2]
    if prev_val <= 0 or last_val <= 0:
        return []
    ratio = last_val / prev_val
    if ratio < ratio_threshold:
        return []

    return [
        Finding(
            rule_id="divergence",
            severity="error",
            message=(
                f"Train loss jumped {ratio:.1f}× ({prev_val:.4g} → {last_val:.4g}) "
                f"between snapshots {prev_idx} and {last_idx}."
            ),
            suggestion=(
                "Stop training. Try lower learning rate, add gradient clipping, "
                "or check for a bad batch / corrupted data."
            ),
            epoch=metrics[-1].epoch,
        )
    ]


def detect_plateau(
    metrics: list[MetricSnapshot],
    *,
    window: int = 5,
    rel_threshold: float = 1e-3,
) -> list[Finding]:
    """Flag when the primary loss is flat across the last ``window`` snapshots.

    Prefers ``val_loss`` as the primary signal, falls back to ``train_loss``.
    """
    if len(metrics) < window:
        return []

    for getter, label in [(_val_loss, "val_loss"), (_train_loss, "train_loss")]:
        recent_values = []
        for snap in metrics[-window:]:
            value = getter(snap)
            if value is None:
                break
            recent_values.append(value)
        if len(recent_values) < window:
            continue

        baseline = max(abs(v) for v in recent_values) or 1.0
        spread = max(recent_values) - min(recent_values)
        if spread / baseline < rel_threshold:
            return [
                Finding(
                    rule_id="plateau",
                    severity="warning",
                    message=(
                        f"{label} flat over last {window} snapshots "
                        f"(spread {spread:.5f}, relative {spread / baseline:.2%})."
                    ),
                    suggestion=(
                        "Try a cosine or step LR schedule, increase model "
                        "capacity, or check whether your data is leaking train→val."
                    ),
                    epoch=metrics[-1].epoch,
                )
            ]
        # If we got a valid window for this signal, don't fall through.
        return []
    return []


def detect_overfitting(
    metrics: list[MetricSnapshot],
    *,
    window: int = 3,
    gap_threshold: float = 0.05,
) -> list[Finding]:
    """Flag when train loss is falling and val loss is rising over a window.

    The third condition — a meaningful absolute gap — keeps the rule
    quiet during very early epochs where the model is still warming up.
    """
    train_series = _series(metrics, _train_loss)
    val_series = _series(metrics, _val_loss)
    if len(train_series) < window + 1 or len(val_series) < window + 1:
        return []

    train_recent_change = train_series[-1][1] - train_series[-window - 1][1]
    val_recent_change = val_series[-1][1] - val_series[-window - 1][1]
    current_gap = val_series[-1][1] - train_series[-1][1]

    if train_recent_change < 0 and val_recent_change > 0 and current_gap > gap_threshold:
        return [
            Finding(
                rule_id="overfitting",
                severity="warning",
                message=(
                    f"train_loss dropped {train_recent_change:.4f} but "
                    f"val_loss rose {val_recent_change:+.4f} over the last "
                    f"{window} snapshots; current gap is {current_gap:.4f}."
                ),
                suggestion=(
                    "Add regularisation (dropout, weight decay), use early "
                    "stopping, or augment / increase the dataset."
                ),
                epoch=metrics[-1].epoch,
            )
        ]
    return []


# --------------------------------------------------------------------------- #
# Combined runner                                                             #
# --------------------------------------------------------------------------- #


def run_all_detectors(metrics: list[MetricSnapshot]) -> list[Finding]:
    """Run every detector and return all findings (NaN first, by severity)."""
    findings: list[Finding] = []
    findings.extend(detect_nan(metrics))
    findings.extend(detect_divergence(metrics))
    findings.extend(detect_overfitting(metrics))
    findings.extend(detect_plateau(metrics))
    return findings
