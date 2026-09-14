"""Anomaly detection over a list of ``MetricSnapshot``.

Pure deterministic functions, no LLM. Each detector takes the metric
history and returns a list of :class:`Finding` for the *current end of
history* — i.e., they are stateless and idempotent, designed to be
re-called whenever a new snapshot arrives during ``watch --follow``.

The four v0.2 detectors. Note that every window below counts *snapshots*
(parsed log records), not epochs — a script that logs per step and one
that logs per epoch present the same run to these rules at very
different resolutions:

* :func:`detect_nan` — any loss-like metric is NaN or ±Inf and has not
  recovered since. Snapshots carrying no loss-like metric at all (a
  trailing summary line such as ``Total time: 412.30``) are skipped
  rather than read as recovery; a NaN followed by a real finite loss is
  still treated as recovered and stays unreported
* :func:`detect_divergence` — train-loss *magnitude* grows >=
  ratio_threshold× between any two consecutive snapshots that carry a
  train loss; the first such jump is reported, with the epoch it
  happened at. Negative losses are included (they are unbounded below),
  and jumps smaller than ``_MIN_DIVERGENCE_JUMP`` in absolute terms are
  treated as noise
* :func:`detect_plateau` — primary loss flat across the last ``window``
  snapshots, which must *each* carry the value (a single interleaved
  snapshot without it disables the rule for that window)
* :func:`detect_overfitting` — train loss falling while val loss rising,
  with a meaningful train/val gap. The two series are windowed
  independently, so when val is logged less often than train the two
  windows span different numbers of epochs, and ``gap_threshold`` is an
  absolute loss value, not a relative one
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from typing import Any

from .logs import (
    MetricSnapshot,
    has_invalid_loss,
    has_loss_metric,
    is_loss_key,
    metric_key_tokens,
)

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


# Canonical spellings only. Matching is separator-, case- and
# order-insensitive (see ``_first_present``), so each entry below also
# covers ``train/loss``, ``Loss/train``, ``train-loss``, ``Train Loss``
# and friends. Add a name here only when it is a genuinely different
# word, not a different way of punctuating one already listed.
_TRAIN_LOSS_KEYS = ("train_loss", "loss", "training_loss", "tr_loss")
_VAL_LOSS_KEYS = ("val_loss", "valid_loss", "validation_loss")

# Smallest absolute growth in loss magnitude that ``detect_divergence``
# will call a jump. A loss this close to zero has converged; the ratios
# it produces are noise, and scanning every consecutive pair would
# otherwise turn that noise into error-severity findings.
_MIN_DIVERGENCE_JUMP = 1e-4


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _first_present(snap: MetricSnapshot, keys: Iterable[str]) -> float | None:
    """First metric in ``snap`` whose name matches one of ``keys``.

    Names are compared as token sets, so one vocabulary entry covers every
    way a framework might punctuate it. An exact-string lookup matched
    neither ``Loss/train`` nor ``train/loss`` — PyTorch's tutorial naming
    and the W&B default respectively — which left plateau, divergence and
    overfitting blind to both while the NaN rule saw them fine.
    """
    available = [(name, metric_key_tokens(name)) for name in snap.metrics]
    for key in keys:
        wanted = metric_key_tokens(key)
        for name, tokens in available:
            if tokens == wanted:
                return snap.metrics[name]
    return None


def _train_loss(snap: MetricSnapshot) -> float | None:
    return _first_present(snap, _TRAIN_LOSS_KEYS)


def _val_loss(snap: MetricSnapshot) -> float | None:
    return _first_present(snap, _VAL_LOSS_KEYS)


def _series(
    metrics: list[MetricSnapshot],
    key_getter: Callable[[MetricSnapshot], float | None],
) -> list[tuple[int, float]]:
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
    """Flag a loss that is NaN or Inf and has not since recovered.

    Only snapshots that actually carry a loss-like metric are consulted.
    A snapshot without one — a trailing ``Total time: 412.30`` summary, an
    interleaved ``lr: 0.001`` line — says nothing about the loss, and is
    not evidence that it recovered. Reading ``metrics[-1]`` alone treated
    any such line as a clean bill of health, so five NaN epochs followed
    by a timing line reported nothing at all while the identical log
    without that line reported the NaN.
    """
    loss_snapshots = [snap for snap in metrics if has_loss_metric(snap)]
    if not loss_snapshots or not has_invalid_loss(loss_snapshots[-1]):
        return []

    # The loss is bad *now*; walk back to where it first went bad so the
    # finding points at the epoch that broke rather than the last one seen.
    first_bad = len(loss_snapshots) - 1
    while first_bad > 0 and has_invalid_loss(loss_snapshots[first_bad - 1]):
        first_bad -= 1
    culprit = loss_snapshots[first_bad]

    bad_keys = [
        key
        for key, value in culprit.metrics.items()
        if is_loss_key(key) and (math.isnan(value) or math.isinf(value))
    ]
    return [
        Finding(
            rule_id="nan",
            severity="error",
            message=(
                "Loss became NaN or Inf: "
                + ", ".join(f"{k}={culprit.metrics[k]!r}" for k in bad_keys)
            ),
            suggestion=(
                "Stop training. Look for log(0), division by zero, exploding "
                "gradients (try lower lr or gradient clipping), or numerical "
                "instability in your loss formula."
            ),
            epoch=culprit.epoch,
        )
    ]


def detect_divergence(
    metrics: list[MetricSnapshot],
    *,
    ratio_threshold: float = 10.0,
) -> list[Finding]:
    """Flag the first ``>= ratio_threshold×`` jump in train-loss magnitude.

    Every consecutive pair is examined, not just the final one. A run that
    blows up at epoch 4 of 8 and then stays high is diverged, but its last
    two points are unremarkable — comparing only those read a 167x
    blow-up as clean, which is precisely the shape a diverged run has by
    the time anyone looks at it.

    Magnitude, not signed value, because log-likelihood, contrastive and
    ELBO losses are routinely negative and are unbounded below: exempting
    non-positive losses made ``[-2.0, -2.1, -2.2, -500.0]`` read clean.
    Comparing ``abs`` treats a plunge to -500 as the blow-up it is, while
    the gentle drift of a healthy negative loss stays far under the
    ratio.
    """
    series = _series(metrics, _train_loss)
    if len(series) < 2:
        return []

    for (prev_idx, prev_val), (last_idx, last_val) in zip(series, series[1:], strict=False):
        prev_magnitude = abs(prev_val)
        last_magnitude = abs(last_val)
        # A zero previous loss makes the ratio meaningless, not infinite.
        if prev_magnitude == 0:
            continue
        # Near zero the ratio is all noise — 1e-9 to 1e-7 is a 100x jump
        # and means nothing. Require the jump to be large in absolute
        # terms as well as relative ones.
        if last_magnitude - prev_magnitude < _MIN_DIVERGENCE_JUMP:
            continue
        ratio = last_magnitude / prev_magnitude
        if ratio < ratio_threshold:
            continue

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
                epoch=metrics[last_idx].epoch,
            )
        ]
    return []


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
