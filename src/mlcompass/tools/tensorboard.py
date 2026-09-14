"""TensorBoard event-file parser.

Loads scalar metrics from ``events.out.tfevents.*`` files (or directories
containing them) into the same :class:`MetricSnapshot` shape the plain-text
log parser produces. This lets ``watch`` consume either source without the
detection rules needing to care which one is in play.

Scalars are read in ``tbparse``'s long (un-pivoted) form, one row per
scalar actually written, because that is the only shape in which a loss
whose *value* is NaN can be told apart from a metric that was simply not
logged at that step. See the comment in :func:`parse_tb_events`.

The ``tbparse`` dependency is loaded lazily so users who never touch
TensorBoard don't pay the import cost.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .logs import MetricSnapshot


class TensorBoardImportError(ImportError):
    """Raised when ``tbparse`` is required but not installed."""


def parse_tb_events(path: Path | str) -> list[MetricSnapshot]:
    """Read a TensorBoard event file or directory into ``MetricSnapshot`` records.

    Args:
        path: Either an ``events.out.tfevents.*`` file or a directory that
            contains one or more such files.

    Returns:
        A list of snapshots in step order. Each snapshot carries the
        ``step`` value, optionally an ``epoch`` value if the script
        emitted one as a scalar, and every other scalar metric found at
        that step.

    Raises:
        TensorBoardImportError: If ``tbparse`` (and its transitive
            ``tensorboard``) is not installed.
    """
    try:
        from tbparse import SummaryReader
    except ImportError as exc:  # pragma: no cover - exercised only without dep
        raise TensorBoardImportError(
            "TensorBoard support requires the `tbparse` package. "
            "Install with: pip install mlcompass[tensorboard]"
        ) from exc

    # ``pivot=False`` is load-bearing, not a style choice. The pivoted frame
    # has one column per tag and fills every (step, tag) at which nothing was
    # written with NaN — so a loss that *became* NaN and a metric that was
    # simply not logged at that step are the same cell, and no amount of
    # inspection downstream can tell them apart. Dropping NaN cells therefore
    # dropped real NaN losses, and ``detect_nan`` (the only error-severity
    # watch rule) could never fire on a TensorBoard source. The long form
    # carries one row per scalar actually written: a NaN loss is a row whose
    # value is NaN, and an unlogged metric has no row at all.
    reader = SummaryReader(str(path), pivot=False)
    df = reader.scalars

    if df is None or len(df) == 0:
        return []

    metrics_by_step: dict[int | None, dict[str, float]] = {}
    epoch_by_step: dict[int | None, int | None] = {}

    steps = df["step"] if "step" in df.columns else [None] * len(df)
    for raw_step, raw_tag, raw_value in zip(steps, df["tag"], df["value"], strict=True):
        step_value = _as_int(raw_step)
        # Every written row materialises its step, so a step whose scalars
        # are all unusable still becomes a snapshot, as it did pre-pivot.
        metrics_by_step.setdefault(step_value, {})
        epoch_by_step.setdefault(step_value, None)

        tag = str(raw_tag)
        if tag == "epoch":
            epoch_by_step[step_value] = _as_int(raw_value)
            continue

        numeric = _as_float(raw_value)
        if numeric is not None:
            metrics_by_step[step_value][tag] = numeric

    return [
        MetricSnapshot(
            epoch=epoch_by_step[step_value],
            step=step_value,
            metrics=metrics_by_step[step_value],
            source_line=f"tensorboard step={step_value}",
        )
        for step_value in sorted(metrics_by_step, key=_step_sort_key)
    ]


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _step_sort_key(step: int | None) -> tuple[int, int]:
    """Order snapshots by step, with an unknown step sorting first."""
    return (0, 0) if step is None else (1, step)


def _as_int(value: Any) -> int | None:
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    # int(nan) raises and int(inf) raises; neither is a usable step/epoch.
    if math.isnan(as_float) or math.isinf(as_float):
        return None
    return int(as_float)


def _as_float(value: Any) -> float | None:
    """Coerce a scalar cell to float, *keeping* NaN and ±Inf.

    A NaN or Inf here is a real measurement — the loss blew up — and is the
    signal ``detect_nan`` exists to report. It must not be filtered out on
    the way in.
    """
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    return float(value)
