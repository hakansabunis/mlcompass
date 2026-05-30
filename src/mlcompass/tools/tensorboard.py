"""TensorBoard event-file parser.

Loads scalar metrics from ``events.out.tfevents.*`` files (or directories
containing them) into the same :class:`MetricSnapshot` shape the plain-text
log parser produces. This lets ``watch`` consume either source without the
detection rules needing to care which one is in play.

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

    reader = SummaryReader(str(path), pivot=True)
    df = reader.scalars

    if df is None or len(df) == 0:
        return []

    snapshots: list[MetricSnapshot] = []
    columns = list(df.columns)
    for _, row in df.iterrows():
        epoch_value: int | None = None
        step_value: int | None = None
        metrics: dict[str, float] = {}

        for col in columns:
            raw = row[col]
            if raw is None or _is_missing(raw):
                continue
            if col == "step":
                step_value = _as_int(raw)
            elif col == "epoch":
                epoch_value = _as_int(raw)
            else:
                numeric = _as_float(raw)
                if numeric is not None:
                    metrics[col] = numeric

        if not metrics and epoch_value is None and step_value is None:
            continue

        snapshots.append(
            MetricSnapshot(
                epoch=epoch_value,
                step=step_value,
                metrics=metrics,
                source_line=f"tensorboard step={step_value}",
            )
        )

    return snapshots


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _is_missing(value: Any) -> bool:
    """True if a pandas cell is NaN/None — without importing pandas eagerly."""
    if value is None:
        return True
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return float(value)
