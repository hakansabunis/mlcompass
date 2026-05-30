"""W&B local run-cache reader.

Reads the JSONL history file that the W&B SDK writes into every local
run directory and converts it into the same :class:`MetricSnapshot`
shape the plain-text and TensorBoard parsers produce.

Layouts the reader recognises:

* The run directory itself
  (``wandb/run-YYYYMMDD_HHMMSS-<id>/``) — the JSONL lives in
  ``files/wandb-history.jsonl``.
* The run's ``files/`` subdirectory directly.
* A path pointing straight at ``wandb-history.jsonl``.

No optional dependency is required — the file is plain JSONL.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .logs import MetricSnapshot

HISTORY_FILENAME = "wandb-history.jsonl"
SUMMARY_FILENAME = "wandb-summary.json"


class WandbRunNotFoundError(FileNotFoundError):
    """Raised when no W&B history file can be located under ``path``."""


def parse_wandb_run(path: Path | str) -> list[MetricSnapshot]:
    """Read a W&B run's history JSONL into a list of snapshots.

    Args:
        path: Either the run directory, the run's ``files/`` subdirectory,
            or the ``wandb-history.jsonl`` file itself.

    Returns:
        Snapshots in the order they appear in the history file.

    Raises:
        WandbRunNotFoundError: If no history file can be located.
    """
    history_file = _locate_history_file(Path(path))

    snapshots: list[MetricSnapshot] = []
    with history_file.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            snap = _row_to_snapshot(row)
            if snap is not None:
                snapshots.append(snap)
    return snapshots


# --------------------------------------------------------------------------- #
# Internals                                                                   #
# --------------------------------------------------------------------------- #


def _locate_history_file(path: Path) -> Path:
    """Resolve ``path`` to an actual ``wandb-history.jsonl`` file."""
    if path.is_file() and path.name == HISTORY_FILENAME:
        return path
    if path.is_dir():
        # Run directory with files/ subdir.
        nested = path / "files" / HISTORY_FILENAME
        if nested.is_file():
            return nested
        # Path already inside files/.
        direct = path / HISTORY_FILENAME
        if direct.is_file():
            return direct
    raise WandbRunNotFoundError(
        f"Could not find {HISTORY_FILENAME} at or under {path}. "
        "Pass either the run directory, its files/ subdirectory, or the "
        "history file itself."
    )


def _row_to_snapshot(row: dict[str, Any]) -> MetricSnapshot | None:
    """Convert one JSONL row from wandb-history.jsonl into a snapshot.

    Underscore-prefixed keys are W&B internals (``_step``, ``_runtime``,
    ``_timestamp``, ``_wandb``); ``_step`` is promoted to ``step`` and the
    rest are ignored. Non-numeric values are also ignored.
    """
    epoch: int | None = None
    step: int | None = None
    metrics: dict[str, float] = {}

    for key, value in row.items():
        if key == "_step":
            step = _as_int(value)
            continue
        if key.startswith("_"):
            continue
        if key == "epoch":
            epoch = _as_int(value)
            continue
        numeric = _as_float(value)
        if numeric is not None:
            metrics[key] = numeric

    if not metrics and epoch is None and step is None:
        return None

    return MetricSnapshot(
        epoch=epoch,
        step=step,
        metrics=metrics,
        source_line=f"wandb step={step}",
    )


def _as_int(value: Any) -> int | None:
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(as_float) or math.isinf(as_float):
        return None
    return int(as_float)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        # bools survive the isinstance(int) check below; we don't want them.
        return None
    if isinstance(value, (int, float)):
        as_float = float(value)
        if math.isnan(as_float) or math.isinf(as_float):
            return None
        return as_float
    return None
