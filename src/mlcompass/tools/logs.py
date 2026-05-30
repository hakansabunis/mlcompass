"""Plain-text training-log parser.

Reads training output produced by ``print(...)``-style logging — the
format the great majority of pyTorch / TensorFlow training scripts emit
out of the box — and turns it into ``MetricSnapshot`` records that the
anomaly detectors can consume.

The parser is intentionally lenient: it accepts ``key=value``,
``key: value``, and ``key value`` pairs, with or without commas, and
recognises the most common epoch / step markers. Lines that don't
contain anything metric-shaped are ignored.

TensorBoard event files and W&B local caches are scheduled for v0.2.1.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# --------------------------------------------------------------------------- #
# Public types                                                                #
# --------------------------------------------------------------------------- #


@dataclass
class MetricSnapshot:
    """A single epoch / step's parsed metrics."""

    epoch: int | None = None
    step: int | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    source_line: str = ""

    @property
    def has_signal(self) -> bool:
        """True if this snapshot carries at least one numeric metric."""
        return bool(self.metrics)


# --------------------------------------------------------------------------- #
# Regex patterns                                                              #
# --------------------------------------------------------------------------- #


# Matches both ``Epoch 5`` and ``Epoch 5/100`` (and ``epoch=5``).
_EPOCH_RE = re.compile(
    r"\b(?:epoch|ep)\b\s*[:=#]?\s*(\d+)\s*(?:/\s*\d+)?",
    re.IGNORECASE,
)

# Matches ``Step 1234`` and ``step=1234`` / ``global_step: 1234``.
_STEP_RE = re.compile(
    r"\b(?:step|global_step|iter)\b\s*[:=]?\s*(\d+)",
    re.IGNORECASE,
)

# Captures ``key=value``, ``key: value``, ``key  value`` pairs where
# value is a Python-style float literal (including ``nan`` / ``inf``).
# We require the key to be at least 2 characters to avoid catching
# single-letter loop variables.
_KV_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]{1,40})\s*[:=]\s*"
    r"(-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|inf|-inf|NaN|Inf)",
)


# Tokens we never treat as metric keys (epoch markers, indices, etc.).
_NON_METRIC_KEYS = frozenset(
    {
        "epoch",
        "ep",
        "step",
        "iter",
        "global_step",
        "batch",
        "batches",
        "num_epochs",
        "total",
    }
)


# --------------------------------------------------------------------------- #
# Parsing                                                                     #
# --------------------------------------------------------------------------- #


def parse_log_line(line: str) -> MetricSnapshot | None:
    """Parse a single line. Returns ``None`` if no signal was found."""
    line = line.strip()
    if not line:
        return None

    epoch_match = _EPOCH_RE.search(line)
    step_match = _STEP_RE.search(line)

    metrics: dict[str, float] = {}
    for match in _KV_RE.finditer(line):
        key = match.group(1)
        if key.lower() in _NON_METRIC_KEYS:
            continue
        raw_value = match.group(2)
        try:
            value = float(raw_value)
        except ValueError:
            continue
        metrics[key] = value

    if not metrics and epoch_match is None and step_match is None:
        return None

    snapshot = MetricSnapshot(
        epoch=int(epoch_match.group(1)) if epoch_match else None,
        step=int(step_match.group(1)) if step_match else None,
        metrics=metrics,
        source_line=line,
    )
    return snapshot if (snapshot.has_signal or snapshot.epoch is not None) else None


def parse_log_text(text: str) -> list[MetricSnapshot]:
    """Parse a whole log blob; returns snapshots in input order."""
    snapshots: list[MetricSnapshot] = []
    for raw_line in text.splitlines():
        snap = parse_log_line(raw_line)
        if snap is not None:
            snapshots.append(snap)
    return snapshots


def parse_log_file(path: Path | str) -> list[MetricSnapshot]:
    """Read a file from disk and parse its lines."""
    return parse_log_text(Path(path).read_text(encoding="utf-8", errors="replace"))


# --------------------------------------------------------------------------- #
# Light helpers used by the watch command                                     #
# --------------------------------------------------------------------------- #


def merge_consecutive_same_epoch(
    snapshots: Iterable[MetricSnapshot],
) -> list[MetricSnapshot]:
    """Merge snapshots that share the same epoch number.

    Many scripts emit one line per metric and one line per epoch, e.g.::

        Epoch 5 train_loss=0.4
        Epoch 5 val_loss=0.5

    This helper collapses such consecutive snapshots into a single
    record so the detectors see one row per epoch.
    """
    merged: list[MetricSnapshot] = []
    for snap in snapshots:
        if (
            merged
            and snap.epoch is not None
            and merged[-1].epoch == snap.epoch
        ):
            merged[-1].metrics.update(snap.metrics)
            if snap.step is not None and merged[-1].step is None:
                merged[-1].step = snap.step
            merged[-1].source_line += " | " + snap.source_line
        else:
            merged.append(snap)
    return merged


# --------------------------------------------------------------------------- #
# Source auto-detection                                                       #
# --------------------------------------------------------------------------- #


def detect_source(path: Path | str) -> str:
    """Identify what kind of metric source ``path`` points to.

    Returns one of:
    - ``"tensorboard"`` — an ``events.out.tfevents.*`` file, or a directory
      containing at least one
    - ``"wandb"`` — a wandb run directory (presence of ``wandb-summary.json``
      or ``wandb-history.jsonl``)
    - ``"plain_text"`` — anything else (the default)
    """
    p = Path(path)
    if p.is_dir():
        if any(p.glob("events.out.tfevents.*")):
            return "tensorboard"
        # Both the W&B run dir and its files/ subdir count as W&B sources.
        wandb_markers = (
            p / "wandb-summary.json",
            p / "wandb-history.jsonl",
            p / "files" / "wandb-summary.json",
            p / "files" / "wandb-history.jsonl",
        )
        if any(m.exists() for m in wandb_markers):
            return "wandb"
        return "plain_text"
    if p.is_file():
        if p.name.startswith("events.out.tfevents"):
            return "tensorboard"
        if p.name in ("wandb-history.jsonl", "wandb-summary.json"):
            return "wandb"
    return "plain_text"


def load_snapshots(path: Path | str) -> tuple[str, list[MetricSnapshot]]:
    """Auto-detect a source type and return ``(source, snapshots)``.

    Plain text falls back to :func:`parse_log_file`. TensorBoard / W&B
    sources delegate to their respective modules with lazy imports.
    """
    source = detect_source(path)
    if source == "tensorboard":
        from .tensorboard import parse_tb_events

        return source, parse_tb_events(path)
    if source == "wandb":
        from .wandb_local import parse_wandb_run

        return source, parse_wandb_run(path)
    return source, parse_log_file(path)


def has_invalid_loss(snapshot: MetricSnapshot) -> bool:
    """True if any *loss-like* metric in the snapshot is NaN or ±Inf."""
    for key, value in snapshot.metrics.items():
        if "loss" not in key.lower():
            continue
        if math.isnan(value) or math.isinf(value):
            return True
    return False
