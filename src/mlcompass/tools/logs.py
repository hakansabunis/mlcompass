"""Plain-text training-log parser.

Reads training output produced by ``print(...)``-style logging — the
format the great majority of pyTorch / TensorFlow training scripts emit
out of the box — and turns it into ``MetricSnapshot`` records that the
anomaly detectors can consume.

The parser accepts ``key=value`` and ``key: value`` pairs, with or
without commas, and recognises the most common epoch / step markers.
Lines that don't contain anything metric-shaped are ignored.

A separator is required: ``_KV_RE`` below matches only ``[:=]``, so
whitespace-separated ``key value`` pairs (``loss 0.41``, the nanoGPT
style) yield no metrics. Values may be float literals in any case
(``nan``, ``NAN``, ``inf``, ``-Infinity``) and may carry thousands
separators (``1,000.0``), which ``f"{loss:,.2f}"`` produces.

TensorBoard event files and W&B local caches are handled by
:func:`load_snapshots` below, which dispatches to ``tools.tensorboard``
and ``tools.wandb_local``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

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

# A metric key: a word of at least 2 characters, optionally continued
# across ``/``, ``.`` or ``-`` separators — ``train/loss``, ``Loss/train``,
# ``val.loss``, ``elapsed-time``. A key that stopped at the separator made
# ``train/loss: 0.42 val/loss: 0.55`` parse as the single pair
# ``{"loss": 0.55}``: both names collapsed to ``loss`` and the train value
# was silently replaced by the val value, after which ``_train_loss``
# returned the val loss to every detector downstream.
#
# Each separator must be followed immediately by another word character,
# so file paths (``/tmp/model.pt``, ``./runs/exp-1``) and the ``-`` used
# as a Keras field delimiter (``loss: 0.4 - val_loss: 0.5``) are not
# swallowed into a key.
_KEY_PATTERN = r"[A-Za-z_][A-Za-z0-9_]{1,40}(?:[/.\-][A-Za-z0-9_]{1,40})*"


def _any_case(word: str) -> str:
    """Regex source matching ``word`` in any mix of cases.

    Spelled out rather than using a scoped ``(?i:...)`` flag group, which
    needs Python 3.11 — this package supports 3.10.
    """
    return "".join(f"[{char.lower()}{char.upper()}]" for char in word)


# The integer part of a number, with or without thousands separators.
# ``f"{loss:,.2f}"`` is an ordinary format string, and a pattern that
# stopped at the comma read ``1,000.0`` as ``1.0`` and ``12,345.67`` as
# ``12.0`` — wrong by three orders of magnitude, silently, feeding every
# threshold downstream. A separator only counts when followed by exactly
# three digits, so a comma used as a field delimiter
# (``train_loss=0.42, val_loss=0.55``) still ends the value.
_INT_PART = r"\d{1,3}(?:,\d{3})+|\d+"

_NUMBER = rf"-?(?:(?:{_INT_PART})(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

# ``NAN`` and ``-Infinity`` are what NumPy and several loggers print;
# matching only ``nan``/``NaN``/``inf``/``Inf`` dropped those pairs
# entirely, so the NaN rule never even saw the metric. The trailing
# boundary stops ``status: infeasible`` from parsing as positive infinity
# — which the bare ``inf`` alternative did. Longest spelling first, so
# ``Infinity`` is not matched as ``Inf`` with a dangling ``inity``.
_NON_FINITE = (
    rf"[-+]?(?:{_any_case('nan')}|{_any_case('infinity')}|{_any_case('inf')})"
    r"(?![A-Za-z0-9_])"
)

# Captures ``key=value`` and ``key: value`` pairs. A separator is
# required: whitespace-only ``key value`` pairs are not matched.
_KV_RE = re.compile(rf"({_KEY_PATTERN})\s*[:=]\s*({_NUMBER}|{_NON_FINITE})")


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
        # Through the shared normaliser, so ``global/step`` is recognised
        # as the step marker it is rather than kept as a metric.
        if normalise_metric_key(key) in _NON_METRIC_KEYS:
            continue
        raw_value = match.group(2)
        try:
            # Thousands separators are matched above but are not part of
            # the literal float() accepts.
            value = float(raw_value.replace(",", ""))
        except ValueError:
            continue
        metrics[key] = value

    # The decision, in one place: a line is a snapshot if it carries a
    # measurement, or an epoch marker (which the report uses as context).
    # A step marker alone is neither — ``iter 4000`` gives the detectors
    # nothing to judge — so it is not a snapshot.
    #
    # This used to be two guards. The first admitted step-only lines and
    # the second dropped them, so the first one's step term was dead code
    # and neither guard stated the rule on its own.
    if not metrics and epoch_match is None:
        return None

    return MetricSnapshot(
        epoch=int(epoch_match.group(1)) if epoch_match else None,
        step=int(step_match.group(1)) if step_match else None,
        metrics=metrics,
        source_line=line,
    )


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
        if merged and snap.epoch is not None and merged[-1].epoch == snap.epoch:
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


# The same metric arrives under different separators and cases depending on
# what wrote it: ``train_loss`` from plain-text logs, ``train/loss`` (the
# W&B default), ``Loss/train`` (PyTorch's own TensorBoard tutorial),
# ``train-loss``, ``Train Loss``. Every comparison of a metric name against
# a known vocabulary goes through here, so the anomaly detectors' key
# tuples and :func:`has_invalid_loss` cannot drift apart again — they did,
# and the substring test saw names the exact-match tuples could not.
_KEY_SEPARATOR_RE = re.compile(r"[\s_/.\-]+")


def normalise_metric_key(key: str) -> str:
    """Fold separator and case variants of a metric name onto one spelling.

    ``Loss/train``, ``Loss.train``, ``loss-train`` and ``Loss Train`` all
    become ``loss_train``. Word *order* is preserved: matching
    ``Loss/train`` to ``train_loss`` is the caller's vocabulary decision,
    not this function's — see :func:`metric_key_tokens`.
    """
    return _KEY_SEPARATOR_RE.sub("_", key.strip().lower()).strip("_")


def metric_key_tokens(key: str) -> frozenset[str]:
    """The words in a metric name, separator-, case- and order-insensitive.

    ``Loss/train`` and ``train_loss`` both yield ``{"train", "loss"}``,
    which is what lets one vocabulary entry cover both conventions.
    """
    return frozenset(normalise_metric_key(key).split("_")) - {""}


def is_loss_key(key: str) -> bool:
    """True if ``key`` names a loss-like metric.

    Deliberately a substring test over the normalised name, so plural and
    suffixed spellings (``total_losses``, ``mse_loss_value``) keep
    counting as they always have.
    """
    return "loss" in normalise_metric_key(key)


def has_loss_metric(snapshot: MetricSnapshot) -> bool:
    """True if the snapshot carries any loss-like metric at all.

    Distinct from :func:`has_invalid_loss`: this asks whether the snapshot
    says anything about the loss, not whether what it says is bad. A
    trailing ``Total time: 412.30`` summary line parses to a snapshot with
    a metric but no loss, and is not evidence about the loss either way.
    """
    return any(is_loss_key(key) for key in snapshot.metrics)


def has_invalid_loss(snapshot: MetricSnapshot) -> bool:
    """True if any *loss-like* metric in the snapshot is NaN or ±Inf."""
    for key, value in snapshot.metrics.items():
        if not is_loss_key(key):
            continue
        if math.isnan(value) or math.isinf(value):
            return True
    return False
