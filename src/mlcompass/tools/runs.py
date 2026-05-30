"""Run records: load, parse, and compare.

A run record on disk follows the layout described in
``ARCHITECTURE.md §2``::

    .mlcompass/runs/<run-id>/
    ├── config.yaml      # hyperparameters + metadata
    ├── metrics.json     # per-epoch metric history
    └── notes.md         # free-text notes (optional)

``config.yaml`` schema::

    name: baseline
    created: 2026-05-29T15:00:00Z
    config:
      lr: 0.001
      batch_size: 64
      optimizer: AdamW
      ...

``metrics.json`` schema::

    {
      "metrics": [
        {"epoch": 0, "train_loss": 0.80, "val_loss": 0.75},
        {"epoch": 1, "train_loss": 0.60, "val_loss": 0.65},
        ...
      ]
    }
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from ..context import ProjectContext

# Metric direction heuristics. We only assign a verdict for metrics whose
# direction we know.
_LOWER_IS_BETTER = ("loss", "error", "mae", "mse", "rmse", "nll", "perplexity")
_HIGHER_IS_BETTER = ("acc", "auc", "f1", "precision", "recall", "iou", "score", "r2")


class RunNotFoundError(FileNotFoundError):
    """Raised when a run cannot be located by identifier or path."""


# --------------------------------------------------------------------------- #
# Public types                                                                #
# --------------------------------------------------------------------------- #


@dataclass
class RunRecord:
    """In-memory representation of one run record on disk."""

    id: str
    path: Path
    name: str | None
    created: str | None
    config: dict[str, Any]
    metrics: list[dict[str, Any]] = field(default_factory=list)
    notes: str | None = None

    @property
    def final_metrics(self) -> dict[str, Any]:
        """Return the last epoch's metric dict, or ``{}`` if empty."""
        if not self.metrics:
            return {}
        return self.metrics[-1]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["path"] = str(self.path)
        return d


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #


def load_run(
    identifier: str | Path,
    *,
    project: ProjectContext | None = None,
) -> RunRecord:
    """Load a run by identifier.

    Resolution order:

    1. If ``identifier`` resolves to an existing directory, treat it as the
       run path.
    2. Otherwise, if ``project`` is given, look under
       ``<project>/runs/<identifier>``.

    Raises:
        RunNotFoundError: If no candidate resolves to a valid run directory.
    """
    candidates: list[Path] = []

    direct = Path(identifier)
    if direct.is_dir():
        candidates.append(direct)

    if project is not None:
        candidates.append(project.path / "runs" / str(identifier))

    for candidate in candidates:
        if _looks_like_run_dir(candidate):
            return _load_run_from_dir(candidate, str(identifier))

    where = ", ".join(str(c) for c in candidates) or "<no candidates>"
    raise RunNotFoundError(
        f"Run {identifier!r} not found. Looked at: {where}. "
        "Expected a directory containing config.yaml and metrics.json."
    )


def _looks_like_run_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.yaml").is_file()


def _load_run_from_dir(path: Path, identifier: str) -> RunRecord:
    config_data: dict[str, Any] = yaml.safe_load(
        (path / "config.yaml").read_text(encoding="utf-8")
    ) or {}

    metrics_path = path / "metrics.json"
    metrics: list[dict[str, Any]] = []
    if metrics_path.is_file():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        # Accept either {"metrics": [...]} or a bare list, defensively.
        if isinstance(payload, dict) and isinstance(payload.get("metrics"), list):
            metrics = payload["metrics"]
        elif isinstance(payload, list):
            metrics = payload

    notes_path = path / "notes.md"
    notes = notes_path.read_text(encoding="utf-8") if notes_path.is_file() else None

    # The on-disk directory name is the canonical run ID; if the user
    # passed a different identifier (e.g. an absolute path), we still
    # surface the directory name as the ID for stable display.
    return RunRecord(
        id=path.name,
        path=path,
        name=config_data.get("name"),
        created=config_data.get("created"),
        config=dict(config_data.get("config") or {}),
        metrics=metrics,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Comparison                                                                  #
# --------------------------------------------------------------------------- #


def compare_runs(a: RunRecord, b: RunRecord) -> dict[str, Any]:
    """Produce a structured comparison between two runs."""
    config_diff = _diff_configs(a.config, b.config)
    metric_comparison = _compare_metrics(a.final_metrics, b.final_metrics)
    verdict, verdict_text = _verdict(metric_comparison)

    return {
        "run_a": _run_summary(a),
        "run_b": _run_summary(b),
        "config_diff": config_diff,
        "metric_comparison": metric_comparison,
        "verdict": verdict,
        "verdict_explanation": verdict_text,
    }


def _run_summary(run: RunRecord) -> dict[str, Any]:
    return {
        "id": run.id,
        "name": run.name,
        "created": run.created,
        "config": run.config,
        "final_metrics": run.final_metrics,
        "epochs": len(run.metrics),
    }


def _diff_configs(a: dict[str, Any], b: dict[str, Any]) -> list[dict[str, Any]]:
    """Return entries for keys whose values differ between ``a`` and ``b``."""
    keys: set[str] = set(a) | set(b)
    diff: list[dict[str, Any]] = []
    for key in sorted(keys):
        a_value = a.get(key, _MISSING)
        b_value = b.get(key, _MISSING)
        if a_value != b_value:
            diff.append(
                {
                    "key": key,
                    "a_value": _serialize(a_value),
                    "b_value": _serialize(b_value),
                }
            )
    return diff


_MISSING = object()


def _serialize(value: Any) -> Any:
    return None if value is _MISSING else value


def _compare_metrics(
    a_final: dict[str, Any],
    b_final: dict[str, Any],
) -> list[dict[str, Any]]:
    """Per-metric comparison for keys present in both runs."""
    rows: list[dict[str, Any]] = []
    common = set(a_final) & set(b_final)
    for name in sorted(common):
        # Skip non-numeric metrics (e.g., epoch counter strings).
        a_value = a_final[name]
        b_value = b_final[name]
        if not isinstance(a_value, (int, float)) or not isinstance(b_value, (int, float)):
            continue
        if name == "epoch":
            continue
        delta = float(b_value) - float(a_value)
        rows.append(
            {
                "name": name,
                "a_value": float(a_value),
                "b_value": float(b_value),
                "delta": delta,
                "better": _which_is_better(name, a_value, b_value),
            }
        )
    return rows


def _which_is_better(name: str, a_value: float, b_value: float) -> str | None:
    """Return ``"a"`` / ``"b"`` / ``None`` based on metric direction heuristics."""
    lower = name.lower()
    if any(token in lower for token in _LOWER_IS_BETTER):
        if a_value < b_value:
            return "a"
        if b_value < a_value:
            return "b"
        return "tie"
    if any(token in lower for token in _HIGHER_IS_BETTER):
        if a_value > b_value:
            return "a"
        if b_value > a_value:
            return "b"
        return "tie"
    return None  # unknown direction


def _verdict(
    metric_comparison: Iterable[dict[str, Any]],
) -> tuple[str, str]:
    """Aggregate per-metric verdicts into an overall verdict."""
    a_wins = 0
    b_wins = 0
    ties = 0
    decisive: list[dict[str, Any]] = []

    for row in metric_comparison:
        better = row.get("better")
        if better == "a":
            a_wins += 1
            decisive.append(row)
        elif better == "b":
            b_wins += 1
            decisive.append(row)
        elif better == "tie":
            ties += 1

    total_decisive = a_wins + b_wins

    if total_decisive == 0:
        return ("inconclusive", "No metrics with a known direction to compare.")
    if a_wins > 0 and b_wins == 0:
        return ("a_better", f"Run A wins on {a_wins}/{total_decisive} known metrics.")
    if b_wins > 0 and a_wins == 0:
        return ("b_better", f"Run B wins on {b_wins}/{total_decisive} known metrics.")
    return (
        "mixed",
        f"Mixed result: A wins {a_wins}, B wins {b_wins}, {ties} tie(s).",
    )
