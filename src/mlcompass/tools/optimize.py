"""Hyperparameter optimization advisor — pure ``numpy``.

Reads a folder of run records (the same on-disk layout the ``compare``
command consumes) and recommends the next configurations to try.

Three layers:

1. **History summary** — best run for the chosen metric, plus a
   leaderboard of the top-K runs.
2. **Sensitivity analysis** — for each numeric hyperparameter, how
   much does the target metric move when that hyperparameter changes
   (rank-correlation against the metric)? Surfaces which knobs matter.
3. **Suggested next configs** — a small portfolio of next-step
   suggestions, each one a perturbation around the current best with
   a stated rationale. We deliberately keep the recipe simple
   (tournament perturbation around the leader, biased toward the most
   sensitive hyperparameters) rather than pretending to be a full
   Gaussian-process optimizer; the LLM strategist mode (``--llm``)
   builds on this with richer reasoning.

No scikit-learn / scipy / Optuna dependency. The numerics we need
(rank correlation, tournament perturbation) are short.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import numpy as np

from ..context import ProjectContext
from .runs import RunRecord, load_run

# Hyperparameter name patterns we treat as numeric when no explicit
# constraint is supplied.
_NUMERIC_HYPER_HINTS = (
    "lr",
    "learning_rate",
    "batch_size",
    "dropout",
    "weight_decay",
    "momentum",
    "epochs",
    "hidden_dim",
    "n_layers",
    "warmup",
    "beta1",
    "beta2",
)


class OptimizeAnalysisError(ValueError):
    """Raised when the optimize input is invalid (no runs, bad metric, …)."""


# --------------------------------------------------------------------------- #
# Public surface                                                              #
# --------------------------------------------------------------------------- #


def load_runs_from_dir(runs_dir: Path | str) -> list[RunRecord]:
    """Load every run record under ``runs_dir`` (skipping non-run subdirs)."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        raise OptimizeAnalysisError(f"Runs directory not found: {runs_dir}")

    records: list[RunRecord] = []
    for entry in sorted(runs_dir.iterdir()):
        if not entry.is_dir():
            continue
        try:
            # _SyntheticProject is a structural stand-in (only the
            # `path` attribute is read by load_run); cast to keep mypy
            # happy without dragging in a real ProjectContext for what
            # is really just a "runs/" parent pointer.
            project = cast(ProjectContext, _SyntheticProject(runs_dir))
            records.append(load_run(entry.name, project=project))
        except Exception:  # noqa: BLE001 — skip anything that doesn't parse.
            continue
    return records


class _SyntheticProject:
    """Stand-in for ``ProjectContext`` so we can reuse ``load_run`` here."""

    def __init__(self, runs_dir: Path) -> None:
        # ``load_run`` looks under ``project.path / "runs" / <id>``.
        self.path = runs_dir.parent


def optimize_history(
    runs: list[RunRecord],
    *,
    metric: str,
    direction: str = "max",
    top_k: int = 5,
    n_suggestions: int = 3,
    constraints: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Summarise a run history and propose the next configurations.

    Args:
        runs: Loaded run records (e.g. from :func:`load_runs_from_dir`).
        metric: Metric name to optimize (case-insensitive). Looks at
            each run's final epoch.
        direction: ``"max"`` (higher is better) or ``"min"``.
        top_k: How many of the top runs to surface in the leaderboard.
        n_suggestions: Number of next-config recommendations to emit.
        constraints: Optional ``{hyperparam: (lo, hi)}`` bounds the
            suggester must stay within.

    Returns:
        Structured dict with keys: ``metric``, ``direction``,
        ``n_runs``, ``best``, ``leaderboard``, ``sensitivity``,
        ``suggestions``, ``warnings``.
    """
    if not runs:
        raise OptimizeAnalysisError(
            "No runs to analyse. Run training first and record it under <project>/runs/<id>/."
        )
    if direction not in {"max", "min"}:
        raise OptimizeAnalysisError(f"direction must be 'max' or 'min', got {direction!r}.")

    warnings: list[str] = []

    scored = _score_runs(runs, metric=metric, direction=direction)
    if not scored:
        raise OptimizeAnalysisError(
            f"No run in the supplied history exposes the metric {metric!r}. "
            "Check the spelling, or that metrics.json includes it on the "
            "final epoch."
        )
    if len(scored) < len(runs):
        skipped = len(runs) - len(scored)
        warnings.append(
            f"{skipped} run(s) had no value for {metric!r} on the final epoch and were skipped."
        )

    scored.sort(key=lambda r: r["score"], reverse=(direction == "max"))
    leaderboard = scored[:top_k]
    best = leaderboard[0]

    sensitivity = _sensitivity(scored, direction=direction)
    suggestions = _suggest_next_configs(
        scored=scored,
        sensitivity=sensitivity,
        n=n_suggestions,
        direction=direction,
        constraints=constraints or {},
    )

    return {
        "metric": metric,
        "direction": direction,
        "n_runs": len(runs),
        "n_scored": len(scored),
        "best": best,
        "leaderboard": leaderboard,
        "sensitivity": sensitivity,
        "suggestions": suggestions,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# Scoring + sensitivity                                                       #
# --------------------------------------------------------------------------- #


def _score_runs(
    runs: list[RunRecord],
    *,
    metric: str,
    direction: str,
) -> list[dict[str, Any]]:
    """Pull the chosen metric off each run's final epoch."""
    out: list[dict[str, Any]] = []
    metric_lower = metric.lower()
    for run in runs:
        final = run.final_metrics or {}
        value = _find_value_case_insensitive(final, metric_lower)
        if value is None or _is_bad_number(value):
            continue
        out.append(
            {
                "id": run.id,
                "name": run.name,
                "config": dict(run.config or {}),
                "score": float(value),
            }
        )
    return out


def _sensitivity(
    scored: list[dict[str, Any]],
    *,
    direction: str,
) -> list[dict[str, Any]]:
    """Per-hyperparameter rank-correlation with the score.

    Spearman-ish: we rank both the hyperparameter values and the
    scores, then compute Pearson r on the ranks. Robust to non-linear
    monotone relationships, no scipy needed.
    """
    if len(scored) < 3:
        return []  # Too few runs for any signal.

    scores = np.array([r["score"] for r in scored], dtype=float)
    rank_score = _rank(scores)
    sign = 1.0 if direction == "max" else -1.0
    rank_score = sign * rank_score  # ⇒ higher rank ⇒ "better" regardless of direction

    out: list[dict[str, Any]] = []
    # Union of hyperparameter names across all configs.
    all_keys = sorted({k for r in scored for k in r["config"]})
    for key in all_keys:
        values_raw = [r["config"].get(key) for r in scored]
        numeric_values: list[float | None] = []
        for v in values_raw:
            if v is None:
                numeric_values.append(None)
                continue
            if isinstance(v, (int, float)) and not _is_bad_number(v):
                numeric_values.append(float(v))
            else:
                # Categorical hyperparameter — record presence but skip
                # rank correlation; we still want to report it exists.
                numeric_values.append(None)

        present = [(i, v) for i, v in enumerate(numeric_values) if v is not None]
        if len(present) < 3 or len({v for _, v in present}) < 2:
            out.append(
                {
                    "hyperparam": key,
                    "kind": "categorical_or_constant",
                    "n_observations": len(present),
                    "correlation": None,
                }
            )
            continue

        idxs = np.array([i for i, _ in present])
        hp_values = np.array([v for _, v in present])
        rank_hp = _rank(hp_values)
        r = _pearson(rank_hp, rank_score[idxs])
        out.append(
            {
                "hyperparam": key,
                "kind": "numeric",
                "n_observations": len(present),
                "correlation": float(r),
                "min": float(np.min(hp_values)),
                "max": float(np.max(hp_values)),
            }
        )

    # Rank by absolute correlation so the most impactful knobs come first.
    out.sort(
        key=lambda d: abs(d["correlation"]) if d["correlation"] is not None else -1,
        reverse=True,
    )
    return out


# --------------------------------------------------------------------------- #
# Suggestion engine                                                           #
# --------------------------------------------------------------------------- #


def _suggest_next_configs(
    *,
    scored: list[dict[str, Any]],
    sensitivity: list[dict[str, Any]],
    n: int,
    direction: str,
    constraints: dict[str, tuple[float, float]],
) -> list[dict[str, Any]]:
    """Tournament perturbation around the leader, biased by sensitivity.

    Suggestion 1: exploit — small perturbation toward the direction the
        most-correlated knob is pointing.
    Suggestion 2: explore — larger perturbation along the same knob,
        opposite direction (escape local optima).
    Suggestion 3..n: vary the next sensitive knob.
    """
    if not scored:
        return []
    leader = scored[0]
    base_config = dict(leader["config"])

    # Numeric, observed-multi-value knobs ranked by absolute corr.
    numeric_sens = [
        s
        for s in sensitivity
        if s.get("kind") == "numeric"
        and s.get("correlation") is not None
        and abs(s["correlation"]) >= 0.05
    ]

    suggestions: list[dict[str, Any]] = []
    if not numeric_sens:
        # Fall back to "try again with the leader" — no signal yet.
        suggestions.append(
            {
                "config": base_config,
                "rationale": (
                    "Not enough variation across runs to find a sensitive "
                    "hyperparameter. Suggest re-running the leader to "
                    "estimate noise, then varying lr / batch_size."
                ),
                "perturbed": [],
            }
        )
        return suggestions[:n]

    # Plan a portfolio: alternate exploit / explore on the top knob,
    # then exploit on each of the next knobs.
    plans: list[tuple[str, float, str]] = []
    top_knob = numeric_sens[0]
    plans.append((top_knob["hyperparam"], 0.5, "exploit:top-knob"))
    plans.append((top_knob["hyperparam"], -0.5, "explore:opposite"))
    for knob in numeric_sens[1:]:
        if len(plans) >= n:
            break
        plans.append((knob["hyperparam"], 0.5, "exploit:next-knob"))

    for hp_name, step_frac, label in plans[:n]:
        sens_entry = next(s for s in numeric_sens if s["hyperparam"] == hp_name)
        new_value = _perturb(
            current=base_config.get(hp_name, sens_entry["min"]),
            corr=sens_entry["correlation"],
            direction=direction,
            step_frac=step_frac,
            obs_min=sens_entry["min"],
            obs_max=sens_entry["max"],
            bounds=constraints.get(hp_name),
        )
        proposed = dict(base_config)
        proposed[hp_name] = new_value
        suggestions.append(
            {
                "config": proposed,
                "rationale": _rationale(label, hp_name, sens_entry, new_value),
                "perturbed": [hp_name],
            }
        )

    return suggestions


def _perturb(
    *,
    current: Any,
    corr: float,
    direction: str,
    step_frac: float,
    obs_min: float,
    obs_max: float,
    bounds: tuple[float, float] | None,
) -> float | int:
    """Step the hyperparameter in the direction that improves the metric.

    For a 'max' direction with positive correlation, larger values
    helped, so step up. We pick the perturbation scheme based on the
    observed distribution: strictly-positive observations (typical for
    learning rates, dropout, weight decay) get **multiplicative** steps
    (×2 / ÷2 ≈ a log-space move), preserving the sign and avoiding
    nonsensical negative learning rates. Mixed-sign distributions get
    **additive** steps relative to the observed range.

    User-supplied ``bounds`` always clamp the result and override the
    "preserve sign" heuristic, so users can deliberately let a value
    cross zero by setting ``bounds=(-x, y)``.
    """
    current_val = float(current) if isinstance(current, (int, float)) else obs_min

    sign = 1.0 if (corr > 0) == (direction == "max") else -1.0
    sign *= 1.0 if step_frac >= 0 else -1.0
    magnitude = abs(step_frac)

    # Strict-positive observations + no user override ⇒ multiplicative.
    use_multiplicative = bounds is None and obs_min > 0 and current_val > 0

    if use_multiplicative:
        # Step half a decade (×~3.16) at full magnitude; ÷~3.16 going down.
        # A 0.5-magnitude move goes ×1.78 / ÷1.78.
        factor = math.pow(10.0, sign * magnitude)
        raw = current_val * factor
        lo = obs_min / 10.0  # never propose less than 1/10 of the smallest observed
        hi = obs_max * 10.0
    else:
        span = max(obs_max - obs_min, abs(current_val) * 0.5, 1e-6)
        raw = current_val + sign * magnitude * span
        if bounds:
            lo, hi = bounds
        else:
            lo, hi = obs_min - span, obs_max + span

    clamped = max(lo, min(hi, raw))

    # If the original was integer-typed, round.
    if isinstance(current, int) and not isinstance(current, bool):
        return int(round(clamped))
    return float(clamped)


def _rationale(
    label: str,
    hp_name: str,
    sens: dict[str, Any],
    new_value: float | int,
) -> str:
    corr = sens["correlation"]
    if label.startswith("exploit"):
        if corr > 0:
            return (
                f"{hp_name} correlates positively with the metric "
                f"(r={corr:+.2f}); step further in the same direction "
                f"to {new_value:g}."
            )
        return (
            f"{hp_name} correlates negatively with the metric "
            f"(r={corr:+.2f}); step further in the opposite direction "
            f"to {new_value:g}."
        )
    return (
        f"Explore: invert the {hp_name} step to {new_value:g} to escape a possible local optimum."
    )


# --------------------------------------------------------------------------- #
# Numerics                                                                    #
# --------------------------------------------------------------------------- #


def _rank(values: np.ndarray) -> np.ndarray:
    """Average-tie ranks — same convention as scipy.stats.rankdata."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    # Average across ties.
    sorted_vals = values[order]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    a_mean, b_mean = np.mean(a), np.mean(b)
    a_centered, b_centered = a - a_mean, b - b_mean
    denom = math.sqrt(float(np.sum(a_centered**2)) * float(np.sum(b_centered**2)))
    if denom == 0:
        return 0.0
    return float(np.sum(a_centered * b_centered) / denom)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _find_value_case_insensitive(
    record: dict[str, Any],
    key_lower: str,
) -> float | None:
    for k, v in record.items():
        if isinstance(k, str) and k.lower() == key_lower:
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None


def _is_bad_number(v: Any) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return True
    return math.isnan(f) or math.isinf(f)


def parse_constraints(spec: str) -> dict[str, tuple[float, float]]:
    """Parse a CLI-style constraint string, e.g. ``"lr:0.0001-0.1,batch_size:16-256"``."""
    out: dict[str, tuple[float, float]] = {}
    if not spec:
        return out
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise OptimizeAnalysisError(f"Bad constraint segment {chunk!r}. Expected 'name:lo-hi'.")
        name, range_str = chunk.split(":", 1)
        name = name.strip()
        if "-" not in range_str:
            raise OptimizeAnalysisError(f"Bad range in constraint {chunk!r}. Expected 'lo-hi'.")
        lo_str, hi_str = range_str.split("-", 1)
        try:
            lo, hi = float(lo_str), float(hi_str)
        except ValueError as e:
            raise OptimizeAnalysisError(
                f"Constraint bounds for {name!r} must be numeric, got {range_str!r}."
            ) from e
        if hi < lo:
            lo, hi = hi, lo
        out[name] = (lo, hi)
    return out


__all__ = [
    "OptimizeAnalysisError",
    "load_runs_from_dir",
    "optimize_history",
    "parse_constraints",
]


# Re-export so tests can iterate handily.
def iter_run_metrics(runs: Iterable[RunRecord]) -> Iterable[dict[str, Any]]:
    for r in runs:
        yield r.final_metrics
