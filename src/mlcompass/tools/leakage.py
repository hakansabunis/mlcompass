"""Deterministic leakage-evidence collector.

Triggered automatically by :func:`mlcompass.tools.evaluation.evaluate`
when the leakage-smell threshold fires (AUC > 0.995 / accuracy > 0.99 /
R² > 0.999 on ≥ 50 rows). The job of this module is **observational**,
not interpretive: it produces a structured ``evidence`` dict that the
optional :mod:`mlcompass.agents.leakage_investigator` agent then
narrates to the user.

Why split tool from agent? **Anti-hallucination**. The investigator
agent is told it may *only* cite facts that appear in this evidence
dict. Hand-wavy speculation has no grounding in the file system and
gets filtered out at the prompt boundary, not at runtime.

Three classes of evidence we gather (all pure pandas + numpy):

1. **Target-feature correlation** — if any non-target / non-pred
   column in the predictions table correlates ≥ 0.99 (in absolute
   value) with ``y_true``, the column is a near-perfect predictor.
   That's the textbook "label leaked into features" pattern.
2. **Exact y_pred == y_true match rate** — fraction of rows where
   the prediction equals the truth. Above 95% for non-trivial tasks
   is a strong signal the model has seen the labels.
3. **Suspicious metric value** — passthrough of the headline metric
   the smell threshold fired on, so the agent can quote it back to
   the user without re-deriving it.

The output is JSON-safe and deliberately small (the agent prompt
budget matters), with a stable schema the agent's pydantic-style
prompt can rely on.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Threshold above which a feature's correlation with the target is
# considered "suspiciously perfect" — high enough that a legitimate
# strong feature can stay below it, low enough that an exact label
# copy stands out clearly.
_FEATURE_CORR_THRESHOLD = 0.99

# Match-rate above which we call exact y_pred / y_true agreement
# suspicious in its own right.
_PERFECT_MATCH_THRESHOLD = 0.95

# Minimum row count for any leakage signal to be statistically
# trustworthy — below this we record the row count as evidence but
# tag the verdict as ``cannot_determine``.
_MIN_TRUSTWORTHY_ROWS = 50


def detect_leakage(
    df: pd.DataFrame,
    *,
    y_true_col: str,
    y_pred_col: str | None,
    y_prob_col: str | None = None,
    task: str,
    suspicious_metric: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Gather structured leakage evidence from a predictions table.

    Args:
        df: The same ``DataFrame`` ``evaluate`` was handed.
        y_true_col: Ground-truth column name (already resolved by
            ``evaluate``).
        y_pred_col: Predicted-label column name, if any.
        y_prob_col: Predicted-probability column name, if any.
        task: ``binary_classification`` / ``multiclass_classification``
            / ``regression`` — drives the correlation method (Pearson
            for regression, Spearman for ordinal label).
        suspicious_metric: ``{"name": "...", "value": ...}`` for the
            metric that tripped the smell threshold. Passed through
            into the evidence so the agent can name it.

    Returns:
        ``{
            "ok": True,
            "row_count": int,
            "suspicious_metric": {name, value} | None,
            "target_feature_correlations": [
                {"feature": str, "correlation": float, "method": str}, ...
            ],
            "perfect_match_rate": float | None,
            "candidate_leak_columns": [str, ...],
            "trustworthy_sample_size": bool,
            "notes": [str, ...],
        }``

        The ``candidate_leak_columns`` field is the practical handle:
        any column the user should look at first. If the list is
        empty AND the perfect-match rate is below threshold, the
        evidence is "metric is suspicious but no obvious source",
        which the agent will surface as ``cannot_determine``.
    """
    notes: list[str] = []
    row_count = int(len(df))
    trustworthy = row_count >= _MIN_TRUSTWORTHY_ROWS
    if not trustworthy:
        notes.append(
            f"Sample size ({row_count}) is below the {_MIN_TRUSTWORTHY_ROWS}-row "
            "trustworthiness threshold; treat all signals below as suggestive only."
        )

    excluded = {c for c in (y_true_col, y_pred_col, y_prob_col) if c is not None}
    candidate_features = [c for c in df.columns if c not in excluded]

    # 1. Target ↔ feature correlations.
    target_correlations = _target_feature_correlations(
        df,
        y_true_col=y_true_col,
        candidate_features=candidate_features,
        task=task,
    )
    candidate_leaks = [
        entry["feature"]
        for entry in target_correlations
        if abs(entry["correlation"]) >= _FEATURE_CORR_THRESHOLD
    ]

    # 2. Exact y_pred == y_true rate.
    perfect_match_rate: float | None = None
    if y_pred_col is not None:
        try:
            y_true = df[y_true_col]
            y_pred = df[y_pred_col]
            if task == "regression":
                # For regression, exact equality is a hard rule
                # (compare numeric values after casting to float).
                a = pd.to_numeric(y_true, errors="coerce")
                b = pd.to_numeric(y_pred, errors="coerce")
                paired = pd.concat([a, b], axis=1).dropna()
                if len(paired) > 0:
                    perfect_match_rate = float((paired.iloc[:, 0] == paired.iloc[:, 1]).mean())
            else:
                perfect_match_rate = float((y_true == y_pred).mean())
        except (KeyError, ValueError):
            notes.append("Could not compute the exact-match rate; column types may differ.")

    return {
        "ok": True,
        "row_count": row_count,
        "trustworthy_sample_size": trustworthy,
        "suspicious_metric": suspicious_metric,
        "target_feature_correlations": target_correlations,
        "perfect_match_rate": perfect_match_rate,
        "candidate_leak_columns": candidate_leaks,
        "notes": notes,
    }


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _target_feature_correlations(
    df: pd.DataFrame,
    *,
    y_true_col: str,
    candidate_features: list[str],
    task: str,
) -> list[dict[str, Any]]:
    """Compute |corr| between y_true and every non-target / non-pred column.

    Implementation notes:

    - **Numeric target** (regression, OR binary 0/1 integer
      classification): plain Pearson. For binary numeric y vs
      continuous x this is exactly the point-biserial r — the right
      statistic, scaled cleanly to ±1 so a leaked feature hits ≈ 1.
      Rank-based "Spearman" would top out around 0.866 for binary
      targets due to tied ranks, missing real leaks.
    - **Non-numeric target** (string class labels): factorize then
      Pearson, which is the rank-based variant we need for arbitrary
      ordering.

    Non-numeric *feature* columns are factorized with ``sort=True`` so
    we still catch a string column that perfectly maps to the label.

    Returns the top-K most correlated features, sorted by absolute
    correlation descending. We keep ``K=10`` so the agent prompt stays
    small even on wide datasets.
    """
    y_raw = df[y_true_col]
    target_is_numeric = bool(pd.api.types.is_numeric_dtype(y_raw))
    out: list[dict[str, Any]] = []

    if target_is_numeric:
        y_series = pd.to_numeric(y_raw, errors="coerce").astype(float)
    else:
        # ``sort=True`` so categorical class labels follow alphabetical
        # order — without sorting, factorize would map "yes"→0 / "no"→1
        # in one file and the opposite in another, flipping the sign.
        y_codes, _ = pd.factorize(y_raw, use_na_sentinel=True, sort=True)
        y_series = pd.Series(y_codes, index=y_raw.index, dtype=float).replace(-1, np.nan)

    for col in candidate_features:
        series = df[col]
        try:
            if pd.api.types.is_numeric_dtype(series):
                feature_series: pd.Series = pd.to_numeric(series, errors="coerce").astype(float)
            else:
                # ``sort=True`` so the integer codes follow the sorted
                # category order — otherwise we'd be measuring a
                # correlation against an arbitrary first-seen mapping.
                codes, _ = pd.factorize(series, use_na_sentinel=True, sort=True)
                feature_series = pd.Series(codes, index=series.index, dtype=float).replace(
                    -1, np.nan
                )

            corr: float | None
            used_method: str
            if target_is_numeric:
                # Compute BOTH plain Pearson and rank-based Pearson
                # (Spearman). Monotone but non-linear leaks (e.g. a
                # feature equal to log(target)) show Pearson ~0.94 yet
                # Spearman = 1.0. Reporting whichever is stronger so a
                # log-of-target leak doesn't slip below the threshold.
                p_linear = _safe_correlation(feature_series, y_series, use_rank=False)
                p_rank = _safe_correlation(feature_series, y_series, use_rank=True)
                pairs: list[tuple[float, str]] = [
                    (c, m)
                    for c, m in ((p_linear, "pearson"), (p_rank, "spearman"))
                    if c is not None and not np.isnan(c)
                ]
                if not pairs:
                    continue
                corr, used_method = max(pairs, key=lambda t: abs(t[0]))
            else:
                # Non-numeric target ⇒ factorized labels, rank-based
                # is the right choice; reporting it directly.
                corr = _safe_correlation(feature_series, y_series, use_rank=True)
                used_method = "spearman"
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if corr is None or np.isnan(corr):
            continue
        out.append({"feature": col, "correlation": float(corr), "method": used_method})

    out.sort(key=lambda d: abs(d["correlation"]), reverse=True)
    return out[:10]


def _safe_correlation(
    feature: pd.Series,
    target: pd.Series,
    *,
    use_rank: bool,
) -> float | None:
    """Pearson (or rank-based Pearson = Spearman) over the aligned pair.

    Scipy-free implementation. Drops aligned NaN before computing, and
    returns ``None`` for degenerate pairs (constant series, fewer than
    two rows) instead of throwing — those just don't make it into the
    evidence dict.
    """
    paired = pd.concat([feature, target], axis=1).dropna()
    if len(paired) < 2:
        return None
    a = paired.iloc[:, 0].to_numpy(dtype=float)
    b = paired.iloc[:, 1].to_numpy(dtype=float)
    if use_rank:
        a = _rankdata(a)
        b = _rankdata(b)
    return _pearson(a, b)


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average-tie ranks. Same convention as ``scipy.stats.rankdata``."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
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


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denom = float(np.sqrt((a_centered**2).sum() * (b_centered**2).sum()))
    if denom == 0:
        return None
    return float((a_centered * b_centered).sum() / denom)


__all__ = ["detect_leakage"]
