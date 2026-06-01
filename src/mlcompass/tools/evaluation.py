"""Post-training evaluation: metrics, confusion matrix, threshold sweep.

Pure ``pandas`` + ``numpy``. Intentionally no scikit-learn dependency —
the formulas we need are short and the smaller install surface matters
for an opt-in advisor CLI.

Supports:
- binary classification     (with optional probability column)
- multi-class classification
- regression

The output dict shape is part of mlcompass's internal contract; the
LLM interpreter and the rich renderer key into the same fields.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Public surface                                                              #
# --------------------------------------------------------------------------- #


SUPPORTED_FORMATS = (".csv", ".parquet", ".xlsx", ".xls", ".jsonl", ".json")

# Column-name heuristics, lower-case match.
_Y_TRUE_HINTS = ("y_true", "label", "target", "actual", "gt", "true", "y")
_Y_PRED_HINTS = ("y_pred", "pred", "prediction", "predicted")
_Y_PROB_HINTS = ("y_prob", "prob", "score", "confidence", "proba")

DEFAULT_THRESHOLD_GRID = (
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.8,
    0.9,
)


class EvaluationError(ValueError):
    """Raised on bad inputs or unsupported formats."""


def load_results(path: Path | str) -> pd.DataFrame:
    """Read a results file by extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise EvaluationError(
        f"Unsupported results format {suffix!r}. Supported: {', '.join(SUPPORTED_FORMATS)}"
    )


def evaluate(
    df: pd.DataFrame,
    *,
    y_true_col: str | None = None,
    y_pred_col: str | None = None,
    y_prob_col: str | None = None,
    task: str | None = None,
    hard_examples_k: int = 5,
    threshold_grid: tuple[float, ...] = DEFAULT_THRESHOLD_GRID,
) -> dict[str, Any]:
    """Run the right evaluation for the inferred task.

    Args:
        df: Predictions table.
        y_true_col / y_pred_col / y_prob_col: Override auto-detection.
        task: ``"binary_classification"`` / ``"multiclass_classification"``
            / ``"regression"``. When ``None``, the task is inferred from
            the values of the detected columns.
        hard_examples_k: How many worst-error rows to surface.
        threshold_grid: Thresholds to sweep for binary classification.

    Returns:
        Structured dict with keys: ``path``, ``task``, ``rows``,
        ``columns``, ``metrics``, ``confusion_matrix``,
        ``threshold_sweep`` (binary only), ``per_class`` (multiclass only),
        ``residuals`` (regression only), ``hard_examples``, ``warnings``.

    Raises:
        EvaluationError: If a column can't be located or task is unknown.
    """
    y_true_col = y_true_col or _detect_column(df, _Y_TRUE_HINTS)
    if y_true_col is None:
        raise EvaluationError(
            "Could not detect a ground-truth column. "
            f"Tried {_Y_TRUE_HINTS}. Pass --y-true to specify."
        )

    y_pred_col = y_pred_col or _detect_column(df, _Y_PRED_HINTS)
    y_prob_col = y_prob_col or _detect_column(df, _Y_PROB_HINTS)

    if y_pred_col is None and y_prob_col is None:
        raise EvaluationError(
            "Could not detect a prediction column. "
            f"Tried {_Y_PRED_HINTS + _Y_PROB_HINTS}. "
            "Pass --y-pred or --y-prob to specify."
        )

    inferred = task or _infer_task(df, y_true_col, y_pred_col, y_prob_col)

    if inferred == "binary_classification":
        result = _evaluate_binary(
            df,
            y_true_col,
            y_pred_col,
            y_prob_col,
            hard_examples_k=hard_examples_k,
            threshold_grid=threshold_grid,
        )
    elif inferred == "multiclass_classification":
        result = _evaluate_multiclass(df, y_true_col, y_pred_col, hard_examples_k=hard_examples_k)
    elif inferred == "regression":
        result = _evaluate_regression(df, y_true_col, y_pred_col, hard_examples_k=hard_examples_k)
    else:
        raise EvaluationError(f"Unknown task type: {inferred}")

    result["task"] = inferred
    result["rows"] = int(len(df))
    result["columns"] = {
        "y_true": y_true_col,
        "y_pred": y_pred_col,
        "y_prob": y_prob_col,
    }

    # v0.7: when the leakage-smell threshold fired, automatically
    # gather structured evidence about WHICH columns might be the
    # source. This stays deterministic (no LLM); the optional
    # ``--llm`` mode then hands the evidence to an investigator
    # agent that narrates it under strict anti-hallucination rules.
    smell = _detect_leakage_smell_metric(result, inferred)
    if smell is not None:
        from .leakage import detect_leakage

        result["leakage_investigation"] = detect_leakage(
            df,
            y_true_col=y_true_col,
            y_pred_col=y_pred_col,
            y_prob_col=y_prob_col,
            task=inferred,
            suspicious_metric=smell,
        )

    return result


def _detect_leakage_smell_metric(
    result: dict[str, Any],
    task: str,
) -> dict[str, Any] | None:
    """Return ``{name, value}`` if the smell threshold fired, else ``None``.

    Drives the auto-leakage-investigation hook above. We sniff the
    ``warnings`` list rather than reach back into the threshold
    constants so the policy stays in one place (the per-task
    too-good-to-be-true helpers).
    """
    warnings = result.get("warnings") or []
    metrics = result.get("metrics") or {}
    smell_keywords = ("Suspiciously", "implausibly", "too good to be true")
    if not any(any(kw.lower() in w.lower() for kw in smell_keywords) for w in warnings):
        return None
    if task == "regression":
        return {"name": "r2", "value": metrics.get("r2")}
    if task == "binary_classification":
        # Pick AUC if present (more sensitive), otherwise accuracy.
        if metrics.get("auc") is not None:
            return {"name": "auc", "value": metrics.get("auc")}
        return {"name": "accuracy", "value": metrics.get("accuracy")}
    if task == "multiclass_classification":
        return {"name": "accuracy", "value": metrics.get("accuracy")}
    return None


# --------------------------------------------------------------------------- #
# Column / task detection                                                     #
# --------------------------------------------------------------------------- #


def _detect_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Return the first column whose lower-cased name matches a candidate."""
    lower_map = {col.lower(): col for col in df.columns}
    for hint in candidates:
        if hint in lower_map:
            return str(lower_map[hint])
    return None


def _infer_task(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str | None,
    y_prob_col: str | None,
) -> str:
    """Heuristically pick the task from the values.

    The ground truth column is authoritative: if ``y_true`` exposes more
    than two distinct labels, the task is multiclass regardless of any
    other signal. Pre-v0.7.2 the analyzer would short-circuit to
    ``binary_classification`` whenever a ``y_prob`` column (even a
    multiclass "max softmax probability" column) was present, which
    surfaced as a hard error during Field Test #4 on the Penguins
    dataset ("Binary task expected 2 distinct labels, got 3"). The
    fix: rank the unique-label signal above the y_prob heuristic.
    """
    y_true = df[y_true_col].dropna()
    if y_true.empty:
        raise EvaluationError(f"Column {y_true_col!r} has no non-null values.")

    nunique = int(y_true.nunique())

    # Numeric target with many distinct values ⇒ regression (unchanged).
    if pd.api.types.is_numeric_dtype(y_true) and nunique > 10:
        return "regression"

    # Three or more distinct labels ⇒ multiclass classification.
    # This takes precedence over the y_prob → binary heuristic because
    # a single "y_prob" column on multiclass output is usually just the
    # max softmax probability, which doesn't change the fact that
    # there are more than two classes.
    if nunique > 2:
        return "multiclass_classification"

    # Two distinct labels OR y_prob in [0, 1] ⇒ binary classification.
    if nunique == 2:
        return "binary_classification"
    if y_prob_col is not None and df[y_prob_col].dropna().between(0, 1, inclusive="both").all():
        return "binary_classification"

    return "multiclass_classification"


# --------------------------------------------------------------------------- #
# Binary classification                                                       #
# --------------------------------------------------------------------------- #


def _evaluate_binary(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str | None,
    y_prob_col: str | None,
    *,
    hard_examples_k: int,
    threshold_grid: tuple[float, ...],
) -> dict[str, Any]:
    rows = df[[c for c in (y_true_col, y_pred_col, y_prob_col) if c is not None]].dropna()
    if len(rows) == 0:
        raise EvaluationError("All prediction rows were null after dropna.")

    y_true_raw = rows[y_true_col]
    positive_label = _binary_positive_label(y_true_raw)
    y_true = (y_true_raw == positive_label).astype(int).to_numpy()

    if y_prob_col is not None:
        y_prob = pd.to_numeric(rows[y_prob_col], errors="coerce").to_numpy()
    else:
        y_prob = None

    if y_pred_col is not None:
        y_pred = (rows[y_pred_col] == positive_label).astype(int).to_numpy()
    elif y_prob is not None:
        y_pred = (y_prob >= 0.5).astype(int)
    else:  # pragma: no cover - guarded earlier
        raise EvaluationError("Need either y_pred or y_prob for binary task.")

    metrics = _binary_metrics(y_true, y_pred)
    confusion = _confusion_matrix_binary(y_true, y_pred)

    sweep: list[dict[str, float]] = []
    best_threshold: dict[str, float] | None = None
    auc: float | None = None
    if y_prob is not None:
        auc = _binary_auc(y_true, y_prob)
        metrics["auc"] = round(auc, 4)
        sweep = _threshold_sweep(y_true, y_prob, threshold_grid)
        best_threshold = max(sweep, key=lambda r: r["f1"])

    warnings = _binary_warnings(metrics, y_true)

    hard = _binary_hard_examples(
        df=df.loc[rows.index],
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        positive_label=positive_label,
        k=hard_examples_k,
    )

    return {
        "metrics": metrics,
        "confusion_matrix": confusion,
        "positive_label": _serialize_label(positive_label),
        "threshold_sweep": sweep,
        "best_threshold": best_threshold,
        "hard_examples": hard,
        "warnings": warnings,
    }


def _binary_positive_label(series: pd.Series) -> Any:
    """Pick the positive class label.

    Order of preference:
    1. ``1`` if labels are ``{0, 1}`` — the universal convention.
    2. ``True`` if labels are ``{True, False}``.
    3. Otherwise the minority class.
    """
    counts = series.value_counts()
    if len(counts) != 2:
        raise EvaluationError(f"Binary task expected 2 distinct labels, got {len(counts)}.")
    unique = set(counts.index)
    if unique <= {0, 1}:
        return 1
    if unique <= {True, False}:
        return True
    return counts.idxmin()


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    return {
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
    }


def _binary_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """ROC AUC via rank-based formula (Wilcoxon–Mann–Whitney statistic)."""
    if np.all(y_true == 0) or np.all(y_true == 1):
        return float("nan")
    order = np.argsort(y_prob)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_prob) + 1)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    sum_ranks_pos = float(ranks[y_true == 1].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    grid: tuple[float, ...],
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for thr in grid:
        y_pred = (y_prob >= thr).astype(int)
        metrics = _binary_metrics(y_true, y_pred)
        rows.append({"threshold": float(thr), **metrics})
    return rows


def _binary_warnings(metrics: dict[str, float], y_true: np.ndarray) -> list[str]:
    warnings: list[str] = []

    positive_rate = float(y_true.mean()) if len(y_true) else 0.0
    n_rows = int(len(y_true))

    if positive_rate < 0.1 and metrics["accuracy"] > 0.9 and metrics["recall"] < 0.5:
        warnings.append(
            f"Class imbalance ({positive_rate * 100:.1f}% positive) inflates "
            "accuracy — recall is the metric to watch."
        )

    if metrics["precision"] < 0.2 and metrics["recall"] > 0.7:
        warnings.append(
            "Recall is high but precision is very low — likely too-aggressive "
            "threshold. Try raising it."
        )
    elif metrics["recall"] < 0.2 and metrics["precision"] > 0.7:
        warnings.append(
            "Precision is high but recall is very low — likely too-conservative "
            "threshold. Try lowering it."
        )

    too_good = _too_good_to_be_true_binary(metrics, n_rows)
    if too_good:
        warnings.append(too_good)

    return warnings


def _too_good_to_be_true_binary(
    metrics: dict[str, float],
    n_rows: int,
) -> str | None:
    """Flag near-perfect metrics as a leakage / split-contamination smell."""
    if n_rows < 50:
        return None  # tiny test sets often look perfect by chance

    issues: list[str] = []
    auc = metrics.get("auc")
    if auc is not None and auc > 0.995:
        issues.append(f"AUC {auc:.4f}")
    if metrics["accuracy"] > 0.99:
        issues.append(f"accuracy {metrics['accuracy']:.4f}")
    if metrics["precision"] >= 0.99 and metrics["recall"] >= 0.99:
        issues.append("precision and recall both ≥ 0.99")
    if not issues:
        return None

    return (
        f"Suspiciously perfect metrics ({', '.join(issues)}). On real-world "
        "data this is almost always a sign of one of: (1) data leakage — the "
        "target value or a near-perfect proxy is in the features, (2) train/"
        "test contamination — the same rows appear in both splits, (3) the "
        "wrong column is being used as y_true / y_pred / y_prob, or (4) the "
        "predictions table was generated on the training set instead of the "
        "held-out set. Sanity-check before believing the score."
    )


def _binary_hard_examples(
    *,
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    positive_label: Any,
    k: int,
) -> list[dict[str, Any]]:
    if y_prob is not None:
        # Distance from "correct probability" — i.e. |y_true - y_prob|.
        score = np.abs(y_true.astype(float) - y_prob)
    else:
        # Treat any misclassification as the same magnitude error.
        score = (y_true != y_pred).astype(float)

    rows: list[dict[str, Any]] = []
    order = np.argsort(-score)
    for idx in order[:k]:
        row_record = df.iloc[int(idx)].to_dict()
        rows.append(
            {
                "row_index": int(df.index[int(idx)]),
                "true_label": _serialize_label(_invert_binary(y_true[idx], positive_label)),
                "predicted_label": _serialize_label(_invert_binary(y_pred[idx], positive_label)),
                "probability": (float(y_prob[idx]) if y_prob is not None else None),
                "row": _serialize_row(row_record),
            }
        )
    return rows


def _invert_binary(value: int, positive_label: Any) -> Any:
    return positive_label if value == 1 else _negation_of(positive_label)


def _negation_of(positive_label: Any) -> Any:
    if positive_label in (0, 1):
        return 0 if positive_label == 1 else 1
    if isinstance(positive_label, bool):
        return not positive_label
    return f"not_{positive_label}"


# --------------------------------------------------------------------------- #
# Multiclass classification                                                   #
# --------------------------------------------------------------------------- #


def _evaluate_multiclass(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str | None,
    *,
    hard_examples_k: int,
) -> dict[str, Any]:
    if y_pred_col is None:
        raise EvaluationError("Multiclass evaluation requires a y_pred column.")
    rows = df[[y_true_col, y_pred_col]].dropna()
    if len(rows) == 0:
        raise EvaluationError("All prediction rows were null after dropna.")

    y_true = rows[y_true_col].to_numpy()
    y_pred = rows[y_pred_col].to_numpy()

    labels = sorted(set(y_true) | set(y_pred), key=_label_sort_key)

    per_class: list[dict[str, Any]] = []
    macro_f1 = 0.0
    weighted_f1 = 0.0
    total = len(y_true)
    for label in labels:
        tp = int(((y_true == label) & (y_pred == label)).sum())
        fp = int(((y_true != label) & (y_pred == label)).sum())
        fn = int(((y_true == label) & (y_pred != label)).sum())
        support = int((y_true == label).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class.append(
            {
                "label": _serialize_label(label),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": support,
            }
        )
        macro_f1 += f1
        weighted_f1 += f1 * support

    accuracy = float((y_true == y_pred).mean()) if total else 0.0
    if labels:
        macro_f1 /= len(labels)
        weighted_f1 /= total if total else 1

    confusion = _confusion_matrix_dense(y_true, y_pred, labels)
    hard = _multiclass_hard_examples(df.loc[rows.index], y_true, y_pred, hard_examples_k)

    overall_metrics = {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
    }
    return {
        "metrics": overall_metrics,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "labels": [_serialize_label(label) for label in labels],
        "hard_examples": hard,
        "warnings": _multiclass_warnings(per_class, overall_metrics, total),
    }


def _confusion_matrix_dense(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[Any],
) -> list[list[int]]:
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for actual, predicted in zip(y_true, y_pred, strict=True):
        matrix[index[actual], index[predicted]] += 1
    return matrix.tolist()


def _multiclass_warnings(
    per_class: list[dict[str, Any]],
    overall_metrics: dict[str, float],
    n_rows: int,
) -> list[str]:
    warnings: list[str] = []
    very_weak = [c for c in per_class if c["f1"] < 0.3 and c["support"] >= 5]
    if very_weak:
        labels = ", ".join(str(c["label"]) for c in very_weak[:3])
        warnings.append(
            f"{len(very_weak)} class(es) have F1 < 0.3 despite having "
            f"≥5 examples ({labels}…). Look at confusion matrix rows."
        )

    too_good = _too_good_to_be_true_multiclass(per_class, overall_metrics, n_rows)
    if too_good:
        warnings.append(too_good)
    return warnings


def _too_good_to_be_true_multiclass(
    per_class: list[dict[str, Any]],
    overall_metrics: dict[str, float],
    n_rows: int,
) -> str | None:
    if n_rows < 50:
        return None

    issues: list[str] = []
    if overall_metrics.get("accuracy", 0) > 0.99:
        issues.append(f"accuracy {overall_metrics['accuracy']:.4f}")
    strong_classes = [c for c in per_class if c["support"] >= 5 and c["f1"] > 0.99]
    if per_class and len(strong_classes) == len(per_class):
        issues.append("every class F1 ≥ 0.99")
    if not issues:
        return None

    return (
        f"Suspiciously perfect metrics ({', '.join(issues)}). On real-world "
        "data this is almost always a sign of data leakage, train/test "
        "contamination, or the predictions table being generated on the "
        "training set. Sanity-check before believing the score."
    )


def _multiclass_hard_examples(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int,
) -> list[dict[str, Any]]:
    wrong_mask = y_true != y_pred
    if not wrong_mask.any():
        return []
    wrong_indices = np.flatnonzero(wrong_mask)
    rows: list[dict[str, Any]] = []
    for idx in wrong_indices[:k]:
        record = df.iloc[int(idx)].to_dict()
        rows.append(
            {
                "row_index": int(df.index[int(idx)]),
                "true_label": _serialize_label(y_true[idx]),
                "predicted_label": _serialize_label(y_pred[idx]),
                "row": _serialize_row(record),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Regression                                                                  #
# --------------------------------------------------------------------------- #


def _evaluate_regression(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str | None,
    *,
    hard_examples_k: int,
) -> dict[str, Any]:
    if y_pred_col is None:
        raise EvaluationError("Regression evaluation requires a y_pred column.")
    rows = df[[y_true_col, y_pred_col]].dropna()
    if len(rows) == 0:
        raise EvaluationError("All prediction rows were null after dropna.")

    y_true = pd.to_numeric(rows[y_true_col], errors="coerce").to_numpy()
    y_pred = pd.to_numeric(rows[y_pred_col], errors="coerce").to_numpy()
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    residuals = y_true - y_pred
    n = len(residuals)
    if n == 0:
        raise EvaluationError("No numeric prediction rows after coercion.")

    mae = float(np.mean(np.abs(residuals)))
    rmse = float(math.sqrt(np.mean(residuals**2)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0

    # Hard examples — largest absolute residuals.
    sub_df = df.loc[rows.index[mask]] if mask.all() else df.loc[rows.index].iloc[mask]
    order = np.argsort(-np.abs(residuals))
    hard: list[dict[str, Any]] = []
    for idx in order[:hard_examples_k]:
        record = sub_df.iloc[int(idx)].to_dict()
        hard.append(
            {
                "row_index": int(sub_df.index[int(idx)]),
                "y_true": float(y_true[idx]),
                "y_pred": float(y_pred[idx]),
                "residual": float(residuals[idx]),
                "row": _serialize_row(record),
            }
        )

    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
    }
    warnings = _regression_warnings(residuals, y_true, r2)

    return {
        "metrics": metrics,
        "residuals": {
            "mean": round(float(np.mean(residuals)), 4),
            "std": round(float(np.std(residuals)), 4),
            "min": round(float(np.min(residuals)), 4),
            "max": round(float(np.max(residuals)), 4),
            "count": int(n),
        },
        "hard_examples": hard,
        "warnings": warnings,
    }


def _regression_warnings(
    residuals: np.ndarray,
    y_true: np.ndarray,
    r2: float,
) -> list[str]:
    warnings: list[str] = []
    bias = float(np.mean(residuals))
    spread = float(np.std(residuals)) or 1.0
    if abs(bias) > 0.5 * spread:
        warnings.append(
            f"Residuals are biased ({bias:+.3f}, ~{abs(bias) / spread:.0%} "
            "of the residual std). Model is systematically over/under-predicting."
        )
    target_range = float(y_true.max() - y_true.min())
    if target_range and spread / target_range > 0.5:
        warnings.append(
            "Residual spread is large relative to the target range; the model "
            "may not be capturing the signal yet."
        )

    too_good = _too_good_to_be_true_regression(r2, len(residuals))
    if too_good:
        warnings.append(too_good)
    return warnings


def _too_good_to_be_true_regression(r2: float, n: int) -> str | None:
    if n < 50 or r2 <= 0.999:
        return None
    return (
        f"Suspiciously high R² ({r2:.4f}). On real-world data this almost "
        "always means data leakage (target value present in features), train/"
        "test contamination, or that the predictions table was scored on the "
        "training set. Sanity-check before believing the fit."
    )


# --------------------------------------------------------------------------- #
# Serialisation helpers                                                       #
# --------------------------------------------------------------------------- #


def _serialize_label(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _serialize_row(record: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in record.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, pd.Timestamp):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out


def _label_sort_key(value: Any) -> Any:
    """Stable sort key for mixed-type labels in multiclass."""
    if isinstance(value, str):
        return (1, value)
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (2, str(value))
