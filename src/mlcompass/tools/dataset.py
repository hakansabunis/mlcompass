"""Dataset analysis: schema, distribution, outliers, target detection.

Pure pandas — no LLM calls. The output is a structured dictionary that
the ModelAdvisor sub-agent consumes during ``mlcompass advise``.

The shape and field names of the returned dict are part of mlcompass's
internal contract; downstream agents key into them, so changes should
be made deliberately and reflected in ARCHITECTURE.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

SUPPORTED_FORMATS = {".csv", ".parquet", ".xlsx", ".xls", ".jsonl", ".json"}

# Column-name heuristics for target detection. Lower-case match.
#
# History of the lists:
# - v0.1: classification-only basics (``target``, ``label``, ``y``,
#   ``churn``, ``fraud``, …).
# - v0.7: regression Kaggle conventions added (``saleprice``,
#   ``price``, ``amount``, …) after Field Test #2 on Ames House Prices
#   found the analyzer couldn't auto-detect an obvious regression target.
# - v0.7.1: classic Kaggle binary classification names added (``survived``,
#   ``purchased``, ``clicked``, …) after Field Test #3 on the Titanic
#   dataset found the analyzer was falling back to the last column for a
#   target name that ought to be unmistakable.
TARGET_NAME_HINTS: dict[str, list[str]] = {
    "high_confidence": [
        # Generic
        "target",
        "label",
        "y",
        # Classification — flagship binary names every Kaggle starter
        # dataset uses. ``survived`` (Titanic), ``churn`` (telco),
        # ``fraud`` (cards), and the ``is_*`` variants of each are the
        # ones we see most consistently on real datasets.
        "churn",
        "fraud",
        "default",
        "is_fraud",
        "is_churn",
        "survived",
        "is_survived",
        "purchased",
        "is_purchase",
        "is_purchased",
        "clicked",
        "is_clicked",
        "converted",
        "is_converted",
        "accepted",
        "approved",
        "winner",
        # Multiclass classification — Penguins (FT#4), MNIST, sentiment,
        # genre, intent, language — the canonical multiclass labels
        # across image / NLP / biology Kaggle competitions.
        "species",
        "class_label",
        "category_label",
        "digit",
        "genre",
        "sentiment",
        "intent",
        "language",
        # Regression — common Kaggle / public-dataset spellings.
        # ``charges`` was added in v0.7.3 after Field Test #5 on the
        # Kaggle Insurance Charges dataset surfaced that this canonical
        # healthcare-cost target name was missing from the list.
        "saleprice",
        "sale_price",
        "price",
        "saleamount",
        "sale_amount",
        "charges",
        "total_charges",
        "medical_cost",
    ],
    "medium_confidence": [
        # Classification — softer signals
        "outcome",
        "result",
        "class",
        "category",
        "response",
        "engagement",
        "subscribed",
        "active",
        # Regression — softer signals
        "amount",
        "value",
        "score",
        "revenue",
        "cost",
        "salary",
        "rating",
        # v0.7.3: more finance / healthcare-style regression targets.
        "fee",
        "tuition",
        "expenses",
        "arpu",
        "spend",
    ],
}


# --------------------------------------------------------------------------- #
# I/O                                                                         #
# --------------------------------------------------------------------------- #


def load_dataset(path: Path | str) -> pd.DataFrame:
    """Load a dataset from disk, auto-detecting the format from extension.

    Supported extensions: ``.csv``, ``.parquet``, ``.xlsx``, ``.xls``,
    ``.jsonl``, ``.json``.

    Raises:
        ValueError: If the file extension is not recognised.
    """
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

    raise ValueError(
        f"Unsupported dataset format: {suffix!r}. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
    )


# --------------------------------------------------------------------------- #
# Top-level analyzer                                                          #
# --------------------------------------------------------------------------- #


def analyze_dataset(
    path: Path | str,
    *,
    target_column: str | None = None,
    sample_rows: int | None = None,
) -> dict[str, Any]:
    """Produce a structured analysis of a dataset on disk.

    Args:
        path: Path to the data file.
        target_column: If known, the target column name. If ``None``, the
            analyzer will try to detect it heuristically.
        sample_rows: If set, only the first ``N`` rows are analyzed.
            Useful for very large files (>10GB) where a full pass is
            wasteful for the advisor's needs.

    Returns:
        A dict with the following top-level keys::

            {
              "path": str,
              "format": str,
              "shape": {"rows": int, "cols": int},
              "columns": [<column_analysis>, ...],
              "target_hint": {"column": str | None,
                              "reason": str,
                              "confidence": "explicit" | "high" | "medium" | "low" | "none"},
              "task_hint": {"type": ..., ...},
              "warnings": [str, ...],
            }
    """
    path = Path(path)
    df = load_dataset(path)

    if sample_rows is not None:
        df = df.head(sample_rows)

    columns = [_analyze_column(df, col) for col in df.columns]

    if target_column is not None:
        target_hint: dict[str, Any] = {
            "column": target_column,
            "reason": "User-specified via --target",
            "confidence": "explicit",
        }
    else:
        target_hint = _detect_target_column(df)

    task_hint = _infer_task_type(df, target_hint)
    warnings = _generate_warnings(columns, target_hint, task_hint)

    return {
        "path": str(path),
        "format": path.suffix.lstrip("."),
        "shape": {"rows": int(len(df)), "cols": int(len(df.columns))},
        "columns": columns,
        "target_hint": target_hint,
        "task_hint": task_hint,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# Column-level analysis                                                       #
# --------------------------------------------------------------------------- #


_YEAR_NAME_HINTS = ("year", "yr", "_yr_", "_year_")
# The plausible-year range we expect human-recorded year values to
# fall inside. Values *outside* this band are passed through unchanged
# but caught later by the data-quality warning.
_YEAR_VALID_RANGE = (1800.0, 2200.0)


def _looks_like_year_column(name: str, series: pd.Series) -> bool:
    """True if the column's NAME suggests a year AND its values look year-shaped.

    Used by ``_classify_column`` to override the otherwise-categorical
    "2 distinct values" rule for the Ames ``Yr Sold`` (2006-2010)
    pattern — small year ranges are still ordered + temporal, not
    nominal categories.
    """
    lower = name.lower()
    if not any(h in lower for h in _YEAR_NAME_HINTS):
        return False
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    try:
        nmin = float(non_null.min())
    except (TypeError, ValueError):
        return False
    lo, hi = _YEAR_VALID_RANGE
    # Accept if the column's minimum value lands in the plausible
    # range — Ames has `Garage Yr Blt` max=2207 (data error) but the
    # bulk is fine, and we only care that this is *temporal*, not
    # that every value is sane.
    return nmin >= lo and nmin <= hi


def _classify_column(series: pd.Series, *, name: str = "") -> str:
    """Return one of: ``numeric``, ``categorical``, ``datetime``, ``boolean``, ``text``."""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        # Field-test UX #2: numeric columns that take only two distinct
        # values are almost always categorical flags (SeniorCitizen
        # pattern). Treat them as categorical so the summariser doesn't
        # report the nonsense "1142 IQR outliers" you get when you run
        # a quantile-based outlier detector on a 0/1 vector.
        #
        # FT#2 follow-up: Year columns (``Yr Sold``, ``Year Built``)
        # often have only a handful of distinct integer values too,
        # but they're ordered temporal data — not nominal categories.
        # If the name matches our year-hint patterns AND the values
        # land in the plausible year range, keep them numeric.
        non_null = series.dropna()
        if (
            len(non_null) > 0
            and non_null.nunique() == 2
            and not _looks_like_year_column(name, series)
        ):
            return "categorical"
        return "numeric"

    # Object dtype: distinguish text from categorical by cardinality.
    #
    # Heuristic, applied in order:
    #   - Empty → call it categorical (best-effort guess, no signal to use).
    #   - nunique <= 10 → categorical regardless of ratio (small low-cardinality
    #     columns survive even tiny datasets where ratio is necessarily high).
    #   - nunique >= 500 → text (almost certainly IDs or free-form).
    #   - ratio > 0.5 → text (most values unique, ID-like).
    #   - Otherwise → categorical (possibly high-cardinality, flagged later).
    non_null = series.dropna()
    if len(non_null) == 0:
        return "categorical"

    nunique = non_null.nunique()
    if nunique <= 10:
        return "categorical"
    if nunique >= 500:
        return "text"
    if nunique / len(non_null) > 0.5:
        return "text"
    return "categorical"


def _looks_numeric_with_dirty_strings(series: pd.Series) -> bool:
    """True if a text column's values are mostly numbers in disguise.

    Detects the classic Telco-style ``TotalCharges`` pattern: a column
    that should be numeric but ends up as ``object`` dtype because a
    handful of rows have empty strings or stray whitespace. We
    consider the column numeric-in-disguise when **≥ 95%** of
    non-empty values parse as floats.

    Lives separately from ``_classify_column`` so we can flag the
    column without changing how the analyzer summarises it: it still
    needs the text summariser to surface the cardinality / length, but
    the warning tells the user it should be cleaned.
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    coerced = pd.to_numeric(non_null, errors="coerce")
    parsed = coerced.notna().sum()
    # Skip columns that have nothing numeric-looking at all — they're
    # just normal text. We need a strong majority to call this out.
    if parsed == 0:
        return False
    return float(parsed) / float(len(non_null)) >= 0.95


def _analyze_column(df: pd.DataFrame, col: str) -> dict[str, Any]:
    """Analyze a single column and return a typed summary dict."""
    s = df[col]
    col_type = _classify_column(s, name=col)

    base: dict[str, Any] = {
        "name": col,
        "type": col_type,
        "dtype": str(s.dtype),
        "missing_count": int(s.isna().sum()),
        "missing_pct": float(s.isna().mean()),
    }

    if col_type == "numeric":
        base.update(_summarize_numeric(s))
    elif col_type == "categorical":
        base.update(_summarize_categorical(s))
    elif col_type == "datetime":
        base.update(_summarize_datetime(s))
    elif col_type == "boolean":
        base.update(_summarize_boolean(s))
    elif col_type == "text":
        base.update(_summarize_text(s))

    return base


def _summarize_numeric(s: pd.Series) -> dict[str, Any]:
    """Numeric column: descriptive stats + IQR and Z-score outlier counts.

    FT#2 follow-up: heavily zero-dominated columns (Ames ``Open Porch SF``,
    ``Enclosed Porch``, …) trigger meaningless IQR outlier counts —
    the quartile bands collapse around zero and every non-zero value
    falls "outside". When >50% of values are exactly zero we tag the
    column as ``sparse`` and skip the IQR / Z-score counters; the
    advisor reader gets a clean ``stats`` block instead of misleading
    "459 IQR outliers" reports.
    """
    s_clean = s.dropna()
    if len(s_clean) == 0:
        return {"stats": None, "outliers": None}

    q = s_clean.quantile([0.25, 0.5, 0.75])
    stats = {
        "mean": float(s_clean.mean()),
        "std": float(s_clean.std()),
        "min": float(s_clean.min()),
        "max": float(s_clean.max()),
        "q25": float(q[0.25]),
        "q50": float(q[0.5]),
        "q75": float(q[0.75]),
    }

    zero_ratio = float((s_clean == 0).mean())
    if zero_ratio > 0.5:
        return {
            "stats": stats,
            "outliers": None,
            "sparse": True,
            "zero_ratio": zero_ratio,
        }

    # IQR outliers
    iqr = q[0.75] - q[0.25]
    lower = q[0.25] - 1.5 * iqr
    upper = q[0.75] + 1.5 * iqr
    iqr_count = int(((s_clean < lower) | (s_clean > upper)).sum())

    # Z-score outliers (|z| > 3) — only meaningful when std > 0
    std = s_clean.std()
    if std > 0:
        z = (s_clean - s_clean.mean()).abs() / std
        z_count = int((z > 3).sum())
    else:
        z_count = 0

    return {
        "stats": stats,
        "outliers": {"iqr_count": iqr_count, "z_score_count": z_count},
        "sparse": False,
        "zero_ratio": zero_ratio,
    }


def _summarize_categorical(s: pd.Series) -> dict[str, Any]:
    """Categorical column: cardinality + top-5 values."""
    s_clean = s.dropna()
    if len(s_clean) == 0:
        return {"cardinality": 0, "top_values": []}

    top = s_clean.value_counts().head(5)
    return {
        "cardinality": int(s_clean.nunique()),
        "top_values": [{"value": str(v), "count": int(c)} for v, c in top.items()],
    }


def _summarize_datetime(s: pd.Series) -> dict[str, Any]:
    """Datetime column: min / max date range."""
    s_clean = s.dropna()
    if len(s_clean) == 0:
        return {"range": None}
    return {
        "range": {
            "min": str(s_clean.min()),
            "max": str(s_clean.max()),
        }
    }


def _summarize_boolean(s: pd.Series) -> dict[str, Any]:
    """Boolean column: true/false counts."""
    s_clean = s.dropna()
    true_count = int(s_clean.astype(bool).sum())
    false_count = int(len(s_clean) - true_count)
    return {"true_count": true_count, "false_count": false_count}


def _summarize_text(s: pd.Series) -> dict[str, Any]:
    """Free-text column: average length + cardinality.

    Also stamps two booleans the warning layer reads:

    - ``looks_numeric``: the column is object-dtype but ≥95% of
      values parse as floats (Telco ``TotalCharges`` pattern).
    - ``looks_like_id``: cardinality equals row count, suggesting a
      unique-per-row identifier (``customerID`` pattern).
    """
    s_clean = s.dropna().astype(str)
    if len(s_clean) == 0:
        return {
            "avg_length": 0.0,
            "cardinality": 0,
            "looks_numeric": False,
            "looks_like_id": False,
        }
    cardinality = int(s_clean.nunique())
    return {
        "avg_length": float(s_clean.str.len().mean()),
        "cardinality": cardinality,
        "looks_numeric": _looks_numeric_with_dirty_strings(s),
        "looks_like_id": cardinality == len(s_clean),
    }


# --------------------------------------------------------------------------- #
# Target + task inference                                                     #
# --------------------------------------------------------------------------- #


def _detect_target_column(df: pd.DataFrame) -> dict[str, Any]:
    """Heuristically detect the most likely target column.

    Strategy (in order):
      1. Exact lower-case name match against the high-confidence list.
      2. Same against the medium-confidence list.
      3. Fall back to the last column if it has low cardinality.
      4. Otherwise return ``column=None`` and ask the user to specify.
    """
    cols_lower = {col.lower(): col for col in df.columns}

    for name in TARGET_NAME_HINTS["high_confidence"]:
        if name in cols_lower:
            return {
                "column": cols_lower[name],
                "reason": f"Column '{cols_lower[name]}' matches common target names",
                "confidence": "high",
            }

    for name in TARGET_NAME_HINTS["medium_confidence"]:
        if name in cols_lower:
            return {
                "column": cols_lower[name],
                "reason": f"Column '{cols_lower[name]}' is a likely target",
                "confidence": "medium",
            }

    # Last-column fallback if low cardinality (a common convention)
    last_col = df.columns[-1]
    nunique = df[last_col].nunique(dropna=True)
    if nunique <= 10:
        return {
            "column": last_col,
            "reason": (
                f"Last column '{last_col}' has low cardinality ({nunique}), often the target"
            ),
            "confidence": "low",
        }

    return {
        "column": None,
        "reason": "No clear target column detected. Specify with --target.",
        "confidence": "none",
    }


def _infer_task_type(df: pd.DataFrame, target_hint: dict[str, Any]) -> dict[str, Any]:
    """Infer ML task type from the detected target column."""
    target_col = target_hint.get("column")
    if target_col is None or target_col not in df.columns:
        return {"type": "unknown", "reason": "No target column"}

    s = df[target_col].dropna()
    if len(s) == 0:
        return {"type": "unknown", "reason": "Target column is empty"}

    nunique = s.nunique()

    if pd.api.types.is_numeric_dtype(s) and nunique > 10:
        return {
            "type": "regression",
            "target_stats": {
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "max": float(s.max()),
            },
        }

    if nunique == 2:
        balance = {str(k): float(v) for k, v in s.value_counts(normalize=True).items()}
        return {"type": "binary_classification", "class_balance": balance}

    if 2 < nunique <= 20:
        distribution = {str(k): float(v) for k, v in s.value_counts(normalize=True).items()}
        return {
            "type": "multiclass_classification",
            "n_classes": int(nunique),
            "class_distribution": distribution,
        }

    return {
        "type": "unknown",
        "reason": f"Target has {nunique} unique values, task type ambiguous",
    }


# --------------------------------------------------------------------------- #
# Warnings                                                                    #
# --------------------------------------------------------------------------- #


def _generate_warnings(
    columns: list[dict[str, Any]],
    target_hint: dict[str, Any],
    task_hint: dict[str, Any],
) -> list[str]:
    """Surface high-level issues for the advisor to highlight."""
    warnings: list[str] = []

    high_missing = [c for c in columns if c["missing_pct"] > 0.5]
    if high_missing:
        names = ", ".join(c["name"] for c in high_missing[:5])
        suffix = "..." if len(high_missing) > 5 else ""
        warnings.append(
            f"{len(high_missing)} column(s) have >50% missing values "
            f"({names}{suffix}); consider dropping or careful imputation."
        )

    medium_missing = [c for c in columns if 0.1 < c["missing_pct"] <= 0.5]
    if medium_missing:
        warnings.append(
            f"{len(medium_missing)} column(s) have 10-50% missing values; "
            "decide on an imputation strategy."
        )

    high_card = [c for c in columns if c["type"] == "categorical" and c.get("cardinality", 0) > 50]
    if high_card:
        warnings.append(
            f"{len(high_card)} categorical column(s) have cardinality >50; "
            "consider target encoding or top-N grouping."
        )

    # Field-test UX #1: text columns whose values are mostly numbers
    # ("TotalCharges" pattern). Almost always a data-cleaning miss.
    dirty_numeric = [c for c in columns if c["type"] == "text" and c.get("looks_numeric")]
    if dirty_numeric:
        names = ", ".join(c["name"] for c in dirty_numeric[:5])
        suffix = "..." if len(dirty_numeric) > 5 else ""
        warnings.append(
            f"{len(dirty_numeric)} column(s) look numeric but contain "
            f"non-numeric values ({names}{suffix}); clean before "
            "training — likely empty strings or stray whitespace."
        )

    # Field-test UX #3: unique-per-row text columns ("customerID"
    # pattern). Strong signal of an ID; using it as a feature
    # memorises the training set.
    id_like = [c for c in columns if c["type"] == "text" and c.get("looks_like_id")]
    if id_like:
        names = ", ".join(c["name"] for c in id_like[:5])
        suffix = "..." if len(id_like) > 5 else ""
        warnings.append(
            f"{len(id_like)} column(s) look like unique-per-row "
            f"identifiers ({names}{suffix}); drop before training so "
            "the model doesn't memorise row IDs."
        )

    # FT#2 follow-up: numeric columns where >50% of values are zero
    # are sparse — Ames ``Open Porch SF``, ``Pool Area``, …. Treating
    # them like normal numerics gives misleading IQR outlier counts;
    # the right move is to either binarise (present / absent) or use a
    # zero-inflated model. Surface this so the advisor can advise.
    sparse_numeric = [c for c in columns if c["type"] == "numeric" and c.get("sparse")]
    if sparse_numeric:
        names = ", ".join(c["name"] for c in sparse_numeric[:5])
        suffix = "..." if len(sparse_numeric) > 5 else ""
        warnings.append(
            f"{len(sparse_numeric)} numeric column(s) are sparse "
            f"(majority-zero) ({names}{suffix}); consider binarising "
            "(presence/absence) or using a zero-inflated model — "
            "outlier statistics are not meaningful here."
        )

    if target_hint.get("confidence") == "none":
        warnings.append("No target column auto-detected. Specify with --target <name>.")

    if task_hint.get("type") == "binary_classification":
        balance = task_hint.get("class_balance", {})
        if balance:
            minority = min(balance.values())
            if minority < 0.2:
                warnings.append(
                    f"Class imbalance detected ({minority * 100:.1f}% minority "
                    "class). Don't optimise accuracy — use AUC/F1/recall@k. "
                    "Consider class_weight='balanced' or focal loss."
                )

    return warnings
