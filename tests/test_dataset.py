"""Tests for the dataset analyzer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mlcompass.tools.dataset import analyze_dataset, load_dataset


def _csv(tmp_path: Path, df: pd.DataFrame, name: str = "data.csv") -> Path:
    """Helper: write DataFrame to a CSV under tmp_path and return the path."""
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


# --------------------------------------------------------------------------- #
# load_dataset                                                                #
# --------------------------------------------------------------------------- #


def test_load_csv(tmp_path: Path) -> None:
    p = _csv(tmp_path, pd.DataFrame({"a": [1, 2, 3]}))
    df = load_dataset(p)
    assert list(df.columns) == ["a"]
    assert len(df) == 3


def test_load_parquet(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    df_in = pd.DataFrame({"a": [1, 2, 3]})
    p = tmp_path / "data.parquet"
    df_in.to_parquet(p)
    df_out = load_dataset(p)
    assert list(df_out.columns) == ["a"]


def test_load_unsupported_format_raises(tmp_path: Path) -> None:
    p = tmp_path / "data.xyz"
    p.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        load_dataset(p)


# --------------------------------------------------------------------------- #
# analyze_dataset — basic shape                                               #
# --------------------------------------------------------------------------- #


def test_analyze_returns_top_level_keys(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    result = analyze_dataset(_csv(tmp_path, df))

    for key in ("path", "format", "shape", "columns", "target_hint", "task_hint", "warnings"):
        assert key in result


def test_analyze_basic_shape(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
    result = analyze_dataset(_csv(tmp_path, df))

    assert result["shape"] == {"rows": 4, "cols": 2}
    assert result["format"] == "csv"
    assert len(result["columns"]) == 2


def test_analyze_sample_rows_limits_analysis(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": list(range(1000))})
    result = analyze_dataset(_csv(tmp_path, df), sample_rows=100)
    assert result["shape"]["rows"] == 100


# --------------------------------------------------------------------------- #
# Column classification                                                       #
# --------------------------------------------------------------------------- #


def test_numeric_column_has_stats_and_outliers(tmp_path: Path) -> None:
    df = pd.DataFrame({"age": [25, 30, 35, 40, 45]})
    col = analyze_dataset(_csv(tmp_path, df))["columns"][0]

    assert col["type"] == "numeric"
    assert col["stats"]["min"] == 25.0
    assert col["stats"]["max"] == 45.0
    assert "outliers" in col
    assert "iqr_count" in col["outliers"]
    assert "z_score_count" in col["outliers"]


def test_categorical_column_has_cardinality_and_top_values(tmp_path: Path) -> None:
    df = pd.DataFrame({"country": ["US", "GB", "US", "FR", "US"]})
    col = analyze_dataset(_csv(tmp_path, df))["columns"][0]

    assert col["type"] == "categorical"
    assert col["cardinality"] == 3
    assert len(col["top_values"]) == 3
    # Most common value is 'US' (3 occurrences)
    assert col["top_values"][0]["value"] == "US"
    assert col["top_values"][0]["count"] == 3


def test_datetime_column_has_range(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    df = pd.DataFrame({"event_date": pd.to_datetime(["2024-01-01", "2024-06-15", "2024-12-31"])})
    p = tmp_path / "data.parquet"  # CSV would lose dtype info
    df.to_parquet(p)

    col = analyze_dataset(p)["columns"][0]
    assert col["type"] == "datetime"
    assert col["range"] is not None
    assert "2024" in col["range"]["min"]


def test_text_column_has_avg_length(tmp_path: Path) -> None:
    df = pd.DataFrame({"review": [f"This is review number {i}" * 3 for i in range(200)]})
    col = analyze_dataset(_csv(tmp_path, df))["columns"][0]
    assert col["type"] == "text"
    assert col["avg_length"] > 0


# --------------------------------------------------------------------------- #
# Missing data                                                                #
# --------------------------------------------------------------------------- #


def test_missing_data_counted(tmp_path: Path) -> None:
    df = pd.DataFrame({"col": [1, 2, None, None, None]})  # 60% missing
    col = analyze_dataset(_csv(tmp_path, df))["columns"][0]

    assert col["missing_count"] == 3
    assert col["missing_pct"] == pytest.approx(0.6)


def test_high_missing_triggers_warning(tmp_path: Path) -> None:
    df = pd.DataFrame({"col": [1, 2, None, None, None]})
    result = analyze_dataset(_csv(tmp_path, df))
    assert any("missing" in w.lower() for w in result["warnings"])


# --------------------------------------------------------------------------- #
# Target detection                                                            #
# --------------------------------------------------------------------------- #


def test_high_confidence_target_via_name_match(tmp_path: Path) -> None:
    df = pd.DataFrame({"feature": [1, 2, 3, 4], "churn": [0, 1, 0, 1]})
    result = analyze_dataset(_csv(tmp_path, df))

    assert result["target_hint"]["column"] == "churn"
    assert result["target_hint"]["confidence"] == "high"


def test_explicit_target_overrides_detection(tmp_path: Path) -> None:
    df = pd.DataFrame({"feature": [1, 2, 3, 4], "churn": [0, 1, 0, 1], "y_custom": [0, 1, 1, 0]})
    result = analyze_dataset(_csv(tmp_path, df), target_column="y_custom")

    assert result["target_hint"]["column"] == "y_custom"
    assert result["target_hint"]["confidence"] == "explicit"


def test_low_confidence_fallback_to_last_column(tmp_path: Path) -> None:
    df = pd.DataFrame({"f1": [1, 2, 3, 4], "f2": [5, 6, 7, 8], "outcome_x": [0, 1, 0, 1]})
    result = analyze_dataset(_csv(tmp_path, df))

    assert result["target_hint"]["column"] == "outcome_x"
    assert result["target_hint"]["confidence"] in ("low", "medium")


def test_no_target_when_no_low_cardinality_column(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "feature_a": list(range(100)),
            "feature_b": list(range(100, 200)),
            "feature_c": [f"id_{i}" for i in range(100)],  # 100 unique strings
        }
    )
    result = analyze_dataset(_csv(tmp_path, df))

    assert result["target_hint"]["column"] is None
    assert result["target_hint"]["confidence"] == "none"


# --------------------------------------------------------------------------- #
# Task inference                                                              #
# --------------------------------------------------------------------------- #


def test_binary_classification_inferred(tmp_path: Path) -> None:
    df = pd.DataFrame({"feature": list(range(10)), "target": [0] * 7 + [1] * 3})
    result = analyze_dataset(_csv(tmp_path, df), target_column="target")

    assert result["task_hint"]["type"] == "binary_classification"
    assert "class_balance" in result["task_hint"]


def test_regression_inferred(tmp_path: Path) -> None:
    df = pd.DataFrame({"feature": list(range(50)), "target": [i * 1.5 + 0.3 for i in range(50)]})
    result = analyze_dataset(_csv(tmp_path, df), target_column="target")

    assert result["task_hint"]["type"] == "regression"
    assert "target_stats" in result["task_hint"]


def test_multiclass_inferred(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "feature": list(range(60)),
            "label": [i % 4 for i in range(60)],  # 4 classes
        }
    )
    result = analyze_dataset(_csv(tmp_path, df), target_column="label")

    assert result["task_hint"]["type"] == "multiclass_classification"
    assert result["task_hint"]["n_classes"] == 4


# --------------------------------------------------------------------------- #
# Warnings                                                                    #
# --------------------------------------------------------------------------- #


def test_class_imbalance_warning(tmp_path: Path) -> None:
    df = pd.DataFrame({"feature": list(range(100)), "churn": [0] * 95 + [1] * 5})
    result = analyze_dataset(_csv(tmp_path, df))
    assert any("imbalance" in w.lower() for w in result["warnings"])


def test_high_cardinality_warning(tmp_path: Path) -> None:
    # 1000 rows, 100 distinct countries (each appearing ~10 times).
    # Classified as categorical (ratio 0.1 < 0.5) with cardinality > 50.
    df = pd.DataFrame(
        {
            "country": [f"country_{i % 100}" for i in range(1000)],
            "target": [0, 1] * 500,
        }
    )
    result = analyze_dataset(_csv(tmp_path, df))
    assert any("cardinality" in w.lower() for w in result["warnings"])


def test_no_warnings_for_clean_dataset(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "age": list(range(30, 130)),
            "income": [i * 1000 for i in range(100)],
            "churn": [i % 2 for i in range(100)],  # perfectly balanced
        }
    )
    result = analyze_dataset(_csv(tmp_path, df))
    # No missing, no imbalance, no high cardinality, target detected
    assert result["warnings"] == []
