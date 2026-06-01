"""Tests for the deterministic evaluation tool layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlcompass.tools.evaluation import (
    DEFAULT_THRESHOLD_GRID,
    EvaluationError,
    evaluate,
    load_results,
)

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _binary_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """A synthetic, almost-perfect binary classifier output."""
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n)
    # Probability skewed toward the correct class with some noise.
    y_prob = np.where(
        y_true == 1,
        rng.uniform(0.55, 0.99, size=n),
        rng.uniform(0.01, 0.45, size=n),
    )
    y_pred = (y_prob >= 0.5).astype(int)
    return pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob})


def _regression_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, n)
    y_true = 2.5 * x + 3
    y_pred = y_true + rng.normal(0, 1.0, size=n)
    return pd.DataFrame({"y_true": y_true, "y_pred": y_pred})


# --------------------------------------------------------------------------- #
# load_results                                                                #
# --------------------------------------------------------------------------- #


def test_load_results_csv(tmp_path: Path) -> None:
    p = tmp_path / "results.csv"
    p.write_text("y_true,y_pred\n1,1\n0,1\n", encoding="utf-8")
    df = load_results(p)
    assert list(df.columns) == ["y_true", "y_pred"]
    assert len(df) == 2


def test_load_results_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "results.jsonl"
    p.write_text(
        '{"y_true": 1, "y_pred": 1}\n{"y_true": 0, "y_pred": 1}\n',
        encoding="utf-8",
    )
    df = load_results(p)
    assert len(df) == 2


def test_load_results_unsupported_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.xyz"
    p.write_text("anything", encoding="utf-8")
    with pytest.raises(EvaluationError, match="Unsupported"):
        load_results(p)


# --------------------------------------------------------------------------- #
# evaluate — column detection                                                 #
# --------------------------------------------------------------------------- #


def test_evaluate_auto_detects_columns() -> None:
    df = _binary_df()
    result = evaluate(df)
    assert result["columns"]["y_true"] == "y_true"
    assert result["columns"]["y_pred"] == "y_pred"
    assert result["columns"]["y_prob"] == "y_prob"


def test_evaluate_raises_without_truth() -> None:
    df = pd.DataFrame({"foo": [1, 2], "bar": [3, 4]})
    with pytest.raises(EvaluationError, match="ground-truth"):
        evaluate(df)


def test_evaluate_raises_without_prediction() -> None:
    df = pd.DataFrame({"y_true": [0, 1, 0, 1]})
    with pytest.raises(EvaluationError, match="prediction"):
        evaluate(df)


# --------------------------------------------------------------------------- #
# Binary classification                                                       #
# --------------------------------------------------------------------------- #


def test_binary_returns_expected_top_keys() -> None:
    result = evaluate(_binary_df())
    for key in (
        "task",
        "rows",
        "columns",
        "metrics",
        "confusion_matrix",
        "threshold_sweep",
        "best_threshold",
        "hard_examples",
        "warnings",
    ):
        assert key in result
    assert result["task"] == "binary_classification"


def test_binary_metric_block_has_expected_keys() -> None:
    result = evaluate(_binary_df())
    for key in ("accuracy", "precision", "recall", "f1", "auc"):
        assert key in result["metrics"]


def test_binary_auc_near_one_for_good_classifier() -> None:
    result = evaluate(_binary_df())
    assert result["metrics"]["auc"] > 0.9


def test_binary_confusion_matrix_counts_total() -> None:
    df = _binary_df()
    result = evaluate(df)
    cm = result["confusion_matrix"]
    assert cm["tp"] + cm["fp"] + cm["fn"] + cm["tn"] == len(df)


def test_binary_threshold_sweep_covers_grid() -> None:
    result = evaluate(_binary_df())
    thresholds = [row["threshold"] for row in result["threshold_sweep"]]
    assert thresholds == list(DEFAULT_THRESHOLD_GRID)


def test_binary_best_threshold_maximises_f1() -> None:
    result = evaluate(_binary_df())
    sweep = result["threshold_sweep"]
    expected_best_f1 = max(row["f1"] for row in sweep)
    assert result["best_threshold"]["f1"] == expected_best_f1


def test_binary_hard_examples_capped_at_k() -> None:
    result = evaluate(_binary_df(n=20), hard_examples_k=3)
    assert len(result["hard_examples"]) <= 3


def test_binary_hard_examples_have_required_fields() -> None:
    result = evaluate(_binary_df(n=20), hard_examples_k=3)
    sample = result["hard_examples"][0]
    for key in ("row_index", "true_label", "predicted_label", "probability", "row"):
        assert key in sample


def test_binary_imbalanced_warning_fires() -> None:
    # 95% negative, model predicts mostly negative — recall is low
    rng = np.random.default_rng(0)
    n = 200
    y_true = (rng.random(n) < 0.05).astype(int)
    y_prob = rng.uniform(0, 0.4, size=n)  # mostly predicts 0
    y_pred = (y_prob >= 0.5).astype(int)  # almost always 0
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob})
    result = evaluate(df)
    assert any("imbalance" in w.lower() for w in result["warnings"])


def test_binary_works_without_probability_column() -> None:
    df = _binary_df()[["y_true", "y_pred"]]
    result = evaluate(df)
    assert "auc" not in result["metrics"]
    assert result["threshold_sweep"] == []
    assert result["best_threshold"] is None


def test_binary_explicit_columns_override_detection() -> None:
    df = _binary_df().rename(columns={"y_true": "actual", "y_pred": "predicted"})
    df = df.rename(columns={"y_prob": "score"})
    result = evaluate(df, y_true_col="actual", y_pred_col="predicted", y_prob_col="score")
    assert result["columns"]["y_true"] == "actual"
    assert result["columns"]["y_pred"] == "predicted"
    assert result["columns"]["y_prob"] == "score"


# --------------------------------------------------------------------------- #
# Multiclass classification                                                   #
# --------------------------------------------------------------------------- #


def test_multiclass_auto_inferred_from_label_count() -> None:
    df = pd.DataFrame(
        {
            "y_true": ["cat", "dog", "fish", "cat", "dog", "fish"] * 5,
            "y_pred": ["cat", "dog", "fish", "dog", "dog", "fish"] * 5,
        }
    )
    result = evaluate(df)
    assert result["task"] == "multiclass_classification"
    assert {pc["label"] for pc in result["per_class"]} == {"cat", "dog", "fish"}


def test_multiclass_metrics_have_expected_keys() -> None:
    df = pd.DataFrame(
        {
            "y_true": [0, 1, 2, 0, 1, 2] * 10,
            "y_pred": [0, 1, 2, 1, 1, 2] * 10,
        }
    )
    result = evaluate(df)
    for key in ("accuracy", "macro_f1", "weighted_f1"):
        assert key in result["metrics"]


def test_multiclass_confusion_matrix_is_square() -> None:
    df = pd.DataFrame(
        {
            "y_true": [0, 1, 2, 0, 1, 2] * 10,
            "y_pred": [0, 1, 2, 1, 1, 2] * 10,
        }
    )
    result = evaluate(df)
    matrix = result["confusion_matrix"]
    n_labels = len(result["labels"])
    assert len(matrix) == n_labels
    assert all(len(row) == n_labels for row in matrix)


def test_multiclass_weak_class_warning_fires() -> None:
    # Three classes so the task is inferred as multiclass; class "a" is
    # never recovered (always predicted as "b") so its F1 is 0.
    df = pd.DataFrame(
        {
            "y_true": ["a"] * 10 + ["b"] * 10 + ["c"] * 10,
            "y_pred": ["b"] * 10 + ["b"] * 10 + ["c"] * 10,
        }
    )
    result = evaluate(df)
    assert result["task"] == "multiclass_classification"
    assert any("f1 < 0.3" in w.lower() for w in result["warnings"])


# --------------------------------------------------------------------------- #
# Regression                                                                  #
# --------------------------------------------------------------------------- #


def test_regression_auto_inferred_from_continuous_target() -> None:
    df = _regression_df()
    result = evaluate(df)
    assert result["task"] == "regression"
    for key in ("mae", "rmse", "r2"):
        assert key in result["metrics"]


def test_regression_r2_close_to_one_for_clean_fit() -> None:
    df = _regression_df(n=200, seed=1)
    result = evaluate(df)
    assert result["metrics"]["r2"] > 0.9


def test_regression_bias_warning_fires_for_biased_residuals() -> None:
    # All predictions are 5 below the truth → strong negative residual bias
    df = pd.DataFrame({"y_true": np.arange(100.0), "y_pred": np.arange(100.0) - 5})
    result = evaluate(df)
    assert any("bias" in w.lower() for w in result["warnings"])


def test_regression_hard_examples_have_residuals() -> None:
    df = _regression_df(n=50, seed=2)
    result = evaluate(df, hard_examples_k=5)
    assert len(result["hard_examples"]) == 5
    for sample in result["hard_examples"]:
        assert "residual" in sample
        assert "y_true" in sample and "y_pred" in sample


# --------------------------------------------------------------------------- #
# Explicit task override                                                      #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Suspicious-metrics ("too good to be true") warnings                         #
# --------------------------------------------------------------------------- #


def test_binary_perfect_metrics_warn_about_leakage() -> None:
    n = 200
    rng = np.random.default_rng(7)
    y_true = rng.integers(0, 2, size=n)
    # Probability == label → perfect AUC, perfect threshold split.
    y_prob = y_true.astype(float)
    y_pred = y_true
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob})
    result = evaluate(df)
    assert any("leakage" in w.lower() for w in result["warnings"])


def test_binary_suspicious_warning_quiet_on_tiny_test_set() -> None:
    n = 20  # below the threshold (50) — must NOT fire
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 2, size=n)
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_true, "y_prob": y_true.astype(float)})
    result = evaluate(df)
    assert not any("leakage" in w.lower() for w in result["warnings"])


def test_multiclass_perfect_classifier_warns() -> None:
    df = pd.DataFrame(
        {
            "y_true": ["a"] * 20 + ["b"] * 20 + ["c"] * 20,
            "y_pred": ["a"] * 20 + ["b"] * 20 + ["c"] * 20,
        }
    )
    result = evaluate(df)
    assert any("leakage" in w.lower() for w in result["warnings"])


def test_regression_perfect_fit_warns() -> None:
    # 100 rows, prediction == truth → R² == 1.0
    x = np.linspace(0, 10, 100)
    df = pd.DataFrame({"y_true": x, "y_pred": x})
    result = evaluate(df)
    assert any("leakage" in w.lower() for w in result["warnings"])


def test_task_override_forces_regression() -> None:
    # 1/0 ints, but user calls regression — should respect override
    df = pd.DataFrame({"y_true": [0, 1, 0, 1, 0] * 5, "y_pred": [0.1, 0.9, 0.2, 0.8, 0.1] * 5})
    result = evaluate(df, task="regression")
    assert result["task"] == "regression"
    assert "mae" in result["metrics"]


# --------------------------------------------------------------------------- #
# Field Test #4 — _infer_task multiclass detection                            #
# --------------------------------------------------------------------------- #


def test_field_ft4_multiclass_inferred_when_y_prob_present() -> None:
    """Penguins regression: 3-label y_true + single ``y_prob`` column
    used to be misclassified as binary because the y_prob heuristic
    short-circuited the inference. v0.7.2: y_true.nunique() > 2 wins.
    """
    n = 60
    df = pd.DataFrame(
        {
            "y_true": [0, 1, 2] * (n // 3),
            "y_pred": [0, 1, 2] * (n // 3),
            # Single "max softmax" column — looks like a binary y_prob but isn't.
            "y_prob": [0.85, 0.90, 0.80] * (n // 3),
        }
    )
    result = evaluate(df)
    assert result["task"] == "multiclass_classification"
    assert "metrics" in result


def test_field_ft4_binary_still_inferred_when_two_labels_with_y_prob() -> None:
    """Genuine binary classification with probability column must STILL
    be inferred as binary — the v0.7.2 fix mustn't regress this path."""
    rng = np.random.default_rng(0)
    n = 100
    y_true = rng.integers(0, 2, size=n)
    df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_true,
            "y_prob": np.where(y_true == 1, 0.85, 0.15),
        }
    )
    result = evaluate(df)
    assert result["task"] == "binary_classification"
    assert "auc" in result["metrics"]


def test_field_ft4_multiclass_string_labels_with_y_prob() -> None:
    """String labels + y_prob ⇒ multiclass too."""
    df = pd.DataFrame(
        {
            "y_true": ["Adelie", "Chinstrap", "Gentoo"] * 20,
            "y_pred": ["Adelie", "Chinstrap", "Gentoo"] * 20,
            "y_prob": [0.95] * 60,
        }
    )
    result = evaluate(df)
    assert result["task"] == "multiclass_classification"
