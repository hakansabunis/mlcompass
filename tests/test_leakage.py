"""Tests for the deterministic leakage-evidence collector."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlcompass.tools.leakage import detect_leakage

# --------------------------------------------------------------------------- #
# Numeric binary target — the Telco / Kaggle Titanic shape                    #
# --------------------------------------------------------------------------- #


def test_detect_numeric_binary_leak_flags_copy_feature() -> None:
    """A feature that's a near-copy of the binary 0/1 target must be flagged."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    df = pd.DataFrame(
        {
            "y_true": y,
            "y_pred": y,
            "honest_feature": rng.normal(0, 1, 200),
            "leaked_label_copy": y + rng.normal(0, 0.001, 200),
        }
    )
    ev = detect_leakage(
        df,
        y_true_col="y_true",
        y_pred_col="y_pred",
        task="binary_classification",
    )
    assert "leaked_label_copy" in ev["candidate_leak_columns"]
    assert "honest_feature" not in ev["candidate_leak_columns"]


def test_clean_predictions_have_empty_candidate_list() -> None:
    """Random predictions vs random feature ⇒ no leakage signal."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=300)
    df = pd.DataFrame(
        {
            "y_true": y,
            "y_pred": rng.integers(0, 2, size=300),
            "feature_a": rng.normal(0, 1, 300),
            "feature_b": rng.normal(0, 1, 300),
        }
    )
    ev = detect_leakage(
        df,
        y_true_col="y_true",
        y_pred_col="y_pred",
        task="binary_classification",
    )
    assert ev["candidate_leak_columns"] == []
    # Random preds vs random truth on a 50/50 split ⇒ match rate ~ 0.5.
    assert 0.35 <= (ev["perfect_match_rate"] or 0.0) <= 0.65


def test_perfect_match_rate_one_for_identical_predictions() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=100)
    df = pd.DataFrame({"y_true": y, "y_pred": y})
    ev = detect_leakage(
        df,
        y_true_col="y_true",
        y_pred_col="y_pred",
        task="binary_classification",
    )
    assert ev["perfect_match_rate"] == 1.0


# --------------------------------------------------------------------------- #
# String binary target — Iris / multiclass-shape labels                       #
# --------------------------------------------------------------------------- #


def test_detect_string_label_leak_via_string_feature() -> None:
    """A string feature that perfectly maps to a string target must be flagged."""
    rng = np.random.default_rng(0)
    y_binary = rng.integers(0, 2, size=200)
    y_str = np.where(y_binary == 1, "yes", "no")
    leak_str = np.where(y_binary == 1, "positive", "negative")
    df = pd.DataFrame(
        {
            "y_true": y_str,
            "y_pred": y_str,
            "honest": rng.normal(0, 1, 200),
            "leaked_text": leak_str,
        }
    )
    ev = detect_leakage(
        df,
        y_true_col="y_true",
        y_pred_col="y_pred",
        task="binary_classification",
    )
    assert "leaked_text" in ev["candidate_leak_columns"]


def test_correlations_use_spearman_for_string_target() -> None:
    """Non-numeric target ⇒ method is 'spearman' (factorise then Pearson)."""
    df = pd.DataFrame(
        {
            "y_true": ["yes", "no"] * 100,
            "y_pred": ["yes", "no"] * 100,
            "feature": list(range(200)),
        }
    )
    ev = detect_leakage(
        df,
        y_true_col="y_true",
        y_pred_col="y_pred",
        task="binary_classification",
    )
    assert ev["target_feature_correlations"][0]["method"] == "spearman"


# --------------------------------------------------------------------------- #
# Regression target — Ames House Prices shape                                 #
# --------------------------------------------------------------------------- #


def test_detect_regression_leak_via_monotone_feature() -> None:
    """A feature that's a monotone transform of the target must be flagged."""
    n = 200
    df = pd.DataFrame(
        {
            "y_true": [100_000 + i * 1000 for i in range(n)],
            "y_pred": [100_000 + i * 1000 for i in range(n)],
            "noise_feature": np.random.default_rng(0).normal(0, 1, n),
            "log_price": [11.5 + i * 0.01 for i in range(n)],  # monotone
        }
    )
    ev = detect_leakage(
        df,
        y_true_col="y_true",
        y_pred_col="y_pred",
        task="regression",
    )
    assert "log_price" in ev["candidate_leak_columns"]
    assert ev["target_feature_correlations"][0]["method"] == "pearson"


def test_correlations_use_pearson_for_numeric_target() -> None:
    df = pd.DataFrame(
        {
            "y_true": list(range(100)),
            "y_pred": list(range(100)),
            "x": list(range(100)),
        }
    )
    ev = detect_leakage(
        df,
        y_true_col="y_true",
        y_pred_col="y_pred",
        task="regression",
    )
    assert ev["target_feature_correlations"][0]["method"] == "pearson"


# --------------------------------------------------------------------------- #
# Sample-size guard                                                           #
# --------------------------------------------------------------------------- #


def test_small_sample_size_flagged_as_untrustworthy() -> None:
    """Below the 50-row trustworthiness threshold a note is appended."""
    df = pd.DataFrame({"y_true": [0, 1] * 10, "y_pred": [0, 1] * 10, "feature": list(range(20))})
    ev = detect_leakage(
        df,
        y_true_col="y_true",
        y_pred_col="y_pred",
        task="binary_classification",
    )
    assert ev["trustworthy_sample_size"] is False
    assert any("trustworthiness" in n.lower() for n in ev["notes"])


def test_large_sample_size_marked_trustworthy() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "y_true": rng.integers(0, 2, size=500),
            "y_pred": rng.integers(0, 2, size=500),
            "feature": rng.normal(0, 1, 500),
        }
    )
    ev = detect_leakage(
        df,
        y_true_col="y_true",
        y_pred_col="y_pred",
        task="binary_classification",
    )
    assert ev["trustworthy_sample_size"] is True
    assert all("trustworthiness" not in n.lower() for n in ev["notes"])


# --------------------------------------------------------------------------- #
# Output schema                                                               #
# --------------------------------------------------------------------------- #


def test_output_schema_contains_all_required_keys() -> None:
    df = pd.DataFrame({"y_true": [0, 1] * 50, "y_pred": [0, 1] * 50})
    ev = detect_leakage(
        df,
        y_true_col="y_true",
        y_pred_col="y_pred",
        task="binary_classification",
        suspicious_metric={"name": "auc", "value": 1.0},
    )
    expected = {
        "ok",
        "row_count",
        "trustworthy_sample_size",
        "suspicious_metric",
        "target_feature_correlations",
        "perfect_match_rate",
        "candidate_leak_columns",
        "notes",
    }
    assert expected.issubset(ev.keys())
    assert ev["suspicious_metric"] == {"name": "auc", "value": 1.0}
    assert ev["row_count"] == 100


def test_correlations_capped_at_ten_entries() -> None:
    """Wide datasets must not blow up the agent prompt."""
    rng = np.random.default_rng(0)
    n = 100
    cols = {"y_true": rng.integers(0, 2, size=n), "y_pred": rng.integers(0, 2, size=n)}
    for i in range(30):
        cols[f"feature_{i}"] = rng.normal(0, 1, n)
    df = pd.DataFrame(cols)
    ev = detect_leakage(
        df,
        y_true_col="y_true",
        y_pred_col="y_pred",
        task="binary_classification",
    )
    assert len(ev["target_feature_correlations"]) <= 10
