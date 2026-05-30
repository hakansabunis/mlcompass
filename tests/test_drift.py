"""Tests for the drift detection tool layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlcompass.tools.drift import (
    DriftAnalysisError,
    _ks_two_sample,
    _psi_from_counts,
    detect_drift,
    load_table,
)

# --------------------------------------------------------------------------- #
# load_table                                                                  #
# --------------------------------------------------------------------------- #


def test_load_table_csv(tmp_path: Path) -> None:
    p = tmp_path / "x.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(p, index=False)
    df = load_table(p)
    assert list(df.columns) == ["a"]
    assert len(df) == 3


def test_load_table_unsupported_format_raises(tmp_path: Path) -> None:
    p = tmp_path / "x.weird"
    p.write_text("noop", encoding="utf-8")
    with pytest.raises(DriftAnalysisError, match="Unsupported format"):
        load_table(p)


# --------------------------------------------------------------------------- #
# PSI / KS unit checks                                                        #
# --------------------------------------------------------------------------- #


def test_psi_from_counts_returns_zero_for_identical_distributions() -> None:
    ref = np.array([100.0, 200.0, 300.0])
    cur = np.array([100.0, 200.0, 300.0])
    assert _psi_from_counts(ref, cur) == pytest.approx(0.0, abs=1e-6)


def test_psi_from_counts_grows_with_divergence() -> None:
    ref = np.array([900.0, 100.0])
    cur = np.array([100.0, 900.0])
    psi = _psi_from_counts(ref, cur)
    assert psi > 2.0  # huge swap of mass ⇒ large PSI


def test_ks_two_sample_identical_arrays_are_zero() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    stat, _ = _ks_two_sample(a, a)
    assert stat == pytest.approx(0.0, abs=1e-9)


def test_ks_two_sample_disjoint_arrays_are_one() -> None:
    a = np.array([0.0, 0.1, 0.2])
    b = np.array([10.0, 11.0, 12.0])
    stat, p = _ks_two_sample(a, b)
    assert stat == pytest.approx(1.0, abs=1e-9)
    assert p < 0.05  # strong drift, low p-value


# --------------------------------------------------------------------------- #
# detect_drift — end-to-end                                                   #
# --------------------------------------------------------------------------- #


@pytest.fixture
def stable_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    ref = pd.DataFrame(
        {
            "x": rng.normal(0, 1, 500),
            "cat": rng.choice(["a", "b", "c"], 500),
        }
    )
    cur = pd.DataFrame(
        {
            "x": rng.normal(0, 1, 500),
            "cat": rng.choice(["a", "b", "c"], 500),
        }
    )
    return ref, cur


@pytest.fixture
def drifted_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(1)
    ref = pd.DataFrame(
        {
            "x": rng.normal(0, 1, 500),
            "cat": rng.choice(["a", "b", "c"], 500),
        }
    )
    cur = pd.DataFrame(
        {
            "x": rng.normal(3, 1.5, 500),  # mean shift + variance change
            "cat": rng.choice(["a", "b", "c"], 500, p=[0.05, 0.05, 0.9]),
        }
    )
    return ref, cur


def test_detect_drift_stable_pair_returns_low_psi(
    stable_pair: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    ref, cur = stable_pair
    result = detect_drift(ref, cur)
    assert result["verdict"]["status"] == "stable"
    assert result["verdict"]["retrain_recommended"] is False
    assert result["aggregate"]["mean_psi"] < 0.10


def test_detect_drift_drifted_pair_triggers_major(
    drifted_pair: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    ref, cur = drifted_pair
    result = detect_drift(ref, cur)
    assert result["verdict"]["status"] == "major_drift"
    assert result["verdict"]["retrain_recommended"] is True
    assert result["aggregate"]["max_psi"] > 0.20


def test_detect_drift_top_drifted_is_sorted(
    drifted_pair: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    ref, cur = drifted_pair
    result = detect_drift(ref, cur, top_n=2)
    top = result["top_drifted"]
    assert len(top) <= 2
    psis = [t["psi"] for t in top]
    assert psis == sorted(psis, reverse=True)


def test_detect_drift_respects_features_filter(
    drifted_pair: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    ref, cur = drifted_pair
    result = detect_drift(ref, cur, features=["x"])
    feature_names = [r["feature"] for r in result["feature_results"]]
    assert feature_names == ["x"]


def test_detect_drift_no_overlap_raises(tmp_path: Path) -> None:
    a = pd.DataFrame({"alpha": [1, 2]})
    b = pd.DataFrame({"beta": [3, 4]})
    with pytest.raises(DriftAnalysisError, match="No overlapping columns"):
        detect_drift(a, b)


def test_detect_drift_handles_string_dtype() -> None:
    """pandas 2.x StringDtype should be classified as categorical, not skipped."""
    ref = pd.DataFrame({"label": pd.array(["a", "b", "c"] * 100, dtype="string")})
    cur = pd.DataFrame({"label": pd.array(["c"] * 300, dtype="string")})
    result = detect_drift(ref, cur)
    label_row = next(r for r in result["feature_results"] if r["feature"] == "label")
    assert label_row["kind"] == "categorical"
    assert label_row["psi"] is not None
    assert label_row["severity"] in {"moderate", "major"}


def test_detect_drift_small_sample_emits_warning() -> None:
    ref = pd.DataFrame({"x": [1.0, 2.0, 3.0]})  # below the 30-sample threshold
    cur = pd.DataFrame({"x": [4.0, 5.0, 6.0]})
    result = detect_drift(ref, cur)
    assert any("small sample" in w for w in result["warnings"])


def test_detect_drift_constant_reference_does_not_crash() -> None:
    """Quantile binning on a constant column must fall back gracefully."""
    ref = pd.DataFrame({"x": [5.0] * 100})
    cur = pd.DataFrame({"x": [5.0] * 100})
    result = detect_drift(ref, cur)
    # No crash + reasonable status (stable since both sides are identical).
    assert result["verdict"]["status"] in {"stable", "unknown"}
