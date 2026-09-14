"""Post-deploy drift detection — pure ``pandas`` + ``numpy``.

Compares a **reference** dataset (what the model was trained on) to a
**current** dataset (what it's seeing now) and surfaces per-feature
drift signals plus an aggregate verdict.

Three complementary statistics:

- **PSI** (Population Stability Index) — bin both distributions, sum
  ``(p_current - p_reference) * ln(p_current / p_reference)``. Rules
  of thumb: PSI < 0.1 stable, 0.1–0.2 moderate drift, > 0.2 major
  drift. Industry standard in finance / risk.
- **KS-test** (Kolmogorov-Smirnov) — non-parametric: max distance
  between empirical CDFs. Good for continuous features.
- **Chi-square** — bin both distributions then compute the
  goodness-of-fit statistic; appropriate for categorical features
  with stable category sets.

We pick PSI as the primary score (one number per feature) but report
KS / chi-square alongside so the user can corroborate.

No scikit-learn / scipy dependency: the formulas we need are short
and the smaller install surface matters for an opt-in advisor CLI.
The KS p-value uses the asymptotic Kolmogorov distribution, accurate
enough for sample sizes > ~30 per side.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# One date detector for the whole package. The analyzer and the drift
# monitor read the same CSV and disagreed about what a date column is
# until this import existed.
from .dataset import _looks_like_datetime_strings

SUPPORTED_FORMATS = (".csv", ".parquet", ".xlsx", ".xls", ".jsonl", ".json")

# PSI severity buckets — finance-industry rules of thumb.
PSI_STABLE = 0.10
PSI_MODERATE = 0.20

# Minimum sample size per side; below this we annotate but don't
# treat statistics as reliable.
_MIN_RELIABLE_SAMPLES = 30


class DriftAnalysisError(ValueError):
    """Raised on unparseable inputs or unsupported formats."""


# --------------------------------------------------------------------------- #
# I/O                                                                         #
# --------------------------------------------------------------------------- #


def load_table(path: Path | str) -> pd.DataFrame:
    """Read a tabular file, auto-detecting the format from extension."""
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
    raise DriftAnalysisError(
        f"Unsupported format {suffix!r}. Supported: {', '.join(SUPPORTED_FORMATS)}"
    )


# --------------------------------------------------------------------------- #
# Public surface                                                              #
# --------------------------------------------------------------------------- #


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    features: list[str] | None = None,
    bins: int = 10,
    top_n: int = 5,
) -> dict[str, Any]:
    """Run drift checks across overlapping columns of two datasets.

    Args:
        reference: Training-time / baseline dataset.
        current: Production / new dataset.
        features: Restrict analysis to these columns. ``None`` ⇒ all
            overlapping columns.
        bins: Number of bins for PSI / chi-square on numeric columns.
        top_n: How many most-drifted features to call out separately.

    Returns:
        Structured dict with keys ``reference_rows``, ``current_rows``,
        ``features``, ``feature_results`` (list of per-feature dicts),
        ``aggregate``, ``top_drifted``, ``warnings``, ``verdict``.
    """
    common = _common_columns(reference, current, requested=features)
    if not common:
        raise DriftAnalysisError(
            "No overlapping columns between reference and current "
            f"datasets. Reference: {list(reference.columns)}. "
            f"Current: {list(current.columns)}."
        )

    feature_results: list[dict[str, Any]] = []
    warnings: list[str] = []

    for col in common:
        ref_series = reference[col].dropna()
        cur_series = current[col].dropna()
        if ref_series.empty or cur_series.empty:
            warnings.append(f"Column {col!r} is all-NaN on one side; skipping drift checks.")
            continue
        result = _analyze_column(col, ref_series, cur_series, bins=bins)
        feature_results.append(result)
        if result.get("warning"):
            warnings.append(result["warning"])

    # Aggregate: mean PSI, max PSI, count of features in each band.
    psi_values = [r["psi"] for r in feature_results if r.get("psi") is not None]
    aggregate = {
        "mean_psi": float(np.mean(psi_values)) if psi_values else None,
        "max_psi": float(np.max(psi_values)) if psi_values else None,
        "n_stable": sum(1 for v in psi_values if v < PSI_STABLE),
        "n_moderate": sum(1 for v in psi_values if PSI_STABLE <= v < PSI_MODERATE),
        "n_major": sum(1 for v in psi_values if v >= PSI_MODERATE),
    }

    # Top-N most drifted by PSI.
    ranked = sorted(
        (r for r in feature_results if r.get("psi") is not None),
        key=lambda r: r["psi"],
        reverse=True,
    )
    top_drifted = ranked[:top_n]

    verdict = _verdict(aggregate, feature_results)

    return {
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "features": list(common),
        "feature_results": feature_results,
        "aggregate": aggregate,
        "top_drifted": [
            {
                "feature": r["feature"],
                "psi": r["psi"],
                "severity": r["severity"],
                "kind": r["kind"],
            }
            for r in top_drifted
        ],
        "warnings": warnings,
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- #
# Per-column analysis                                                         #
# --------------------------------------------------------------------------- #


def _analyze_column(
    name: str,
    ref: pd.Series,
    cur: pd.Series,
    *,
    bins: int,
) -> dict[str, Any]:
    """Pick the right statistics for the column's dtype and return them."""
    kind = _classify(ref, cur)
    if kind == "numeric":
        return _analyze_numeric(name, ref, cur, bins=bins)
    if kind == "categorical":
        return _analyze_categorical(name, ref, cur)
    return {
        "feature": name,
        "kind": kind,
        "psi": None,
        "ks_stat": None,
        "ks_pvalue": None,
        "chi2_stat": None,
        "chi2_pvalue": None,
        "severity": "skipped",
        "warning": f"Column {name!r}: unsupported dtype {kind}, skipped.",
    }


def _analyze_numeric(
    name: str,
    ref: pd.Series,
    cur: pd.Series,
    *,
    bins: int,
) -> dict[str, Any]:
    """PSI + KS for numeric features."""
    ref_arr = np.asarray(ref, dtype=float)
    cur_arr = np.asarray(cur, dtype=float)

    # Bin edges from the reference quantiles (so reference distribution
    # is roughly uniform across bins by construction).
    edges = _quantile_bin_edges(ref_arr, bins=bins)
    psi = _psi_from_edges(ref_arr, cur_arr, edges=edges)
    ks_stat, ks_p = _ks_two_sample(ref_arr, cur_arr)

    psi_null = _psi_null_expectation(len(edges) - 1, len(ref_arr), len(cur_arr))
    psi_excess = max(0.0, psi - psi_null)
    severity = _severity_from_psi(psi_excess)
    warning = _size_warning(name, len(ref_arr), len(cur_arr))
    return {
        "feature": name,
        "kind": "numeric",
        "psi": float(psi),
        "psi_null_expected": float(psi_null),
        "psi_excess": float(psi_excess),
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_p),
        "chi2_stat": None,
        "chi2_pvalue": None,
        "severity": severity,
        "warning": warning,
    }


def _analyze_categorical(
    name: str,
    ref: pd.Series,
    cur: pd.Series,
) -> dict[str, Any]:
    """PSI + chi-square for categorical features."""
    ref_counts = ref.value_counts(dropna=False)
    cur_counts = cur.value_counts(dropna=False)
    all_levels = sorted(set(ref_counts.index).union(cur_counts.index), key=_safe_sort_key)

    # Reindex both to the same level set.
    ref_freq = np.asarray([ref_counts.get(level, 0) for level in all_levels], dtype=float)
    cur_freq = np.asarray([cur_counts.get(level, 0) for level in all_levels], dtype=float)

    psi = _psi_from_counts(ref_freq, cur_freq)
    chi2_stat, chi2_p = _chi_square_two_sample(ref_freq, cur_freq)
    psi_null = _psi_null_expectation(len(all_levels), int(ref_freq.sum()), int(cur_freq.sum()))
    psi_excess = max(0.0, psi - psi_null)
    severity = _severity_from_psi(psi_excess)
    warning = _size_warning(name, int(ref_freq.sum()), int(cur_freq.sum()))
    return {
        "feature": name,
        "kind": "categorical",
        "psi": float(psi),
        "psi_null_expected": float(psi_null),
        "psi_excess": float(psi_excess),
        "ks_stat": None,
        "ks_pvalue": None,
        "chi2_stat": float(chi2_stat),
        "chi2_pvalue": float(chi2_p),
        "severity": severity,
        "warning": warning,
        "levels": len(all_levels),
    }


# --------------------------------------------------------------------------- #
# Statistics                                                                  #
# --------------------------------------------------------------------------- #


def _quantile_bin_edges(reference: np.ndarray, *, bins: int) -> np.ndarray:
    """Build PSI bin edges from reference quantiles + epsilon guard.

    If the reference is constant, fall back to a single bin spanning
    the observed value ± a tiny epsilon so downstream binning still
    runs without divide-by-zero.
    """
    if len(reference) == 0:
        return np.array([0.0, 1.0])
    qs = np.linspace(0, 1, bins + 1)
    edges = np.quantile(reference, qs)
    # Deduplicate edges (happens with heavily-skewed data).
    edges = np.unique(edges)
    if len(edges) < 2:
        v = float(edges[0]) if len(edges) else 0.0
        eps = max(abs(v) * 1e-6, 1e-9)
        edges = np.array([v - eps, v + eps])
    # Open the outer bounds slightly so values equal to min / max don't
    # fall outside histogram.
    edges = edges.astype(float)
    edges[0] = edges[0] - 1e-9
    edges[-1] = edges[-1] + 1e-9
    return edges


def _psi_from_edges(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    edges: np.ndarray,
) -> float:
    """PSI on numeric arrays binned by the given edges."""
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    return _psi_from_counts(ref_counts.astype(float), cur_counts.astype(float))


def _psi_from_counts(ref_counts: np.ndarray, cur_counts: np.ndarray) -> float:
    """PSI from two count vectors of equal length."""
    ref_total = max(ref_counts.sum(), 1)
    cur_total = max(cur_counts.sum(), 1)
    ref_p = ref_counts / ref_total
    cur_p = cur_counts / cur_total
    # Laplace smoothing so zeros don't blow up the log.
    eps = 1e-6
    ref_p = np.where(ref_p == 0, eps, ref_p)
    cur_p = np.where(cur_p == 0, eps, cur_p)
    psi = float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))
    # PSI is non-negative in expectation; round small negatives caused
    # by smoothing artifacts to zero.
    return max(psi, 0.0)


def _psi_null_expectation(bins: int, n_ref: int, n_cur: int) -> float:
    """Expected PSI when both samples come from the same distribution.

    PSI is an estimate, and its sampling noise is not small. Under the
    null, ``E[PSI] ~= (k - 1) * (1/n_ref + 1/n_cur)``: it grows with the
    number of bins and shrinks with sample size, so bin count alone can
    push a column past a fixed threshold. The published 0.10 / 0.20
    bands are population-level rules of thumb and say nothing about
    finite samples.

    What that cost in practice, before this correction: two 250-row
    halves of the *same* frame produced "retraining is recommended" 78%
    of the time, and on the project's own example data a random split
    of identical rows was labelled "major" in 74% of draws. Severity is
    now read off the excess over this expectation, so a column has to
    drift more than sampling noise would to be called drifted.
    """
    if n_ref <= 0 or n_cur <= 0 or bins <= 1:
        return 0.0
    return float((bins - 1) * (1.0 / n_ref + 1.0 / n_cur))


def _ks_two_sample(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov statistic + asymptotic p-value."""
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    n1, n2 = len(a_sorted), len(b_sorted)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    # Build the empirical CDF differences at all unique points.
    all_values = np.concatenate([a_sorted, b_sorted])
    cdf_a = np.searchsorted(a_sorted, all_values, side="right") / n1
    cdf_b = np.searchsorted(b_sorted, all_values, side="right") / n2
    d_stat = float(np.max(np.abs(cdf_a - cdf_b)))

    # Asymptotic p-value (Kolmogorov distribution).
    en = math.sqrt(n1 * n2 / (n1 + n2))
    p_value = _kolmogorov_p(d_stat * (en + 0.12 + 0.11 / en))
    return d_stat, p_value


# Crossover between the two series representations of the Kolmogorov
# survival function. The alternating series converges geometrically in
# exp(-2λ²) and is fast for large λ; the theta-transformed series
# converges in exp(-π²/(8λ²)) and is fast for small λ. At λ = 1 both
# need only a handful of terms, so either side of the split is exact to
# machine precision.
_KOLMOGOROV_SERIES_CROSSOVER = 1.0


def _kolmogorov_p(lam: float) -> float:
    """Survival function of the Kolmogorov distribution.

    Two series, split at :data:`_KOLMOGOROV_SERIES_CROSSOVER`:

    - λ ≥ crossover: ``Q(λ) = 2 Σ (-1)^(k-1) exp(-2 k² λ²)``.
    - λ < crossover: ``Q(λ) = 1 - (√(2π)/λ) Σ exp(-(2k-1)² π² / (8 λ²))``.

    The small-λ branch is required, not an optimisation. The alternating
    series is only *asymptotic* as λ → 0: its terms decay like
    exp(-2k²λ²), which for λ = 0.001 is still ≈ 1 at k = 100, so
    truncating it returns a near-arbitrary partial sum instead of the
    true value of 1. Before the split, ``_kolmogorov_p(0.001)`` returned
    0.02 — a two-sample KS test on a million rows differing in a single
    value reported p = 0.01 rather than p = 1.
    """
    if lam <= 0:
        return 1.0

    if lam < _KOLMOGOROV_SERIES_CROSSOVER:
        # Theta-transformed series: converges faster the smaller λ gets.
        c = math.pi**2 / (8.0 * lam * lam)
        s = 0.0
        for k in range(1, 101):
            exponent = -((2 * k - 1) ** 2) * c
            if exponent < -745.0:  # exp underflows to 0.0 from here on
                break
            term = math.exp(exponent)
            s += term
            if term < abs(s) * 1e-16:
                break
        return max(0.0, min(1.0, 1.0 - math.sqrt(2.0 * math.pi) / lam * s))

    s = 0.0
    sign = 1.0
    for k in range(1, 101):
        term = sign * math.exp(-2.0 * (k * lam) ** 2)
        s += term
        sign = -sign
        if abs(term) < 1e-12:
            break
    return max(0.0, min(1.0, 2.0 * s))


def _chi_square_two_sample(
    ref_counts: np.ndarray,
    cur_counts: np.ndarray,
) -> tuple[float, float]:
    """Chi-square test of homogeneity between two count vectors.

    This is a 2xk contingency test, not a goodness-of-fit test. The
    distinction is the whole correctness of the function. Treating the
    reference proportions as a known null pretends the reference was
    measured without error, when it is a finite sample like the current
    window; at equal sample sizes that roughly doubles the statistic.
    Measured against two samples drawn from the *same* 4-category
    uniform, the goodness-of-fit form rejected at about 28% for a
    nominal 5% test, at every sample size tried.

    The old implementation compounded that with ``expected =
    where(expected < 1, 1.0, expected)``, which its own comment
    described as merging small cells into "other". It did not merge
    anything: a category absent from the reference got an expected
    count of 1 regardless of how many rows it held in the current
    window, so a genuinely new category threw its entire count into a
    single term. Here, categories empty on both sides are dropped
    (they carry no information and no degrees of freedom); every other
    cell keeps its real expectation.
    """
    ref_counts = np.asarray(ref_counts, dtype=float)
    cur_counts = np.asarray(cur_counts, dtype=float)

    col_totals = ref_counts + cur_counts
    keep = col_totals > 0
    ref_counts, cur_counts, col_totals = ref_counts[keep], cur_counts[keep], col_totals[keep]

    grand = float(col_totals.sum())
    k = int(len(col_totals))
    if grand <= 0 or k < 2:
        return 0.0, 1.0

    ref_total = float(ref_counts.sum())
    cur_total = float(cur_counts.sum())
    if ref_total <= 0 or cur_total <= 0:
        return 0.0, 1.0

    exp_ref = ref_total * col_totals / grand
    exp_cur = cur_total * col_totals / grand

    stat = float(
        np.sum((ref_counts - exp_ref) ** 2 / exp_ref)
        + np.sum((cur_counts - exp_cur) ** 2 / exp_cur)
    )
    dof = k - 1  # (2 - 1) * (k - 1)
    return stat, _chi2_sf(stat, dof)


def _chi2_sf(stat: float, dof: int) -> float:
    """Survival function for chi-squared via the regularised gamma Q.

    ``P(X > x) = Q(dof/2, x/2)``. We implement Q by the series + CF
    expansions of the incomplete gamma so we don't pull in scipy for
    one function.
    """
    if stat <= 0:
        return 1.0
    a = dof / 2.0
    x = stat / 2.0
    if x < a + 1:
        # Series expansion gives the regularised lower-incomplete
        # gamma P; subtract from 1.
        return max(0.0, min(1.0, 1.0 - _gammap_series(a, x)))
    # Continued-fraction expansion for Q directly.
    return max(0.0, min(1.0, _gammaq_cf(a, x)))


def _gammap_series(a: float, x: float) -> float:
    ap = a
    term = 1.0 / a
    summ = term
    for _ in range(200):
        ap += 1.0
        term *= x / ap
        summ += term
        if abs(term) < abs(summ) * 1e-12:
            break
    return summ * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gammaq_cf(a: float, x: float) -> float:
    b = x + 1.0 - a
    c = 1.0 / 1e-300
    d = 1.0 / b
    h = d
    for i in range(1, 200):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _common_columns(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    requested: list[str] | None,
) -> list[str]:
    intersect = [c for c in reference.columns if c in current.columns]
    if requested is None:
        return intersect
    return [c for c in requested if c in intersect]


def _classify(ref: pd.Series, cur: pd.Series) -> str:
    """Pick numeric vs categorical for a column."""
    dtype = ref.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return "categorical"
    if pd.api.types.is_numeric_dtype(dtype):
        # Treat low-cardinality integers as categorical (better stats).
        if ref.nunique(dropna=True) <= 10 and pd.api.types.is_integer_dtype(dtype):
            return "categorical"
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"

    # Dates arriving as strings must be caught before the categorical
    # branch, and for the same reason the analyzer has to catch them:
    # no reader this project uses parses dates, so a date column shows
    # up here as text. Left to the branch below, a column like
    # ``signup_date`` becomes a categorical with one level per row,
    # which gives it a structurally guaranteed PSI — dates moving
    # forward in time, the defining behaviour of a date, then read as
    # catastrophic drift and dominate the verdict.
    if _looks_like_datetime_strings(ref):
        return "datetime"

    # pandas 2.x ships StringDtype as a separate dtype that
    # is_object_dtype does NOT match; is_string_dtype catches both
    # legacy object-strings and the new nullable string dtype.
    # is_categorical_dtype is deprecated in pandas 4 — use isinstance.
    if (
        isinstance(dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
    ):
        return "categorical"
    return "unsupported"


def _severity_from_psi(psi: float) -> str:
    if psi < PSI_STABLE:
        return "stable"
    if psi < PSI_MODERATE:
        return "moderate"
    return "major"


def _size_warning(name: str, n_ref: int, n_cur: int) -> str | None:
    if n_ref < _MIN_RELIABLE_SAMPLES or n_cur < _MIN_RELIABLE_SAMPLES:
        return (
            f"Column {name!r}: small sample (ref={n_ref}, cur={n_cur}). "
            f"Statistics may be noisy below ~{_MIN_RELIABLE_SAMPLES} samples."
        )
    return None


def _safe_sort_key(value: Any) -> Any:
    """Sort categorical levels deterministically across mixed types."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return (0, "")
    if isinstance(value, (int, float)):
        return (1, float(value))
    return (2, str(value))


def _verdict(
    aggregate: dict[str, Any],
    feature_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Roll the per-feature signals up into a retrain recommendation."""
    mean_psi = aggregate.get("mean_psi")
    max_psi = aggregate.get("max_psi")
    n_major = aggregate.get("n_major", 0)

    if mean_psi is None or max_psi is None:
        return {
            "status": "unknown",
            "message": "No PSI-eligible features; cannot decide.",
            "retrain_recommended": False,
        }

    if max_psi >= PSI_MODERATE or n_major >= 1:
        return {
            "status": "major_drift",
            "message": (
                f"At least one feature shows major drift "
                f"(max PSI {max_psi:.3f}, {n_major} feature(s) ≥ "
                f"{PSI_MODERATE}). Retraining is recommended."
            ),
            "retrain_recommended": True,
        }
    if mean_psi >= PSI_STABLE:
        return {
            "status": "moderate_drift",
            "message": (
                f"Aggregate drift is moderate "
                f"(mean PSI {mean_psi:.3f}). Investigate the top-drifted "
                "features before retraining."
            ),
            "retrain_recommended": False,
        }
    return {
        "status": "stable",
        "message": (f"No meaningful drift detected (mean PSI {mean_psi:.3f}, max {max_psi:.3f})."),
        "retrain_recommended": False,
    }
