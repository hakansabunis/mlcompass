"""FabBench leak injectors — the Phase-2 instrument (Q1_ROADMAP.md, R1/R2).

Each injector plants ONE controlled leakage pattern into a predictions frame
and returns the name of the planted column (the completeness anchor), or None
for patterns that leak through the prediction channel instead of a column.
The evidence is then produced by the SHIPPED ``detect_leakage`` — the
benchmark narrates exactly the evidence shape the product surfaces.

The six patterns cover distinct mechanisms (not six flavors of one leak):

  exact_copy     — feature IS the target (Pearson 1.0). The trivial floor.
  noisy_proxy    — target + small noise (rho ~0.995). Near-copy that a lazy
                   threshold could miss; linear mechanism.
  monotone_log   — log-transformed target (Pearson ~0.94, Spearman 1.0).
                   The June 2026 task: increasing nonlinear monotone.
  inverse_target — 1/(target+c) (Pearson strongly NEGATIVE, Spearman -1).
                   Decreasing monotone; exercises the abs()/two-sided path.
  binned_target  — target quantile-binned to group means. Models a target
                   encoding computed on train+test (the classic CV leak):
                   a step function of y, discrete, Spearman ~1.
  contamination  — NO leak column: y_pred equals y_true exactly on ~97% of
                   rows (memorized duplicate rows). Leaks through the
                   perfect-match channel (>= 0.95, the contract's committable
                   threshold); the evidence has no candidate column, so the
                   narrator faces an anchor-free shape.

Design rule: every injector is deterministic given (df, target, seed) and
must be reliably detectable by ``detect_leakage`` at its shipped thresholds
(candidate |rho| >= 0.99, or perfect-match >= 0.95) — verified by tests with
no API calls. Deviation note (analysis_plan.md §8): the roadmap's tentative
"derived-ratio" pattern was replaced by ``inverse_target`` because a ratio's
correlation with the target depends on the divisor's variance and cannot be
guaranteed to cross the detector threshold on arbitrary datasets;
``inverse_target`` tests the same "derived quantity" idea deterministically.
"""

from __future__ import annotations

from typing import Any, Protocol

__all__ = [
    "ANCHOR_SUFFIX",
    "CROWDED_ANCHOR",
    "CROWDED_ANCHOR_R",
    "CROWDED_BAND",
    "CROWDED_N",
    "CROWDED_SEED",
    "INJECTORS",
    "crowded_frame",
    "expected_anchor",
    "inject",
    "list_injectors",
]


class _Injector(Protocol):
    def __call__(self, df: Any, y: Any, target: str, rng: Any) -> str | None: ...


# A3.10 (analysis_plan.md §8): injected columns carry dataset-plausible
# NEUTRAL names. The June 2026 runs used `*_leak` names, which telegraph the
# answer to the narrator (a salience confound disclosed in Threats to
# Validity); the campaign's anchors must not label themselves.
ANCHOR_SUFFIX: dict[str, str] = {
    "exact_copy": "adj",
    "noisy_proxy": "est",
    "monotone_log": "idx",
    "inverse_target": "norm",
    "binned_target": "grp",
}


def expected_anchor(target: str, injector: str) -> str | None:
    """The column name an injector plants (None for contamination).

    Single source of truth for verify scripts and tests.
    """
    suffix = ANCHOR_SUFFIX.get(injector)
    return f"{target}_{suffix}" if suffix else None


def _exact_copy(df: Any, y: Any, target: str, rng: Any) -> str | None:
    col = expected_anchor(target, "exact_copy")
    df[col] = y
    return col


def _noisy_proxy(df: Any, y: Any, target: str, rng: Any) -> str | None:
    # sigma_noise = 0.1 * sigma_y  ->  rho = 1/sqrt(1.01) ~ 0.995 > threshold.
    col = expected_anchor(target, "noisy_proxy")
    scale = float(y.std()) or 1.0
    df[col] = y + rng.normal(0.0, 0.1 * scale, len(y))
    return col


def _monotone_log(df: Any, y: Any, target: str, rng: Any) -> str | None:
    import numpy as np

    col = expected_anchor(target, "monotone_log")
    df[col] = np.log(y - y.min() + 1.0) + rng.normal(0.0, 0.01, len(y))
    return col


def _inverse_target(df: Any, y: Any, target: str, rng: Any) -> str | None:
    col = expected_anchor(target, "inverse_target")
    df[col] = 1.0 / (y - y.min() + 1.0)
    return col


def _binned_target(df: Any, y: Any, target: str, rng: Any) -> str | None:
    import pandas as pd

    col = expected_anchor(target, "binned_target")
    bins = pd.qcut(pd.Series(y), q=20, labels=False, duplicates="drop")
    df[col] = pd.Series(y).groupby(bins).transform("mean").to_numpy()
    return col


def _contamination(df: Any, y: Any, target: str, rng: Any) -> str | None:
    # Overwrite y_pred AFTER the caller builds its default predictions: the
    # model "memorized" ~97% of rows (train/test duplication), so y_pred ==
    # y_true exactly there. No column is planted; the leak shows up in the
    # perfect-match rate (>= 0.95 keeps the verdict committable under the
    # contract's rule 4) and the evidence has no anchor.
    n = len(y)
    memorized = rng.random(n) < 0.97
    preds = df["y_pred"].to_numpy(copy=True)
    preds[memorized] = y[memorized]
    df["y_pred"] = preds
    return None


# --------------------------------------------------------------------------- #
# Frozen instance #14 — synthetic_crowded (analysis_plan.md A3.4)             #
# --------------------------------------------------------------------------- #
#
# The value-channel stressor: ten features whose correlations sit in a narrow
# 0.900-0.924 band spaced exactly 0.003 apart, plus ONE anchor at 0.995. Any
# specific value restated from prose is easy to garble (neighboring values
# are nearly identical), while entity fabrication is unaffected — the frame
# stresses the claims channel without touching the enum domain.
#
# Names are dataset-plausible and neutral per A3.10: the anchor does not
# label itself.

CROWDED_SEED = 0
CROWDED_N = 1200
CROWDED_ANCHOR = "sensor_ref"
CROWDED_ANCHOR_R = 0.995
CROWDED_BAND: tuple[float, ...] = tuple(round(0.900 + 0.003 * i, 3) for i in range(9))


def crowded_frame(seed: int = CROWDED_SEED, n: int = CROWDED_N) -> tuple[Any, str]:
    """The frozen ``synthetic_crowded`` predictions frame (instance #14).

    Sample Pearson correlations are constructed EXACTLY (Gram-Schmidt: each
    feature is r*z + sqrt(1-r^2)*u with u orthogonalized against the
    standardized target in-sample), so the evidence values land on the frozen
    grid to ~1e-12 — the 0.003 spacing survives with no rounding luck. The
    rank (Spearman) correlation of such a mix is strictly below its Pearson
    value, so the shipped detector's max(pearson, spearman) reports the exact
    constructed numbers.

    Returns (df, target_column); ``df`` already carries ``y_pred``.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    y = rng.normal(50.0, 10.0, n)
    z = (y - y.mean()) / y.std()

    def _unit_orthogonal(e: Any) -> Any:
        e = e - e.mean()
        e = e - (e @ z) / (z @ z) * z
        return e / e.std()

    df = pd.DataFrame({"y_true": y})
    for name, r in [(CROWDED_ANCHOR, CROWDED_ANCHOR_R)] + [
        (f"sensor_{i:02d}", r) for i, r in enumerate(CROWDED_BAND)
    ]:
        u = _unit_orthogonal(rng.normal(0.0, 1.0, n))
        df[name] = r * z + float(np.sqrt(1.0 - r * r)) * u
    df["y_pred"] = y + rng.normal(0.0, 0.001 * y.std(), n)
    return df, "y_true"


INJECTORS: dict[str, _Injector] = {
    "exact_copy": _exact_copy,
    "noisy_proxy": _noisy_proxy,
    "monotone_log": _monotone_log,
    "inverse_target": _inverse_target,
    "binned_target": _binned_target,
    "contamination": _contamination,
}


def list_injectors() -> list[str]:
    return sorted(INJECTORS)


def inject(df: Any, y: Any, target: str, injector: str, rng: Any) -> str | None:
    """Apply the named injector in place; return the anchor column (or None).

    ``df`` must already carry a ``y_pred`` column (the near-perfect
    predictions); ``contamination`` rewrites part of it, the column-planting
    injectors leave it alone.
    """
    if injector not in INJECTORS:
        raise SystemExit(f"Unknown --injector '{injector}'. Options: {', '.join(list_injectors())}")
    return INJECTORS[injector](df, y, target, rng)
